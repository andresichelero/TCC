# -*- coding: utf-8 -*-
"""
Pipeline BDA-DNN (Baseado em bda_dnn.py)

Este script foi refatorado para ser uma 'biblioteca' callable.
A lógica principal foi encapsulada na função run_bda_dnn_pipeline.

Ele depende de 'pipeline_utils.py' para:
- DataHandler (load, preprocess, split)
- Metrics (calculate_all_metrics)
- Plotting (plot_dnn_training_history, etc.)
- Constantes Globais (FS, HIGHCUT_HZ, etc.)
"""

import gc
import os
import time
import json
import sys
import pywt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Importa os utilitários compartilhados
try:
    import pipeline_utils
    from pipeline_utils import (
        DataHandler, Metrics, Plotting, NumpyEncoder,
        CLASS_NAMES, NUM_CLASSES, FS, HIGHCUT_HZ, FILTER_ORDER,
        TEST_SIZE, VAL_SIZE
    )
except ImportError:
    print("ERRO: Não foi possível importar 'pipeline_utils.py'. "
          "Certifique-se que o arquivo está no mesmo diretório.")
    sys.exit(1)


# --- Constantes Específicas do Pipeline BDA-DNN ---

# Pré-processamento
SWT_WAVELET = "db4"
SWT_LEVEL = 4

# Parâmetros da DNN para Treino Final
DNN_TRAINING_PARAMS_FINAL = {"epochs": 250, "batch_size": 16, "patience": 30}

# Parâmetros dos Otimizadores
N_AGENTS_OPTIMIZERS = 10
T_MAX_ITER_OPTIMIZERS = 100

# Parâmetros Fitness (Conforme Artigo)
ALPHA_FITNESS = 0.99
BETA_FITNESS = 0.01

# Nível de verbosidade para os otimizadores (logs)
# 0 = silencioso, 1 = básico, 2 = detalhado
VERBOSE_OPTIMIZER_LEVEL = 1 
SAVE_PLOTS_PER_RUN = True # Enable plots for individual runs


# --- Classes de Lógica Específicas do BDA-DNN ---

class FeatureExtractor:
    """
    Agrupa métodos estáticos para extração de características (SWT e estatísticas).
    (Mantido de bda_dnn.py)
    """

    @staticmethod
    def _is_valid_coeffs_array(coeffs, min_len=1, segment_idx=-1, band_name_debug="N/A"):
        if not isinstance(coeffs, np.ndarray) or coeffs.ndim == 0:
            return False
        if len(coeffs) < min_len:
            return False
        return True

    @staticmethod
    def calculate_mav(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mav_input"): return np.nan
        return np.mean(np.abs(coeffs))

    @staticmethod
    def calculate_std_dev(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_std_input"): return np.nan
        return np.std(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

    @staticmethod
    def calculate_skewness(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_skew_input"): return np.nan
        if np.all(coeffs == coeffs[0]):
            return 0.0
        val = skew(coeffs, bias=False)
        return val

    @staticmethod
    def calculate_kurtosis_val(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=4, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_kurt_input"): return np.nan
        if np.all(coeffs == coeffs[0]):
            return np.nan
        val = kurtosis(coeffs, fisher=False, bias=False)
        return val

    @staticmethod
    def calculate_rms(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_rms_input"): return np.nan
        return np.sqrt(np.mean(coeffs**2))

    @staticmethod
    def calculate_mavs_ratio(coeffs_band_numerator, mav_denominator_band, segment_idx=-1, band_name_debug="N/A"):
        is_coeffs_band_valid = FeatureExtractor._is_valid_coeffs_array(coeffs_band_numerator, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num")
        if not is_coeffs_band_valid or np.isnan(mav_denominator_band):
            return np.nan
        mav_coeffs_numerator = FeatureExtractor.calculate_mav(coeffs_band_numerator, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num_mav")
        if np.isnan(mav_coeffs_numerator): return np.nan
        if mav_denominator_band == 0:
            return np.nan
        return mav_coeffs_numerator / mav_denominator_band

    @staticmethod
    def calculate_activity(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_act_input"): return np.nan
        return np.var(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

    @staticmethod
    def calculate_mobility(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_input"): return np.nan
        var_coeffs = np.var(coeffs, ddof=1)
        if var_coeffs < 1e-10:
            return 0.0
        diff_coeffs = np.diff(coeffs)
        if not FeatureExtractor._is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_diff_input"):
            return np.nan
        var_diff = np.var(diff_coeffs, ddof=1)
        ratio = var_diff / var_coeffs
        if ratio < 0:
            return np.nan
        return np.sqrt(ratio)

    @staticmethod
    def calculate_complexity(coeffs, segment_idx=-1, band_name_debug="N/A"):
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_input"):
            return np.nan
        mobility_coeffs = FeatureExtractor.calculate_mobility(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_coeffs")
        if np.isnan(mobility_coeffs): return np.nan
        if mobility_coeffs < 1e-10:
            return 0.0
        diff_coeffs = np.diff(coeffs)
        if not FeatureExtractor._is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_diff_for_mob_input"):
             return np.nan
        mobility_diff = FeatureExtractor.calculate_mobility(diff_coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_diff")
        if np.isnan(mobility_diff): return np.nan
        return mobility_diff / mobility_coeffs

    feature_functions_base = {
        'MAV': calculate_mav, 'StdDev': calculate_std_dev, 'Skewness': calculate_skewness,
        'Kurtosis': calculate_kurtosis_val, 'RMS': calculate_rms, 'Activity': calculate_activity,
        'Mobility': calculate_mobility, 'Complexity': calculate_complexity,
    }

    @staticmethod
    def get_swt_subbands_recursive(signal, wavelet, current_level, max_level, band_prefix=""):
        if current_level == max_level:
            return [(band_prefix, signal)]
        coeffs_level_1 = pywt.swt(signal, wavelet, level=1, trim_approx=True, norm=True)
        cA = coeffs_level_1[0]
        cD = coeffs_level_1[1]
        subbands = []
        subbands.extend(FeatureExtractor.get_swt_subbands_recursive(cA, wavelet, current_level + 1, max_level, band_prefix + "A"))
        subbands.extend(FeatureExtractor.get_swt_subbands_recursive(cD, wavelet, current_level + 1, max_level, band_prefix + "D"))
        return subbands

    @staticmethod
    def extract_swt_features(eeg_data, wavelet='db4', level=4):
        num_segments = eeg_data.shape[0]
        original_signal_length = eeg_data.shape[1]

        signal_length_for_swt = original_signal_length
        if original_signal_length % 2 != 0:
            signal_length_for_swt = original_signal_length - 1
            if VERBOSE_OPTIMIZER_LEVEL > 1:
                print(f"Aviso: Comprimento original do sinal ({original_signal_length}) é ímpar. "
                      f"Será truncado para {signal_length_for_swt} para SWT.", flush=True)

        num_base_features = len(FeatureExtractor.feature_functions_base) # 8
        num_subbands = 2**level # 16 para level=4
        total_features_to_extract = (num_subbands * num_base_features) + (num_subbands - 1) # 143

        feature_matrix = np.full((num_segments, total_features_to_extract), np.nan, dtype=np.float32)
        feature_names = []
        ref_band_name_for_mav_ratio = 'A' * level # "AAAA"

        print(f"Iniciando extração de {total_features_to_extract} características SWT...")
        for i in tqdm(range(num_segments), desc="Extraindo Características SWT (143)", leave=False):
            signal_full = eeg_data[i, :]
            signal_to_process = signal_full[:signal_length_for_swt]

            if len(signal_to_process) < pywt.Wavelet(wavelet).dec_len:
                continue

            all_16_subbands_tuples = []
            try:
                all_16_subbands_tuples = FeatureExtractor.get_swt_subbands_recursive(signal_to_process, wavelet, 0, level)
            except Exception as e_swt_call:
                if i < 10 and VERBOSE_OPTIMIZER_LEVEL > 1: 
                    print(f"Debug (seg {i}): Erro na decomposição SWT recursiva: {e_swt_call}. Features serão NaN.", flush=True)
                continue

            if len(all_16_subbands_tuples) != num_subbands:
                continue

            mav_ref_band_value = np.nan
            for band_name_tuple, coeffs_tuple in all_16_subbands_tuples:
                if band_name_tuple == ref_band_name_for_mav_ratio:
                    mav_ref_band_value = FeatureExtractor.calculate_mav(coeffs_tuple, segment_idx=i, band_name_debug=band_name_tuple)
                    break
            
            feature_col_idx = 0
            # 1. Calcula as 8 características base (128 features)
            for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
                for feat_name_key, feat_func in FeatureExtractor.feature_functions_base.items():
                    value = feat_func(coeffs_current_band, segment_idx=i, band_name_debug=band_name)
                    feature_matrix[i, feature_col_idx] = value
                    if i == 0: feature_names.append(f"{band_name}_{feat_name_key}")
                    feature_col_idx += 1

            # 2. Calcula as 15 MAVsRatios
            for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
                if band_name == ref_band_name_for_mav_ratio:
                    continue
                value_mav_ratio = FeatureExtractor.calculate_mavs_ratio(coeffs_current_band, mav_ref_band_value, segment_idx=i, band_name_debug=band_name)
                feature_matrix[i, feature_col_idx] = value_mav_ratio
                if i == 0: feature_names.append(f"{band_name}_MAVsRatio_vs_{ref_band_name_for_mav_ratio}")
                feature_col_idx += 1
                
        if num_segments > 0:
            print(f"Matriz de características (143) extraída: {feature_matrix.shape}", flush=True)
            if np.isnan(feature_matrix).any():
                num_nan_features = np.sum(np.isnan(feature_matrix))
                total_possible_features_in_matrix = feature_matrix.size
                percent_nan = (num_nan_features / total_possible_features_in_matrix) * 100 if total_possible_features_in_matrix > 0 else 0
                print(f"Alerta: {num_nan_features} ({percent_nan:.2f}%) valores NaN encontrados na matriz de características.", flush=True)

        return feature_matrix, feature_names

class BinaryDragonflyAlgorithm:
    """
    Implementação do BDA.
    (Mantido de bda_dnn.py, com logging ajustado)
    """
    def __init__(
        self,
        N, T, dim, fitness_func, X_train_feat, y_train,
        s=0.1, a=0.1, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.85,
        tau_min=0.01, tau_max=4.0, clip_step_min=-6.0, clip_step_max=6.0,
        alpha_fitness=0.99, beta_fitness=0.01, seed=None,
        verbose_optimizer_level=0, stagnation_limit=5, reinitialization_percent=0.7,
        c_cohesion_final=0.9, s_separation_final=0.01,
        min_features=1, max_features=None,
    ):
        self.N = N
        self.T = T
        self.dim = dim
        self.fitness_func = fitness_func
        self.X_train_feat = X_train_feat
        self.y_train = y_train
        self.alpha_fitness = alpha_fitness
        self.beta_fitness = beta_fitness
        self.s = s
        self.a = a
        self.c_cohesion = c_cohesion
        self.f_food = f_food
        self.e_enemy = e_enemy
        self.w_inertia = w_inertia
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.clip_step_min = clip_step_min
        self.clip_step_max = clip_step_max
        self.verbose_optimizer_level = verbose_optimizer_level
        self.c_cohesion_final = c_cohesion_final
        self.s_separation_final = s_separation_final
        self.mutation_boost_prob = 0.30
        self.mutation_boost_bit_prob = 0.40
        self.mutation_boost_interval = 10

        if seed is not None:
            np.random.seed(seed)

        if max_features is None:
            max_features = dim

        def create_valid_position():
            position = np.zeros(dim, dtype=np.int8)
            num_to_select = np.random.randint(min_features, max_features + 1)
            indices = np.random.choice(dim, num_to_select, replace=False)
            position[indices] = 1
            return position

        self._create_valid_position = create_valid_position
        self.min_features = min_features
        self.max_features = max_features
        self.positions = np.array([create_valid_position() for _ in range(self.N)], dtype=np.int8)
        self.steps = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1
        self.fitness_values = np.full(self.N, np.inf)
        self.food_pos = np.zeros(self.dim, dtype=int)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim, dtype=int)
        self.enemy_fitness = -np.inf
        self.convergence_curve = np.zeros(self.T)
        self.best_accuracy_curve = np.zeros(self.T)
        self.best_num_features_curve = np.zeros(self.T)
        self.solutions_history = []
        self.population_fitness_history = np.zeros((self.N, self.T))  # New: track population fitness
        self.feature_selection_history = np.zeros((self.T, self.dim))  # New: track feature selection
        self._stagnation_counter = 0
        self._last_best_fitness = np.inf
        self.stagnation_limit = stagnation_limit
        self.reinitialization_percent = reinitialization_percent

    def _initialize_population_fitness(self):
        if self.verbose_optimizer_level > 0:
            print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(
            range(self.N), desc="BDA Init Fitness",
            disable=self.verbose_optimizer_level == 0, leave=False
        ):
            results = self.fitness_func(
                self.positions[i, :], self.X_train_feat, self.y_train,
                alpha=self.alpha_fitness, beta=self.beta_fitness,
                verbose_level=0, # Nível de verbose da função fitness (0)
            )
            self.fitness_values[i] = results["fitness"]
            self.solutions_history.append((self.fitness_values[i], self.positions[i, :].copy()))
            
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
                self.best_accuracy_curve[0] = results["accuracy"]
                self.best_num_features_curve[0] = results["num_features"]
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()

        if self.convergence_curve[0] == 0:
            self.convergence_curve[0] = self.food_fitness
            
        if np.isinf(self.food_fitness) and self.verbose_optimizer_level > 0:
            print("ALERTA BDA: Nenhuma solução inicial válida encontrada!")
        if self.verbose_optimizer_level > 0:
            print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
            
        self._last_best_fitness = self.food_fitness

    def _reinitialize_worst_agents(self):
        num_to_reinitialize = int(self.N * self.reinitialization_percent)
        if num_to_reinitialize == 0: return
        worst_indices = np.argsort(self.fitness_values)[-num_to_reinitialize:] 
        for i in worst_indices:
            self.positions[i, :] = self._create_valid_position()
            self.steps[i, :] = np.random.uniform(-1, 1, size=self.dim) * 0.1
            results = self.fitness_func(
                self.positions[i, :], self.X_train_feat, self.y_train,
                alpha=self.alpha_fitness, beta=self.beta_fitness, verbose_level=0,
            )
            self.fitness_values[i] = results["fitness"]
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()

    def run(self):
        self._initialize_population_fitness()
        if np.isinf(self.food_fitness) and self.N > 0:
            if self.verbose_optimizer_level > 0: print("BDA: Otimização não pode prosseguir (fitness inicial Inf).")
            if np.sum(self.food_pos) == 0: self.food_pos = self.positions[0, :].copy()
            return self.food_pos, self.food_fitness, self.convergence_curve, None, None, None, None, None
        elif self.N == 0:
            if self.verbose_optimizer_level > 0: print("BDA: Tamanho da população é 0.")
            return np.array([]), np.inf, self.convergence_curve, None, None, None, None, None

        if self.verbose_optimizer_level > 0:
            print(f"\nIniciando otimização BDA por {self.T} iterações...")
        mutation_boost_counter = 0

        for t in tqdm(range(self.T), desc="BDA Iterations", 
                      disable=self.verbose_optimizer_level == 0, leave=False):
            ratio = t / (self.T - 1) if self.T > 1 else 1.0
            current_tau = max((1.0 - ratio) * self.tau_max + ratio * self.tau_min, 1e-5)
            current_s = self.s - t * ((self.s - self.s_separation_final) / self.T)
            current_c = self.c_cohesion + t * ((self.c_cohesion_final - self.c_cohesion) / self.T)

            for i in range(self.N):
                S_i, A_i, C_sum_Xj = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
                num_neighbors_for_A_C = 0
                for j in range(self.N):
                    if i == j: continue
                    S_i += self.positions[j, :] - self.positions[i, :]
                    A_i += self.steps[j, :]
                    C_sum_Xj += self.positions[j, :]
                    num_neighbors_for_A_C += 1
                A_i = A_i / num_neighbors_for_A_C if num_neighbors_for_A_C > 0 else np.zeros(self.dim)
                C_i = (C_sum_Xj / num_neighbors_for_A_C) - self.positions[i, :] if num_neighbors_for_A_C > 0 else np.zeros(self.dim)
                Fi = self.food_pos - self.positions[i, :]
                Ei = self.enemy_pos + self.positions[i, :]
                behavioral_sum = (current_s * S_i + self.a * A_i + current_c * C_i + self.f_food * Fi + self.e_enemy * Ei)
                current_step_velocity = np.clip(behavioral_sum + self.w_inertia * self.steps[i, :], self.clip_step_min, self.clip_step_max)
                self.steps[i, :] = current_step_velocity
                v_shaped_prob = np.abs(np.tanh(self.steps[i, :] / current_tau))
                flip_mask = np.random.rand(self.dim) < v_shaped_prob
                new_position_i = self.positions[i, :].copy()
                new_position_i[flip_mask] = 1 - new_position_i[flip_mask]
                self.positions[i, :] = new_position_i
                
                results = self.fitness_func(
                    self.positions[i, :], self.X_train_feat, self.y_train,
                    alpha=self.alpha_fitness, beta=self.beta_fitness, verbose_level=0,
                )
                current_fitness = results["fitness"]
                self.fitness_values[i] = current_fitness
                self.solutions_history.append((current_fitness, self.positions[i, :].copy()))

                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness:
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()

            if self.food_fitness < self._last_best_fitness:
                self._last_best_fitness = self.food_fitness
                self._stagnation_counter = 0
                mutation_boost_counter = 0
            else:
                self._stagnation_counter += 1
                mutation_boost_counter += 1

            if mutation_boost_counter >= self.mutation_boost_interval:
                num_agents_boost = max(1, int(self.N * self.mutation_boost_prob))
                boost_indices = np.random.choice(self.N, num_agents_boost, replace=False)
                for idx in boost_indices:
                    num_bits_boost = max(1, int(self.dim * self.mutation_boost_bit_prob))
                    bits_to_mutate = np.random.choice(self.dim, num_bits_boost, replace=False)
                    for d in bits_to_mutate:
                        s_prob = 1 / (1 + np.exp(-self.steps[idx, d] / current_tau))
                        self.positions[idx, d] = 1 if np.random.rand() < s_prob else 0
                    n_selected = np.sum(self.positions[idx, :])
                    if n_selected < self.min_features:
                        zeros = np.where(self.positions[idx, :] == 0)[0]
                        if len(zeros) > 0:
                            n_to_activate = min(len(zeros), self.min_features - n_selected)
                            to_activate = np.random.choice(zeros, n_to_activate, replace=False)
                            self.positions[idx, to_activate] = 1
                mutation_boost_counter = 0

            if self._stagnation_counter >= self.stagnation_limit:
                self._reinitialize_worst_agents()
                self._stagnation_counter = 0

            self.convergence_curve[t] = self.food_fitness
            if (self.verbose_optimizer_level > 1 and (t + 1) % 10 == 0):
                print(
                    f"BDA Iter {t+1}/{self.T} - Melhor Fitness: {self.food_fitness:.4f}"
                )
                
            best_results_this_iter = self.fitness_func(
                self.food_pos, self.X_train_feat, self.y_train,
                self.alpha_fitness, self.beta_fitness, verbose_level=0
            )
            self.best_accuracy_curve[t] = best_results_this_iter["accuracy"]
            self.best_num_features_curve[t] = best_results_this_iter["num_features"]
            
            # Store population fitness and feature selection for this iteration
            self.population_fitness_history[:, t] = self.fitness_values
            self.feature_selection_history[t, :] = self.food_pos

        n_selected_final = np.sum(self.food_pos)
        if n_selected_final < self.min_features:
            zeros = np.where(self.food_pos == 0)[0]
            if len(zeros) > 0:
                n_to_activate = min(len(zeros), self.min_features - n_selected_final)
                to_activate = np.random.choice(zeros, n_to_activate, replace=False)
                self.food_pos[to_activate] = 1
                
        if self.verbose_optimizer_level > 0:
            print(f"\nBDA Otimização Concluída. Melhor fitness: {self.food_fitness:.4f}")
            print(f"Número de features selecionadas: {np.sum(self.food_pos)} de {self.dim}")
            
        return (
            self.food_pos,
            self.food_fitness,
            self.convergence_curve,
            self.best_accuracy_curve,
            self.best_num_features_curve,
            self.solutions_history,
            self.population_fitness_history,  # New
            self.feature_selection_history,  # New
        )

class ModelBuilder:
    """Agrupa a lógica de construção do modelo DNN."""
    
    @staticmethod
    def build_dnn_model(num_selected_features, num_classes=3, jit_compile_dnn=False):
        """Constrói e compila o modelo DNN (MLP) final."""
        if num_selected_features <= 0:
            raise ValueError("O número de características selecionadas deve ser > 0.")

        model = tf.keras.Sequential(name="MLP_Classifier_Final")
        model.add(tf.keras.layers.Input(shape=(num_selected_features,), name="Input_Layer"))
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_1_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_1_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_1_Dropout"))
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_2_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_2_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_2_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_2_Dropout"))
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_3_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_3_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_3_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_3_Dropout"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

        opt = tf.keras.optimizers.Adam()
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=jit_compile_dnn,
        )
        return model

class PipelineHelpers:
    """Agrupa funções auxiliares do pipeline BDA-DNN."""

    @staticmethod
    def create_fitness_function_for_optimizer(X_train_features, y_train_labels, alpha, beta, random_seed):
        """
        Cria e retorna uma função de fitness otimizada que encapsula um KNN pré-configurado.
        """
        knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance', algorithm="kd_tree")
        
        n_folds = 10
        min_samples_per_class = np.min(np.bincount(y_train_labels))
        if min_samples_per_class < n_folds:
            if VERBOSE_OPTIMIZER_LEVEL > 0:
                print(f"Aviso Fitness: Ajustando n_folds de {n_folds} para {min_samples_per_class} (mínimo de amostras).")
            n_folds = max(2, min_samples_per_class)
            
        cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

        def evaluate_fitness_configured(binary_feature_vector, *args, **kwargs):
            selected_indices = np.where(binary_feature_vector == 1)[0]
            num_selected = len(selected_indices)
            total_features = len(binary_feature_vector)
            
            X_train_selected = X_train_features[:, selected_indices]

            try:
                accuracies = cross_val_score(
                    knn_classifier, X_train_selected, y_train_labels,
                    cv=cv_splitter, scoring="accuracy", n_jobs=-1,
                )
                mean_accuracy = np.mean(accuracies)
            except ValueError:
                mean_accuracy = 0.0

            error_rate = 1.0 - mean_accuracy
            feature_ratio = num_selected / total_features
            fitness = alpha * error_rate + beta * feature_ratio

            return {
                "fitness": fitness,
                "accuracy": mean_accuracy,
                "num_features": num_selected,
            }
        return evaluate_fitness_configured

    @staticmethod
    def train_and_evaluate_final_model(
        model_name, selected_features_vector,
        X_train_full_all_feat, y_train_full,
        X_test_all_feat, y_test,
        dnn_params, class_names,
        opt_fitness_score, plots_dir
    ):
        """Treina e avalia o modelo DNN final com um subconjunto de features."""
        print(f"\n--- Treinamento e Avaliação Final: {model_name} ---")
        selected_indices = np.where(selected_features_vector == 1)[0]
        num_selected = len(selected_indices)

        if num_selected == 0:
            print(f"ERRO: {model_name} não selecionou nenhuma feature. Avaliação abortada.")
            return None, None

        print(f"{model_name}: Selecionou {num_selected} características.")
        
        # Seleção de features
        X_train_full_selected = X_train_full_all_feat[:, selected_indices]
        X_test_selected = X_test_all_feat[:, selected_indices]
        
        tf.keras.backend.clear_session()
        final_model = ModelBuilder.build_dnn_model(
            num_selected_features=num_selected,
            num_classes=len(class_names),
            jit_compile_dnn=True,
        )
        if VERBOSE_OPTIMIZER_LEVEL > 1:
            final_model.summary()

        print(f"Iniciando treinamento final do modelo {model_name}...")
        early_stopping_final = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=dnn_params.get("patience", 30),
            restore_best_weights=True, verbose=1 if VERBOSE_OPTIMIZER_LEVEL > 0 else 0,
        )
        
        history = final_model.fit(
            X_train_full_selected,
            y_train_full,
            epochs=dnn_params.get("epochs", 150),
            batch_size=dnn_params.get("batch_size", 16),
            validation_split=0.15,
            callbacks=[early_stopping_final],
            verbose=1 if VERBOSE_OPTIMIZER_LEVEL > 0 else 0,
        )

        history_data = history.history
        if SAVE_PLOTS_PER_RUN:
            Plotting.plot_dnn_training_history(
                history_data, plots_dir, SAVE_PLOTS_PER_RUN,
                title=f"Histórico de Treino Final - {model_name}",
                filename=f"final_dnn_history_{model_name}.png",
            )

        print(f"\nAvaliando {model_name} no conjunto de teste...")
        y_pred_test_probs = final_model.predict(X_test_selected)
        y_pred_test = np.argmax(y_pred_test_probs, axis=1)

        metrics = Metrics.calculate_all_metrics(y_test, y_pred_test, class_names=class_names)
        metrics["num_selected_features"] = num_selected
        metrics["selected_feature_indices"] = selected_indices.tolist()
        metrics["fitness_score_from_optimizer"] = opt_fitness_score

        del final_model
        gc.collect()
        return metrics, history_data

# --- Função Principal do Pipeline ---

def run_bda_dnn_pipeline(run_id, base_results_dir, global_constants, random_seed_for_run, X_full_feat=None, feature_names=None, raw_labels=None):
    """
    Encapsula a execução completa do pipeline BDA-DNN para uma única execução.
    
    Args:
        run_id (int): Identificador desta execução (ex: 1, 2, ..., 50).
        base_results_dir (str): Diretório base para salvar os resultados desta execução.
        global_constants (dict): Dicionário de constantes globais (ex: BASE_DATA_DIR).
        random_seed_for_run (int): A seed a ser usada para esta execução específica.
        X_full_feat (np.ndarray): Matriz de características extraídas (se fornecida, pula extração).
        feature_names (list): Nomes das características (se fornecida).

    Returns:
        dict: Um dicionário contendo as métricas finais e resultados desta execução.
    """
    start_time_total = time.time()
    
    # 1. Configurar diretórios e seeds para esta execução
    RUN_RESULTS_DIR = os.path.join(base_results_dir, f"run_{run_id:02d}_seed_{random_seed_for_run}")
    PLOTS_DIR = os.path.join(RUN_RESULTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Define as seeds para esta execução
    np.random.seed(random_seed_for_run)
    tf.random.set_seed(random_seed_for_run)
    
    print(f"Iniciando BDA-DNN Run {run_id} (Seed: {random_seed_for_run})")
    print(f"Resultados individuais em: {RUN_RESULTS_DIR}")

    # Configure GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs disponíveis: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth habilitado para GPUs.")
        except RuntimeError as e:
            print(f"Erro ao habilitar memory growth: {e}")
    else:
        print("Nenhuma GPU encontrada. TensorFlow usará CPU.")

    # Dicionário para armazenar todos os resultados desta execução
    run_results = {
        "run_id": run_id,
        "seed": random_seed_for_run,
        "pipeline_name": "BDA_DNN"
    }

    try:
        if X_full_feat is None or feature_names is None:
            # Fallback: carregar e processar dados (não deveria acontecer)
            print("\n--- 1. Carregando Dados (Utils) ---")
            BASE_DATA_DIR = global_constants["BASE_DATA_DIR"]
            raw_data, raw_labels = DataHandler.load_bonn_data(BASE_DATA_DIR)
            
            print("\n--- 2. Pré-processando Dados (Utils) ---")
            data_processed = DataHandler.preprocess_eeg(
                raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
            )
            
            print("\n--- 3. Extraindo Características SWT ---")
            X_full_feat, feature_names = FeatureExtractor.extract_swt_features(data_processed, wavelet=SWT_WAVELET, level=SWT_LEVEL)
        else:
            # Usar dados pré-processados
            pass

        # 4. Dividir Dados (usando DataHandler do pipeline_utils)
        X_train_feat, X_val_feat, X_test_feat, y_train, y_val, y_test = (
            DataHandler.split_data(
                X_full_feat, raw_labels,
                test_size=TEST_SIZE, val_size=VAL_SIZE,
                random_state=random_seed_for_run
            )
        )
        del X_full_feat # Libera memória
        gc.collect()

        DIM_FEATURES = X_train_feat.shape[1]
        run_results["total_features_extracted"] = DIM_FEATURES

        # 6. Otimização com BDA
        print("\n\n--- 5. Otimização com BDA ---")
        start_time_bda_opt = time.time()
        
        fitness_function_for_bda = PipelineHelpers.create_fitness_function_for_optimizer(
            X_train_feat, y_train, ALPHA_FITNESS, BETA_FITNESS, random_seed_for_run
        )

        bda = BinaryDragonflyAlgorithm(
            N=N_AGENTS_OPTIMIZERS, T=T_MAX_ITER_OPTIMIZERS, dim=DIM_FEATURES,
            fitness_func=fitness_function_for_bda,
            X_train_feat=X_train_feat, y_train=y_train,
            alpha_fitness=ALPHA_FITNESS, beta_fitness=BETA_FITNESS,
            seed=random_seed_for_run,
            verbose_optimizer_level=VERBOSE_OPTIMIZER_LEVEL,
            min_features=1, max_features=None,
        )
        Sf_bda, best_fitness_bda, convergence_bda, acc_curve_bda, nfeat_curve_bda, _, pop_fitness_hist, feat_sel_hist = bda.run()
        
        run_results["bda_optimization_time_sec"] = time.time() - start_time_bda_opt
        run_results["bda_best_fitness"] = best_fitness_bda
        run_results["bda_selected_features_vector"] = Sf_bda
        run_results["bda_num_selected_features"] = int(np.sum(Sf_bda))

        if SAVE_PLOTS_PER_RUN:
            bda_diagnostic_curves = {
                "Melhor Fitness": convergence_bda,
                "Acurácia KNN (CV)": np.array(acc_curve_bda) * 100,
                "Nº de Features": nfeat_curve_bda,
            }
            Plotting.plot_optimization_diagnostics(
                bda_diagnostic_curves, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                title=f"Diagnóstico BDA - Run {run_id}",
                filename="bda_diagnostics.png",
            )
            Plotting.plot_feature_selection_heatmap(
                feat_sel_hist, feature_names, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                title=f"Frequência Seleção Features BDA - Run {run_id}",
                filename="bda_feature_selection.png",
            )
        
        del bda, fitness_function_for_bda # Libera memória
        gc.collect()
        
        # 7. Treinamento e Avaliação Final da DNN
        print("\n\n--- 6. Treinamento e Avaliação Final da DNN ---")
        start_time_dnn_train = time.time()
        
        X_train_full_feat_final = np.concatenate((X_train_feat, X_val_feat), axis=0)
        y_train_full_labels_final = np.concatenate((y_train, y_val), axis=0)
        del X_train_feat, X_val_feat # Libera memória
        gc.collect()

        model_name = f"BDA-F{run_results['bda_num_selected_features']}-Run{run_id}"
        
        final_metrics, history_data = PipelineHelpers.train_and_evaluate_final_model(
            model_name=model_name,
            selected_features_vector=Sf_bda,
            X_train_full_all_feat=X_train_full_feat_final,
            y_train_full=y_train_full_labels_final,
            X_test_all_feat=X_test_feat,
            y_test=y_test,
            dnn_params=DNN_TRAINING_PARAMS_FINAL,
            class_names=CLASS_NAMES,
            opt_fitness_score=best_fitness_bda,
            plots_dir=PLOTS_DIR
        )
        
        run_results["dnn_train_eval_time_sec"] = time.time() - start_time_dnn_train
        
        if final_metrics:
            run_results["final_metrics"] = final_metrics
            run_results["final_accuracy"] = final_metrics.get("accuracy", 0.0)
            run_results["num_selected_features"] = final_metrics.get("num_selected_features", 0)
        
        del X_train_full_feat_final, y_train_full_labels_final, X_test_feat, y_test # Libera memória
        gc.collect()

    except Exception as e:
        print(f"ERRO na execução {run_id} (BDA-DNN): {e}")
        import traceback
        traceback.print_exc()
        run_results["error"] = str(e)

    # 8. Finalização
    total_execution_time = time.time() - start_time_total
    run_results["execution_time_sec"] = total_execution_time
    print(f"BDA-DNN Run {run_id} concluída. Tempo total: {total_execution_time/60:.2f} minutos.")
    
    # Salva os resultados individuais desta execução
    results_file_path = os.path.join(RUN_RESULTS_DIR, "run_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(run_results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Erro ao salvar resultados individuais da Run {run_id}: {e}")

    return run_results

# Este arquivo não deve ser executado diretamente, mas sim importado por main.py
if __name__ == "__main__":
    print("Este é um script de biblioteca de pipeline (BDA-DNN) e deve ser chamado por 'main.py'.")
