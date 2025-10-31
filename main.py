# main.py
import gc, os, time, datetime, json, sys
import pywt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Constantes Globais ---
# Limites de features para a função fitness
MIN_FEATURES = 15
MAX_FEATURES = 28

# Diretórios
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(current_dir, "data")
BASE_RESULTS_DIR = os.path.join(current_dir, "results")
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, f"run_{run_timestamp}")
PLOTS_DIR_MAIN = os.path.join(RUN_RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR_MAIN, exist_ok=True)

# Seeds e configurações aleatórias
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configurações de plot
SAVE_PLOTS_DEFAULT = True
SAVE_PLOTS = SAVE_PLOTS_DEFAULT
PLOTS_DIR = PLOTS_DIR_MAIN # O diretório de plotagem padrão

# Configurações de treinamento avançado
ENABLE_FEATURE_COUNT_FILTER = False # Filtra candidatos por contagem de features
TARGET_FEATURE_COUNT = 19 # Contagem de features alvo se o filtro estiver ativo
FINAL_MODEL_ACCURACY_THRESHOLD = 0.95 # Limiar de acurácia para salvar um modelo final
MAX_FINAL_MODELS_TO_KEEP = 5 # Número máximo de modelos finais a salvar

# Parâmetros do Dataset e Pré-processamento
FS = 173.61 # Frequência de amostragem
HIGHCUT_HZ = 40.0 # Frequência de corte do filtro passa-baixas
FILTER_ORDER = 4 # Ordem do filtro Butterworth
SWT_WAVELET = "db4" # Wavelet para Stationary Wavelet Transform
SWT_LEVEL = 4 # Nível da decomposição SWT

# Parâmetros da Divisão de Dados
TEST_SIZE = 0.20 # Proporção do conjunto de teste
VAL_SIZE = 0.15 # Proporção do conjunto de validação

# Parâmetros da DNN para Treino Final
DNN_TRAINING_PARAMS_FINAL = {"epochs": 250, "batch_size": 16, "patience": 30}

# Parâmetros dos Otimizadores
N_AGENTS_OPTIMIZERS = 10 # Número de agentes (vaga-lumes)
T_MAX_ITER_OPTIMIZERS = 100 # Número máximo de iterações

# Parâmetros Fitness (Conforme Artigo)
ALPHA_FITNESS = 0.99 # Peso para a taxa de erro
BETA_FITNESS = 0.01 # Peso para o número de características

# Nível de verbosidade para os otimizadores (logs)
VERBOSE_OPTIMIZER_LEVEL = 1

# Nomes das classes
class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]

# --- Classes Auxiliares ---

class NumpyEncoder(json.JSONEncoder):
    """Codificador JSON customizado para lidar com tipos de dados NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Classes Principais de Lógica ---

class DataHandler:
    """Classe para manipulação de dados (carga, pré-processamento, divisão)."""
    
    @staticmethod
    def load_bonn_data(base_path):
        """
        Carrega os dados dos conjuntos A, D, E do dataset BONN.
        Args:
            base_path (str): Caminho para o diretório contendo as pastas Set A, Set D, Set E.
        Returns:
            tuple: (dados_np, rotulos_np)
                   dados_np: Array NumPy (n_segmentos, n_amostras)
                   rotulos_np: Array NumPy (n_segmentos,)
        """
        data_segments = []
        labels = []
        # Mapeamento diretório -> rótulo
        # Conjunto A (Normal): 0
        # Conjunto D (Interictal): 1
        # Conjunto E (Ictal): 2
        sets_labels = {'A': 0, 'D': 1, 'E': 2}
        expected_segment_length = 4096 # Ajustado para 4096 (ignora a primeira amostra se 4097)

        print("Iniciando carregamento dos dados...")
        for set_name, label in sets_labels.items():
            set_path = os.path.join(base_path, set_name)
            if not os.path.isdir(set_path):
                print(f"Aviso: Diretório não encontrado: {set_path}. Pulando.")
                continue

            fnames = sorted([f for f in os.listdir(set_path) if f.lower().endswith('.txt')])
            print(f"Carregando {len(fnames)} arquivos de {set_name}...")

            for fname in tqdm(fnames, desc=f"Lendo {set_name}"):
                file_path = os.path.join(set_path, fname)
                try:
                    segment_data = pd.read_csv(file_path, header=None, dtype=np.float32).values.flatten()

                    # Ajuste para arquivos com 4097 amostras (comum nesse dataset)
                    if len(segment_data) == 4097:
                        segment_data = segment_data[1:]

                    if len(segment_data) == expected_segment_length:
                        data_segments.append(segment_data)
                        labels.append(label)
                    else:
                        print(f"Aviso: Arquivo {fname} com tamanho inesperado {len(segment_data)}. Ignorando.")
                except Exception as e:
                    print(f"Erro ao carregar {fname}: {e}")

        if not data_segments:
            raise ValueError("Nenhum dado foi carregado. Verifique os caminhos e o formato dos arquivos.")

        dados_np = np.array(data_segments)
        rotulos_np = np.array(labels, dtype=int)
        print(f"Dados carregados: {dados_np.shape}, Rótulos: {rotulos_np.shape}")
        return dados_np, rotulos_np

    @staticmethod
    def preprocess_eeg(data, fs=173.61, lowcut_hz=None, highcut_hz=40.0, order=4):
        """
        Aplica filtro Butterworth e normalização Min-Max para [-1, 1] aos sinais EEG.
        Args:
            data (np.ndarray): Array de sinais EEG (n_segmentos, n_amostras).
            fs (float): Frequência de amostragem.
            lowcut_hz (float, optional): Frequência de corte inferior para passa-banda.
            highcut_hz (float): Frequência de corte superior.
            order (int): Ordem do filtro Butterworth.
        Returns:
            np.ndarray: Sinais EEG pré-processados e normalizados.
        """
        processed_data = np.zeros_like(data)
        nyq = 0.5 * fs

        if lowcut_hz is not None and highcut_hz is not None:
            low = lowcut_hz / nyq
            high = highcut_hz / nyq
            if low >= high:
                raise ValueError(f"Frequência de corte inferior ({lowcut_hz} Hz) deve ser menor que a superior ({highcut_hz} Hz).")
            if low <= 0: 
                 b, a = butter(order, high, btype='low', analog=False)
                 print(f"Aplicando filtro passa-baixas com corte em {highcut_hz} Hz (lowcut <=0).")
            else:
                b, a = butter(order, [low, high], btype='band', analog=False)
                print(f"Aplicando filtro passa-banda entre {lowcut_hz} Hz e {highcut_hz} Hz.")
        elif highcut_hz is not None:
            high = highcut_hz / nyq
            b, a = butter(order, high, btype='low', analog=False)
            print(f"Aplicando filtro passa-baixas com corte em {highcut_hz} Hz.")
        else:
            print("Nenhum filtro aplicado, pois highcut_hz não foi fornecido.")
            b, a = None, None


        print("Iniciando pré-processamento (filtragem e normalização)...")
        for i in tqdm(range(data.shape[0]), desc="Pré-processando segmentos"):
            signal = data[i, :]
            if b is not None and a is not None:
                filtered_signal = filtfilt(b, a, signal)
            else:
                filtered_signal = signal 

            # Normaliza para [-1, 1] usando MinMaxScaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()

            processed_data[i, :] = normalized_signal
        print("Pré-processamento concluído.")
        return processed_data

    @staticmethod
    def split_data(data, labels, test_size=0.15, val_size=0.15, random_state=42):
        """
        Divide os dados em conjuntos de treino, validação e teste estratificados.
        Args:
            data (np.ndarray): Dados pré-processados.
            labels (np.ndarray): Rótulos correspondentes.
            test_size (float): Proporção para o conjunto de teste.
            val_size (float): Proporção para o conjunto de validação (relativo ao total).
            random_state (int): Seed para reprodutibilidade.
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if not (0 < test_size < 1) or not (0 < val_size < 1) or not (0 < test_size + val_size < 1) :
            raise ValueError("test_size e val_size devem estar entre 0 e 1, e sua soma deve ser menor que 1.")

        # 1. Divide em treino+validação e teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # 2. Calcula o tamanho da validação relativo ao conjunto temporário (treino+validação)
        relative_val_size = val_size / (1.0 - test_size)

        # 3. Divide o conjunto temporário em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
        )

        print("Dados divididos em conjuntos de treino, validação e teste:")
        print(f"  Treino:    {X_train.shape[0]} amostras")
        print(f"  Validação: {X_val.shape[0]} amostras")
        print(f"  Teste:     {X_test.shape[0]} amostras")
        return X_train, X_val, X_test, y_train, y_val, y_test

class FeatureExtractor:
    """
    Agrupa métodos estáticos para extração de características (SWT e estatísticas).
    """

    @staticmethod
    def _is_valid_coeffs_array(coeffs, min_len=1, segment_idx=-1, band_name_debug="N/A"):
        """Verifica se o array de coeficientes é válido para cálculo."""
        if not isinstance(coeffs, np.ndarray) or coeffs.ndim == 0:
            return False
        if len(coeffs) < min_len:
            return False
        return True

    @staticmethod
    def calculate_mav(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula o Mean Absolute Value (MAV)."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mav_input"): return np.nan
        return np.mean(np.abs(coeffs))

    @staticmethod
    def calculate_std_dev(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula o Desvio Padrão (StdDev)."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_std_input"): return np.nan
        return np.std(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

    @staticmethod
    def calculate_skewness(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula a Assimetria (Skewness)."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_skew_input"): return np.nan
        if np.all(coeffs == coeffs[0]):
            return 0.0
        val = skew(coeffs, bias=False)
        return val

    @staticmethod
    def calculate_kurtosis_val(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula a Curtose (Kurtosis)."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=4, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_kurt_input"): return np.nan
        if np.all(coeffs == coeffs[0]):
            return np.nan
        val = kurtosis(coeffs, fisher=False, bias=False)
        return val

    @staticmethod
    def calculate_rms(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula o Root Mean Square (RMS)."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_rms_input"): return np.nan
        return np.sqrt(np.mean(coeffs**2))

    @staticmethod
    def calculate_mavs_ratio(coeffs_band_numerator, mav_denominator_band, segment_idx=-1, band_name_debug="N/A"):
        """Calcula a razão MAV(banda_numerador) / MAV(banda_denominador)."""
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
        """Calcula a Atividade (Variância) do sinal."""
        if not FeatureExtractor._is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_act_input"): return np.nan
        return np.var(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

    @staticmethod
    def calculate_mobility(coeffs, segment_idx=-1, band_name_debug="N/A"):
        """Calcula a Mobilidade de Hjorth."""
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
        """Calcula a Complexidade de Hjorth."""
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

    # Dicionário de funções base para as 8 características
    feature_functions_base = {
        'MAV': calculate_mav, 'StdDev': calculate_std_dev, 'Skewness': calculate_skewness,
        'Kurtosis': calculate_kurtosis_val, 'RMS': calculate_rms, 'Activity': calculate_activity,
        'Mobility': calculate_mobility, 'Complexity': calculate_complexity,
    }

    @staticmethod
    def get_swt_subbands_recursive(signal, wavelet, current_level, max_level, band_prefix=""):
        """
        Decompõe recursivamente o sinal usando SWT para obter 2^max_level sub-bandas.
        Retorna uma lista de tuplas (nome_da_banda, coeficientes_da_banda).
        """
        if current_level == max_level:
            return [(band_prefix, signal)]

        # Decomposição de nível 1
        coeffs_level_1 = pywt.swt(signal, wavelet, level=1, trim_approx=True, norm=True)
        cA = coeffs_level_1[0]
        cD = coeffs_level_1[1]

        subbands = []
        # Chamadas recursivas para bandas de Aproximação (A) e Detalhe (D)
        subbands.extend(FeatureExtractor.get_swt_subbands_recursive(cA, wavelet, current_level + 1, max_level, band_prefix + "A"))
        subbands.extend(FeatureExtractor.get_swt_subbands_recursive(cD, wavelet, current_level + 1, max_level, band_prefix + "D"))
        return subbands

    @staticmethod
    def extract_swt_features(eeg_data, wavelet='db4', level=4):
        """
        Extrai 143 características de EEG usando decomposição SWT completa de 4 níveis.
        - 16 sub-bandas são geradas (2^level).
        - 8 características base (feature_functions_base) são calculadas para cada (128 features).
        - 15 características de Razão de MAVs (MAVsRatio) são calculadas em relação à banda 'AAAA'.
        Total = 128 + 15 = 143 features.
        """
        num_segments = eeg_data.shape[0]
        original_signal_length = eeg_data.shape[1]

        # Garante comprimento par para SWT
        signal_length_for_swt = original_signal_length
        if original_signal_length % 2 != 0:
            signal_length_for_swt = original_signal_length - 1
            print(f"Aviso: Comprimento original do sinal ({original_signal_length}) é ímpar. "
                  f"Será truncado para {signal_length_for_swt} para SWT.", flush=True)

        num_base_features = len(FeatureExtractor.feature_functions_base) # 8
        num_subbands = 2**level # 16 para level=4
        total_features_to_extract = (num_subbands * num_base_features) + (num_subbands - 1) # 143

        feature_matrix = np.full((num_segments, total_features_to_extract), np.nan)
        feature_names = [] # Será populada na primeira iteração

        ref_band_name_for_mav_ratio = 'A' * level # "AAAA"

        print(f"Iniciando extração de {total_features_to_extract} características SWT...", flush=True)
        for i in tqdm(range(num_segments), desc="Extraindo Características SWT (143)"):
            signal_full = eeg_data[i, :]
            signal_to_process = signal_full[:signal_length_for_swt]

            if len(signal_to_process) < pywt.Wavelet(wavelet).dec_len:
                continue

            all_16_subbands_tuples = []
            try:
                # Gera as 16 sub-bandas (ex: AAAA, AAAD, ..., DDDD)
                all_16_subbands_tuples = FeatureExtractor.get_swt_subbands_recursive(signal_to_process, wavelet, 0, level)
            except Exception as e_swt_call:
                if i < 10: print(f"Debug (seg {i}): Erro na decomposição SWT recursiva: {e_swt_call}. Features serão NaN.", flush=True)
                continue

            if len(all_16_subbands_tuples) != num_subbands:
                continue

            # Extrair MAV da banda de referência primeiro
            mav_ref_band_value = np.nan
            for band_name_tuple, coeffs_tuple in all_16_subbands_tuples:
                if band_name_tuple == ref_band_name_for_mav_ratio:
                    mav_ref_band_value = FeatureExtractor.calculate_mav(coeffs_tuple, segment_idx=i, band_name_debug=band_name_tuple)
                    break
            
            if np.isnan(mav_ref_band_value) and i < 10:
                print(f"Debug (seg {i}): MAV da banda de referência '{ref_band_name_for_mav_ratio}' é NaN. Ratios serão NaN.", flush=True)

            feature_col_idx = 0
            # 1. Calcula as 8 características base para todas as 16 sub-bandas (128 features)
            for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
                for feat_name_key, feat_func in FeatureExtractor.feature_functions_base.items():
                    # Referencia a função estática correta
                    value = feat_func(coeffs_current_band, segment_idx=i, band_name_debug=band_name)
                    feature_matrix[i, feature_col_idx] = value
                    if i == 0: feature_names.append(f"{band_name}_{feat_name_key}")
                    feature_col_idx += 1

            # 2. Calcula as 15 MAVsRatios
            for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
                if band_name == ref_band_name_for_mav_ratio:
                    continue # Pula a banda de referência

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
        else:
            print("Nenhum segmento para processar na extração de features.", flush=True)

        return feature_matrix, feature_names

class BinaryDragonflyAlgorithm:
    def __init__(
        self,
        N,
        T,
        dim,
        fitness_func,
        X_train_feat,
        y_train,
        s=0.1,
        a=0.1,
        c_cohesion=0.7,
        f_food=1.0,
        e_enemy=1.0,
        w_inertia=0.85,
        tau_min=0.01,
        tau_max=4.0,
        clip_step_min=-6.0,
        clip_step_max=6.0,
        alpha_fitness=0.99,
        beta_fitness=0.01,
        seed=None,
        verbose_optimizer_level=0,
        stagnation_limit=5,
        reinitialization_percent=0.7,
        # Parâmetros para pesos adaptativos
        c_cohesion_final=0.9,
        s_separation_final=0.01,
        # Limites de features
        min_features=1,
        max_features=None,
    ):
        """
        Inicializa o Otimizador Binary Dragonfly Algorithm (BDA).
        Args:
            N (int): Número de agentes (população).
            T (int): Número máximo de iterações.
            dim (int): Dimensão do problema (número total de features).
            fitness_func (callable): Função para avaliar a aptidão de um vetor binário.
            X_train_feat (np.ndarray): Dados de treino (features).
            y_train (np.ndarray): Rótulos de treino.
            s (float): Peso da Separação.
            a (float): Peso do Alinhamento.
            c_cohesion (float): Peso inicial da Coesão.
            f_food (float): Peso da fonte de Comida (melhor solução).
            e_enemy (float): Peso do Inimigo (pior solução).
            w_inertia (float): Peso da Inércia (movimento anterior).
            tau_min (float): Valor mínimo para o parâmetro tau da função V-shaped.
            tau_max (float): Valor máximo para o parâmetro tau da função V-shaped.
            clip_step_min/max (float): Limites para o vetor de passo (velocidade).
            alpha_fitness (float): Peso alpha (erro) para a função de fitness.
            beta_fitness (float): Peso beta (features) para a função de fitness.
            seed (int): Seed para reprodutibilidade.
            verbose_optimizer_level (int): Nível de log.
            stagnation_limit (int): Iterações sem melhora antes de re-inicializar.
            reinitialization_percent (float): % de piores agentes a re-inicializar.
            c_cohesion_final (float): Valor final da Coesão (adaptativo).
            s_separation_final (float): Valor final da Separação (adaptativo).
            min_features (int): Mínimo de features permitidas na solução.
            max_features (int): Máximo de features permitidas na solução.
        """

        self.N = N
        self.T = T
        self.dim = dim
        self.fitness_func = fitness_func
        self.X_train_feat = X_train_feat
        self.y_train = y_train
        self.alpha_fitness = alpha_fitness
        self.beta_fitness = beta_fitness

        # Pesos BDA
        self.s = s
        self.a = a
        self.c_cohesion = c_cohesion
        self.f_food = f_food
        self.e_enemy = e_enemy
        self.w_inertia = w_inertia
        
        # Parâmetros da função de transferência
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.clip_step_min = clip_step_min
        self.clip_step_max = clip_step_max
        self.verbose_optimizer_level = verbose_optimizer_level
        self.c_cohesion_final = c_cohesion_final
        self.s_separation_final = s_separation_final

        # Parâmetros de Mutation Boost (para estagnação)
        self.mutation_boost_prob = 0.30  # 30% dos agentes
        self.mutation_boost_bit_prob = 0.40  # 40% dos bits
        self.mutation_boost_interval = 10  # a cada 10 iterações sem melhora

        if seed is not None:
            np.random.seed(seed)

        if max_features is None:
            max_features = dim

        def create_valid_position():
            """Cria uma única solução aleatória dentro dos limites de features."""
            position = np.zeros(dim, dtype=np.int8)
            num_to_select = np.random.randint(min_features, max_features + 1)
            indices = np.random.choice(dim, num_to_select, replace=False)
            position[indices] = 1
            return position

        self._create_valid_position = create_valid_position
        self.min_features = min_features
        self.max_features = max_features

        # Inicializa posições e passos
        self.positions = np.array([create_valid_position() for _ in range(self.N)], dtype=np.int8)
        self.steps = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1
        self.fitness_values = np.full(self.N, np.inf)

        # Melhor (Comida) e Pior (Inimigo)
        self.food_pos = np.zeros(self.dim, dtype=int)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim, dtype=int)
        self.enemy_fitness = -np.inf

        # Histórico
        self.convergence_curve = np.zeros(self.T)
        self.best_accuracy_curve = np.zeros(self.T)
        self.best_num_features_curve = np.zeros(self.T)
        self.solutions_history = []
        
        # Controle de Estagnação
        self._stagnation_counter = 0
        self._last_best_fitness = np.inf
        self.stagnation_limit = stagnation_limit
        self.reinitialization_percent = reinitialization_percent

        
    def _initialize_population_fitness(self):
        """Calcula o fitness inicial para toda a população."""
        if self.verbose_optimizer_level > 0:
            print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(
            range(self.N),
            desc="BDA Init Fitness",
            disable=self.verbose_optimizer_level == 0,
        ):
            results = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=1, # Nível de verbose da função fitness
            )
            self.fitness_values[i] = results["fitness"]
            self.solutions_history.append((self.fitness_values[i], self.positions[i, :].copy()))
            
            # Atualiza Comida (melhor)
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
                self.best_accuracy_curve[0] = results["accuracy"]
                self.best_num_features_curve[0] = results["num_features"]
            # Atualiza Inimigo (pior)
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()

        if self.convergence_curve[0] == 0:
            self.convergence_curve[0] = self.food_fitness
            
        if np.isinf(self.food_fitness) and self.verbose_optimizer_level > 0:
            print("ALERTA BDA: Nenhuma solução inicial válida encontrada, food_fitness é infinito!")
        if self.verbose_optimizer_level > 0:
            print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
            print(f"BDA: Pior fitness inicial (Enemy): {self.enemy_fitness:.4f}")
            
        self._last_best_fitness = self.food_fitness


    def _reinitialize_worst_agents(self):
        """Encontra os piores agentes e os substitui por novas soluções aleatórias."""
        num_to_reinitialize = int(self.N * self.reinitialization_percent)
        if num_to_reinitialize == 0:
            return

        # Pega os índices dos piores agentes
        worst_indices = np.argsort(self.fitness_values)[-num_to_reinitialize:] 

        for i in worst_indices:
            self.positions[i, :] = self._create_valid_position() # Nova posição válida
            self.steps[i, :] = np.random.uniform(-1, 1, size=self.dim) * 0.1 # Novo passo
            
            # Reavalia o fitness da nova posição
            results = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=0, # Menos verboso
            )
            self.fitness_values[i] = results["fitness"]
            
            # Atualiza Comida/Inimigo se necessário
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()


    def run(self):
        """Executa o loop principal de otimização do BDA."""
        self._initialize_population_fitness()
        
        # Validação inicial
        if np.isinf(self.food_fitness) and self.N > 0:
            if self.verbose_optimizer_level > 0:
                print("BDA: Otimização não pode prosseguir pois o fitness inicial é infinito.")
            if np.sum(self.food_pos) == 0: # Garante que não retorna vetor de zeros
                self.food_pos = self.positions[0, :].copy()
            return self.food_pos, self.food_fitness, self.convergence_curve, None, None, None
        elif self.N == 0:
            if self.verbose_optimizer_level > 0:
                print("BDA: Tamanho da população é 0. Não é possível executar.")
            return np.array([]), np.inf, self.convergence_curve, None, None, None

        if self.verbose_optimizer_level > 0:
            print(f"\nIniciando otimização BDA por {self.T} iterações...")

        mutation_boost_counter = 0

        for t in tqdm(range(self.T), desc="BDA Iterations", disable=self.verbose_optimizer_level == 0):
            if self.T > 1:
                ratio = t / (self.T - 1)
            else:
                ratio = 1.0

            # --- Atualização Adaptativa dos Parâmetros ---
            # Tau (Função V-shaped) diminui
            current_tau = (1.0 - ratio) * self.tau_max + ratio * self.tau_min
            current_tau = max(current_tau, 1e-5)
            # Separação (s) diminui
            current_s = self.s - t * ((self.s - self.s_separation_final) / self.T)
            # Coesão (c) aumenta
            current_c = self.c_cohesion + t * ((self.c_cohesion_final - self.c_cohesion) / self.T)

            for i in range(self.N):
                # --- Cálculo dos 5 Comportamentos BDA ---
                S_i = np.zeros(self.dim) # Separação
                A_i = np.zeros(self.dim) # Alinhamento
                C_sum_Xj = np.zeros(self.dim) # Coesão (soma)
                num_neighbors_for_A_C = 0
                
                for j in range(self.N):
                    if i == j:
                        continue
                    S_i += self.positions[j, :] - self.positions[i, :] # Separação
                    A_i += self.steps[j, :] # Alinhamento
                    C_sum_Xj += self.positions[j, :] # Coesão
                    num_neighbors_for_A_C += 1
                    
                if num_neighbors_for_A_C > 0:
                    A_i /= num_neighbors_for_A_C # Média do Alinhamento
                    C_i = (C_sum_Xj / num_neighbors_for_A_C) - self.positions[i, :] # Vetor Coesão
                else:
                    A_i = np.zeros(self.dim)
                    C_i = np.zeros(self.dim)
                    
                Fi = self.food_pos - self.positions[i, :] # Comida (Atração)
                Ei = self.enemy_pos + self.positions[i, :] # Inimigo (Repulsão)
                
                # Soma dos comportamentos
                behavioral_sum = (
                    current_s * S_i
                    + self.a * A_i
                    + current_c * C_i
                    + self.f_food * Fi
                    + self.e_enemy * Ei
                )
                
                # --- Atualização do Vetor de Passo (Velocidade) ---
                current_step_velocity = behavioral_sum + self.w_inertia * self.steps[i, :]
                current_step_velocity = np.clip(current_step_velocity, self.clip_step_min, self.clip_step_max)
                self.steps[i, :] = current_step_velocity
                
                # --- Atualização da Posição (Binarização) ---
                # Função de transferência V-Shaped
                v_shaped_prob = np.abs(np.tanh(self.steps[i, :] / current_tau))
                flip_mask = np.random.rand(self.dim) < v_shaped_prob
                new_position_i = self.positions[i, :].copy()
                new_position_i[flip_mask] = 1 - new_position_i[flip_mask]
                self.positions[i, :] = new_position_i
                
                # --- Avaliação do Fitness da Nova Posição ---
                results = self.fitness_func(
                    self.positions[i, :],
                    self.X_train_feat,
                    self.y_train,
                    alpha=self.alpha_fitness,
                    beta=self.beta_fitness,
                    verbose_level=0, # Fitness não verboso no loop
                )
                current_fitness = results["fitness"]
                self.fitness_values[i] = current_fitness
                self.solutions_history.append((current_fitness, self.positions[i, :].copy()))

                # Atualiza Comida e Inimigo
                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness:
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()

            # --- Verificação de Estagnação e Mutação ---
            if self.food_fitness < self._last_best_fitness:
                self._last_best_fitness = self.food_fitness
                self._stagnation_counter = 0
                mutation_boost_counter = 0
            else:
                self._stagnation_counter += 1
                mutation_boost_counter += 1

            # Mutation Boost: aplica mutação "S-shaped"
            if mutation_boost_counter >= self.mutation_boost_interval:
                num_agents_boost = max(1, int(self.N * self.mutation_boost_prob))
                boost_indices = np.random.choice(self.N, num_agents_boost, replace=False)
                for idx in boost_indices:
                    num_bits_boost = max(1, int(self.dim * self.mutation_boost_bit_prob))
                    bits_to_mutate = np.random.choice(self.dim, num_bits_boost, replace=False)
                    for d in bits_to_mutate:
                        # Função S-Shaped
                        s_prob = 1 / (1 + np.exp(-self.steps[idx, d] / current_tau))
                        self.positions[idx, d] = 1 if np.random.rand() < s_prob else 0
                    
                    # Correção: garantir limites de features após o boost
                    n_selected = np.sum(self.positions[idx, :])
                    if n_selected < self.min_features:
                        zeros = np.where(self.positions[idx, :] == 0)[0]
                        if len(zeros) > 0:
                            n_to_activate = min(len(zeros), self.min_features - n_selected)
                            to_activate = np.random.choice(zeros, n_to_activate, replace=False)
                            self.positions[idx, to_activate] = 1
                    elif n_selected > self.max_features:
                        ones = np.where(self.positions[idx, :] == 1)[0]
                        if len(ones) > 0:
                            n_to_deactivate = min(len(ones), n_selected - self.max_features)
                            to_deactivate = np.random.choice(ones, n_to_deactivate, replace=False)
                            self.positions[idx, to_deactivate] = 0
                            
                mutation_boost_counter = 0

            # Re-inicialização (se estagnação profunda)
            if self._stagnation_counter >= self.stagnation_limit:
                self._reinitialize_worst_agents()
                self._stagnation_counter = 0

            # Salva métricas da iteração
            self.convergence_curve[t] = self.food_fitness
            if (self.verbose_optimizer_level > 0 and (t + 1) % 10 == 0):
                print(
                    f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "
                    f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}"
                )
                
            # Re-avalia o melhor agente para salvar acurácia e n_features
            best_results_this_iter = self.fitness_func(
                self.food_pos,
                self.X_train_feat,
                self.y_train,
                self.alpha_fitness,
                self.beta_fitness,
                verbose_level=0
            )
            self.best_accuracy_curve[t] = best_results_this_iter["accuracy"]
            self.best_num_features_curve[t] = best_results_this_iter["num_features"]

        # --- Checagem/correção final: garantir que food_pos respeita os limites ---
        n_selected_final = np.sum(self.food_pos)
        if n_selected_final < self.min_features:
            zeros = np.where(self.food_pos == 0)[0]
            if len(zeros) > 0:
                n_to_activate = min(len(zeros), self.min_features - n_selected_final)
                to_activate = np.random.choice(zeros, n_to_activate, replace=False)
                self.food_pos[to_activate] = 1
        elif n_selected_final > self.max_features:
            ones = np.where(self.food_pos == 1)[0]
            if len(ones) > 0:
                n_to_deactivate = min(len(ones), n_selected_final - self.max_features)
                to_deactivate = np.random.choice(ones, n_to_deactivate, replace=False)
                self.food_pos[to_deactivate] = 0
                
        if self.verbose_optimizer_level > 0:
            print(f"\nBDA Otimização Concluída. Melhor fitness encontrado: {self.food_fitness:.4f}")
            num_selected_bda = np.sum(self.food_pos)
            print(f"Número de features selecionadas pelo BDA: {num_selected_bda} de {self.dim}")
            
        return (
            self.food_pos,
            self.food_fitness,
            self.convergence_curve,
            self.best_accuracy_curve,
            self.best_num_features_curve,
            self.solutions_history,
        )

class ModelBuilder:
    """Agrupa a lógica de construção do modelo DNN."""
    
    @staticmethod
    def build_dnn_model(num_selected_features, num_classes=3, jit_compile_dnn=False):
        """
        Constrói e compila o modelo DNN (MLP) final.
        Conforme o artigo: 3 camadas ocultas, 10 neurônios, ativação sigmoidal.
        Args:
            num_selected_features (int): Número de características de entrada.
            num_classes (int): Número de classes de saída (default é 3).
            jit_compile_dnn (bool): Se deve compilar com XLA (JIT).
        Returns:
            keras.Model: Modelo Keras compilado.
        """
        if num_selected_features <= 0:
            raise ValueError(
                "O número de características selecionadas deve ser maior que zero."
            )

        model = tf.keras.Sequential(name="MLP_Classifier_Final")
        model.add(tf.keras.layers.Input(shape=(num_selected_features,), name="Input_Layer"))

        # Arquitetura (3 camadas ocultas, 10 neurônios, sigmoidal)
        # Adição de BatchNormalization e Dropout para estabilidade e regularização.

        # Camada Oculta 1
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_1_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_1_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_1_Dropout"))

        # Camada Oculta 2
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_2_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_2_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_2_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_2_Dropout"))

        # Camada Oculta 3
        model.add(tf.keras.layers.Dense(10, use_bias=False, name="Hidden_Layer_3_Dense"))
        model.add(tf.keras.layers.BatchNormalization(name="Hidden_Layer_3_BN"))
        model.add(tf.keras.layers.Activation("sigmoid", name="Hidden_Layer_3_Sigmoid"))
        model.add(tf.keras.layers.Dropout(0.1, name="Hidden_Layer_3_Dropout"))

        # Camada de Saída
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

        opt = tf.keras.optimizers.Adam()
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy", # Adequado para rótulos inteiros
            metrics=["accuracy"],
            jit_compile=jit_compile_dnn,
        )

        return model

class Metrics:
    """Agrupa funções de cálculo de métricas de avaliação."""
    
    @staticmethod
    def calculate_specificity(y_true, y_pred, class_label, num_classes):
        """
        Calcula a especificidade para uma classe específica.
        Especificidade = Verdadeiros Negativos / (Verdadeiros Negativos + Falsos Positivos)
        Args:
            y_true (np.ndarray): Rótulos verdadeiros.
            y_pred (np.ndarray): Rótulos previstos.
            class_label (int): O rótulo da classe para a qual calcular a especificidade (0, 1, 2,...).
            num_classes (int): Número total de classes.
        Returns:
            float: Valor da especificidade.
        """
        # Converte para um problema binário (Classe X vs. Não-Classe X)
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)

        # Matriz de confusão para o problema binário
        # [[TN, FP],
        #  [FN, TP]]
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

        if cm_binary.shape == (2,2):
            tn = cm_binary[0, 0]  # Verdadeiros Negativos
            fp = cm_binary[0, 1]  # Falsos Positivos
        elif cm_binary.shape == (1,1) and np.all(y_true_binary == 0): # Apenas negativos
            tn = cm_binary[0,0]
            fp = 0
        elif cm_binary.shape == (1,1) and np.all(y_true_binary == 1): # Apenas positivos
            tn = 0
            fp = 0
        else:
            return np.nan

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, class_names=None):
        """
        Calcula e imprime acurácia, relatório de classificação e especificidade por classe.
        Args:
            y_true (np.ndarray): Rótulos verdadeiros.
            y_pred (np.ndarray): Rótulos previstos.
            class_names (list, optional): Nomes das classes para o relatório.
        Returns:
            dict: Dicionário contendo as métricas.
        """
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist() # Convertendo para lista para serialização JSON
        }

        print(f"\nMatriz de Confusão:\n{cm}")
        print(f"\nAcurácia Geral: {acc:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        num_unique_classes = len(unique_labels)
        
        specificities = {}
        print("Especificidade por classe:", flush=True)
        
        if class_names:
            target_classes_for_specificity = range(len(class_names))
        else:
            target_classes_for_specificity = sorted(unique_labels)


        for class_val in target_classes_for_specificity:
            spec = Metrics.calculate_specificity(y_true, y_pred, class_label=class_val, num_classes=num_unique_classes)
            class_name_str = class_names[class_val] if class_names and class_val < len(class_names) else f"Classe {class_val}"
            print(f"  - {class_name_str}: {spec:.4f}", flush=True)
            # Gera uma chave limpa para o JSON
            spec_key = f"specificity_{class_name_str.replace(' ', '_').replace('(', '').replace(')', '')}"
            specificities[spec_key] = spec
        
        metrics["specificities"] = specificities
        return metrics

class Plotting:
    """Agrupa todas as funções de plotagem."""
    
    @staticmethod
    def _handle_plot(fig, filename, plots_dir, save_plots, title=""):
        """Função auxiliar interna para salvar ou mostrar plot."""
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            filepath = os.path.join(plots_dir, filename)
            fig.savefig(filepath)
            print(f"Plot salvo em: {filepath}", flush=True)
            plt.close(fig) # Fecha a figura para liberar memória
        else:
            if title: fig.suptitle(title, fontsize=16)
            plt.show()

    @staticmethod
    def plot_dnn_training_history(history, plots_dir, save_plots, title="Histórico de Treinamento DNN", filename="dnn_training_history.png"):
        """Plota o histórico de treinamento de um modelo Keras (loss e accuracy)."""
        if not history:
            print("Nenhum histórico de treinamento para plotar.", flush=True)
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if 'loss' in history and 'val_loss' in history:
            ax1.plot(history['loss'], label='Loss Treino')
            ax1.plot(history['val_loss'], label='Loss Validação')
            ax1.set_title('Loss do Modelo')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Época')
            ax1.legend(loc='upper right')
            ax1.grid(True)
        else:
            ax1.set_title('Dados de Loss Indisponíveis')

        if 'accuracy' in history and 'val_accuracy' in history:
            ax2.plot(history['accuracy'], label='Acurácia Treino')
            ax2.plot(history['val_accuracy'], label='Acurácia Validação')
            ax2.set_title('Acurácia do Modelo')
            ax2.set_ylabel('Acurácia')
            ax2.set_xlabel('Época')
            ax2.legend(loc='lower right')
            ax2.grid(True)
        else:
            ax2.set_title('Dados de Acurácia Indisponíveis')

        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_convergence_curves(curves, labels, plots_dir, save_plots, title="Curvas de Convergência", filename="optimizers_convergence.png"):
        """Plota múltiplas curvas de convergência (fitness vs. iteração)."""
        fig = plt.figure(figsize=(10, 6))
        for curve, label in zip(curves, labels):
            plt.plot(curve, label=label)
        plt.title(title)
        plt.xlabel("Iteração")
        plt.ylabel("Melhor Fitness")
        plt.legend()
        plt.grid(True)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_count_distribution(solutions_dict, plots_dir, save_plots, filename="feature_count_distribution.png"):
        """Plota um histograma da distribuição do número de features nas soluções únicas."""
        all_counts = []
        for algo_name, solutions in solutions_dict.items():
            if not solutions:
                continue
            feature_counts = [np.sum(sol) for fit, sol in solutions]
            counts = Counter(feature_counts)
            for num_features, count in counts.items():
                all_counts.append([algo_name, num_features, count])

        if not all_counts:
            print("Nenhuma solução para plotar a distribuição de contagem de features.")
            return

        df = pd.DataFrame(all_counts, columns=['Algoritmo', 'Numero de Features', 'Contagem'])

        fig = plt.figure(figsize=(15, 8))
        sns.barplot(data=df, x='Numero de Features', y='Contagem', hue='Algoritmo', dodge=True)
        plt.title('Distribuição da Contagem de Features por Solução Única Encontrada')
        plt.xlabel('Número de Features Selecionadas')
        plt.ylabel('Nº de Soluções Únicas Encontradas')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, "Distribuição de Features")

    @staticmethod
    def plot_final_metrics_comparison_bars(results_dict, class_labels, plots_dir, save_plots, base_filename="final_metrics"):
        """Cria gráficos de barras comparando as métricas finais (Acurácia, Recall, etc.)."""
        if not results_dict:
            return

        pipelines = list(results_dict.keys())
        if not pipelines: return

        overall_metrics = ['accuracy']
        metric_values_overall = {m: [] for m in overall_metrics + ['f1_macro']}
        per_class_metrics = ['precision', 'recall', 'f1-score', 'specificity']
        metric_values_per_class = {f"{cls_label}_{m}": [] for cls_label in class_labels for m in per_class_metrics}
        num_selected_features_list = []

        for pipe_name in pipelines:
            res = results_dict.get(pipe_name)
            if not res: continue

            num_selected_features_list.append(res.get('num_selected_features', 0))
            metric_values_overall['accuracy'].append(res.get('accuracy', 0) * 100)
            
            if 'classification_report' in res and isinstance(res['classification_report'], dict):
                metric_values_overall['f1_macro'].append(res['classification_report'].get('macro avg', {}).get('f1-score', 0) * 100)
            else:
                metric_values_overall['f1_macro'].append(0)

            for cls_idx, cls_label_key in enumerate(class_labels):
                if 'classification_report' in res and isinstance(res['classification_report'], dict):
                    class_report_data = res['classification_report'].get(cls_label_key, {})
                    metric_values_per_class[f"{cls_label_key}_precision"].append(class_report_data.get('precision', 0) * 100)
                    metric_values_per_class[f"{cls_label_key}_recall"].append(class_report_data.get('recall', 0) * 100)
                    metric_values_per_class[f"{cls_label_key}_f1-score"].append(class_report_data.get('f1-score', 0) * 100)
                else:
                    metric_values_per_class[f"{cls_label_key}_precision"].append(0)
                    metric_values_per_class[f"{cls_label_key}_recall"].append(0)
                    metric_values_per_class[f"{cls_label_key}_f1-score"].append(0)

                if 'specificities' in res and isinstance(res['specificities'], dict):
                     # Gera a chave de especificidade (ex: 'specificity_Normal_0')
                     spec_key = f"specificity_{cls_label_key.replace(' ', '_').replace('(', '').replace(')', '')}"
                     metric_values_per_class[f"{cls_label_key}_specificity"].append(res['specificities'].get(spec_key, 0) * 100)
                else:
                     metric_values_per_class[f"{cls_label_key}_specificity"].append(0)

        x = np.arange(len(pipelines))

        # Plot Acurácia e F1-Macro
        fig_overall, ax_overall = plt.subplots(figsize=(10, 6))
        width = 0.35
        rects1 = ax_overall.bar(x - width/2, metric_values_overall['accuracy'], width, label='Acurácia (%)')
        rects2 = ax_overall.bar(x + width/2, metric_values_overall['f1_macro'], width, label='F1-Score Macro (%)')
        ax_overall.set_ylabel('Pontuação (%)')
        ax_overall.set_title('Comparação de Acurácia Geral e F1-Score Macro')
        ax_overall.set_xticks(x)
        ax_overall.set_xticklabels(pipelines)
        ax_overall.legend()
        ax_overall.bar_label(rects1, padding=3, fmt='%.2f')
        ax_overall.bar_label(rects2, padding=3, fmt='%.2f')
        ax_overall.set_ylim(0, 105)
        plt.tight_layout()
        Plotting._handle_plot(fig_overall, f"{base_filename}_overall_accuracy_f1.png", plots_dir, save_plots, "Acurácia e F1-Macro")

        # Plot Recall (Sensibilidade) por Classe
        fig_recall, ax_recall = plt.subplots(figsize=(12, 7))
        width_recall = 0.25
        # Ajusta offsets se houver mais ou menos classes
        if len(class_labels) == 3:
            offsets = np.array([-width_recall, 0, width_recall])
        else:
            offsets = (np.arange(len(class_labels)) - (len(class_labels) - 1) / 2) * width_recall

        for i, cls_label_key in enumerate(class_labels):
            recalls = metric_values_per_class[f"{cls_label_key}_recall"]
            rects = ax_recall.bar(x + offsets[i], recalls, width_recall, label=f'Recall {cls_label_key}')
            ax_recall.bar_label(rects, padding=3, fmt='%.2f')

        ax_recall.set_ylabel('Recall / Sensibilidade (%)')
        ax_recall.set_title('Comparação de Recall (Sensibilidade) por Classe')
        ax_recall.set_xticks(x)
        ax_recall.set_xticklabels(pipelines)
        ax_recall.legend(loc='upper left', bbox_to_anchor=(1,1))
        ax_recall.set_ylim(0, 105)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        Plotting._handle_plot(fig_recall, f"{base_filename}_recall_per_class.png", plots_dir, save_plots, "Recall por Classe")

        # Plot Número de Features
        fig_nfeat, ax_nfeat = plt.subplots(figsize=(8, 5))
        rects_nfeat = ax_nfeat.bar(pipelines, num_selected_features_list, color=['skyblue', 'lightcoral'])
        ax_nfeat.set_ylabel('Número de Features Selecionadas')
        ax_nfeat.set_title('Número de Features Selecionadas por Algoritmo')
        ax_nfeat.bar_label(rects_nfeat, padding=3)
        plt.tight_layout()
        Plotting._handle_plot(fig_nfeat, f"{base_filename}_num_features.png", plots_dir, save_plots, "Número de Features")

    @staticmethod
    def plot_eeg_segments(segments_dict, fs, plots_dir, save_plots, n_segments_to_plot=1, base_filename="eeg_segment"):
        """Plota exemplos de segmentos EEG (tempo) e seus espectros de frequência (PSD)."""
        if not segments_dict:
            return

        num_types = len(segments_dict)
        keys = list(segments_dict.keys())
        
        min_available_segments = 0
        valid_datas = [data for data in segments_dict.values() if data is not None and data.ndim > 1 and data.shape[0] > 0]
        if valid_datas:
             min_available_segments = min(data.shape[0] for data in valid_datas)

        if min_available_segments == 0:
            print("Aviso: Nenhum segmento disponível para plotar em plot_eeg_segments.")
            return
        segments_to_plot_actual = min(n_segments_to_plot, min_available_segments)

        for i in range(segments_to_plot_actual):
            fig, axs = plt.subplots(num_types, 2, figsize=(15, 5 * num_types))
            if num_types == 1:
                axs = np.array([axs]) 

            for row_idx, key in enumerate(keys):
                data_array = segments_dict[key]
                if data_array is None or data_array.shape[0] <= i:
                    if axs[row_idx, 0]: axs[row_idx, 0].set_title(f"{key} - Segmento {i} (Dados Indisponíveis)")
                    if axs[row_idx, 1]: axs[row_idx, 1].set_title(f"Espectro {key} - Segmento {i} (Dados Indisponíveis)")
                    continue

                segment = data_array[i, :]
                time_vector = np.arange(segment.size) / fs

                # Domínio do tempo
                axs[row_idx, 0].plot(time_vector, segment)
                axs[row_idx, 0].set_title(f"{key} - Segmento {i} (Domínio do Tempo)")
                axs[row_idx, 0].set_xlabel("Tempo (s)")
                axs[row_idx, 0].set_ylabel("Amplitude")
                axs[row_idx, 0].grid(True)

                # Domínio da frequência (PSD via Welch)
                frequencies, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
                axs[row_idx, 1].semilogy(frequencies, psd) # Escala Log
                axs[row_idx, 1].set_title(f"Espectro {key} - Segmento {i} (PSD via Welch)")
                axs[row_idx, 1].set_xlabel("Frequência (Hz)")
                axs[row_idx, 1].set_ylabel("PSD (V^2/Hz)")
                axs[row_idx, 1].grid(True)
                axs[row_idx, 1].set_xlim(0, fs / 2) # Limita até a frequência de Nyquist

            plt.tight_layout()
            Plotting._handle_plot(fig, f"{base_filename}_example_{i}.png", plots_dir, save_plots, f"Exemplo de Segmento EEG {i}")

    @staticmethod
    def plot_swt_coefficients(coeffs_map, plots_dir, save_plots, segment_idx=0, base_filename="swt_coeffs"):
        """Plota os coeficientes SWT (A/D) para um segmento."""
        if not coeffs_map:
            return
        
        band_names = list(coeffs_map.keys())
        num_bands_to_plot = len(band_names)
        if num_bands_to_plot == 0:
            return

        fig, axs = plt.subplots(num_bands_to_plot, 1, figsize=(12, 3 * num_bands_to_plot), sharex=True)
        if num_bands_to_plot == 1:
                axs = np.array([axs])

        for idx, band_name in enumerate(band_names):
            coeffs = coeffs_map[band_name]
            if coeffs is not None and isinstance(coeffs, np.ndarray) and coeffs.ndim > 0 and coeffs.size > 0:
                axs[idx].plot(coeffs)
                axs[idx].set_title(f"Coeficientes {band_name}")
                axs[idx].set_ylabel("Amplitude")
            else:
                axs[idx].set_title(f"Coeficientes {band_name} (Indisponível)")
        
        axs[-1].set_xlabel("Amostra do Coeficiente")
        plt.tight_layout()
        Plotting._handle_plot(fig, f"{base_filename}_segment_{segment_idx}.png", plots_dir, save_plots, f"Coeficientes SWT - Segmento {segment_idx}")

    @staticmethod
    def plot_data_distribution_pca(X_dict, y_dict, plots_dir, save_plots, title="Distribuição dos Dados (PCA)", filename="data_distribution_pca.png", class_names=None):
        """Plota a distribuição dos conjuntos de dados (treino/val/teste) usando PCA 2D."""
        if not X_dict or not y_dict:
            print("Dicionários X ou y estão vazios. Não é possível plotar a distribuição.")
            return

        combined_X = []
        combined_y = []
        set_labels = [] # Para diferenciar treino, val, teste

        print(f"Plotando distribuição para os conjuntos: {list(X_dict.keys())}")

        for set_name, X_data in X_dict.items():
            if X_data is None or X_data.shape[0] == 0:
                print(f"Aviso: Conjunto '{set_name}' está vazio. Pulando.")
                continue
            if y_dict.get(set_name) is None or len(y_dict[set_name]) != X_data.shape[0]:
                print(f"Aviso: Rótulos para '{set_name}' ausentes ou incorretos. Pulando.")
                continue
            
            combined_X.append(X_data)
            combined_y.append(y_dict[set_name])
            set_labels.extend([set_name] * X_data.shape[0])

        if not combined_X:
            print("Nenhum dado válido para combinar e plotar.")
            return

        X_all = np.vstack(combined_X)
        y_all = np.concatenate(combined_y)
        
        # 1. Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        # 2. Aplicar PCA
        if X_scaled.shape[1] < 2:
            print("Número de features < 2, não é possível aplicar PCA para 2D.")
            return

        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_scaled)

        df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        df_pca['label'] = y_all
        df_pca['set'] = set_labels

        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"PCA: Variância explicada (2D): {explained_variance_ratio.sum()*100:.2f}%")

        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter_plot = sns.scatterplot(
            x="PC1", y="PC2",
            hue="label", # Cor por classe
            style="set", # Marcador por conjunto (Treino, Val, Teste)
            data=df_pca,
            palette="plasma",
            s=50,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel(f'Componente Principal 1 ({explained_variance_ratio[0]*100:.2f}% var)')
        ax.set_ylabel(f'Componente Principal 2 ({explained_variance_ratio[1]*100:.2f}% var)')
        ax.grid(True)

        # Ajusta legendas
        handles, labels = scatter_plot.get_legend_handles_labels()
        num_classes = len(np.unique(y_all))
        num_sets = len(np.unique(set_labels))
        hue_handles = handles[1:num_classes+1]
        hue_labels_orig = labels[1:num_classes+1]
        style_handles = handles[num_classes+2 : num_classes+2+num_sets]
        style_labels = labels[num_classes+2 : num_classes+2+num_sets]

        if class_names:
            final_hue_labels = [class_names[int(lbl)] if lbl.isdigit() and int(lbl) < len(class_names) else lbl for lbl in hue_labels_orig]
        else:
            final_hue_labels = hue_labels_orig
            
        leg1 = ax.legend(hue_handles, final_hue_labels, title='Classe', loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.add_artist(leg1)
        
        if style_handles and style_labels:
            ax.legend(style_handles, style_labels, title='Conjunto', loc='lower left', bbox_to_anchor=(1.01, 0))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_optimization_diagnostics(
        curves_dict, plots_dir, save_plots, title="Diagnóstico da Otimização", filename="opt_diagnostics.png"
    ):
        """Plota métricas de diagnóstico do otimizador (fitness, acurácia, n_features)."""
        num_plots = len(curves_dict)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1:
            axs = [axs]

        iterations = range(len(next(iter(curves_dict.values()))))

        for ax, (metric_name, curve_data) in zip(axs, curves_dict.items()):
            ax.plot(iterations, curve_data, marker=".")
            ax.set_title(f"Evolução de '{metric_name}' do Melhor Agente")
            ax.set_ylabel(metric_name)
            ax.grid(True)

        axs[-1].set_xlabel("Iteração")
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def visualize_knn_decision_boundary(
        X_train_all_features,
        y_train,
        selected_features_vector,
        plots_dir, save_plots,
        class_names=None,
        title="Fronteira de Decisão KNN",
        filename="knn_decision_boundary.png",
    ):
        """Visualiza a fronteira de decisão 2D (via PCA) para o KNN."""

        selected_indices = np.where(selected_features_vector == 1)[0]
        if len(selected_indices) < 2:
            print("Visualização da fronteira de decisão requer pelo menos 2 features.")
            return

        # 1. Seleciona e Padroniza
        X_selected = X_train_all_features[:, selected_indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # 2. Reduz para 2D com PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)

        # 3. Treina KNN nos dados 2D
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(X_2d, y_train)

        # 4. Cria meshgrid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 5. Plota
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1], c=y_train, s=20, edgecolor="k", cmap=plt.cm.RdYlBu
        )
        plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        plt.title(title)

        if class_names:
            legend1 = plt.legend(
                handles=scatter.legend_elements()[0], labels=class_names, title="Classes"
            )
            plt.gca().add_artist(legend1)

        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

class PipelineHelpers:
    """Agrupa funções auxiliares do pipeline (criação de fitness, treino final, etc.)."""

    @staticmethod
    def create_fitness_function_for_optimizer(X_train_features, y_train_labels, alpha, beta, random_seed):
        """
        Cria e retorna uma função de fitness otimizada que encapsula um KNN pré-configurado.
        """
        # Configura o classificador KNN uma única vez
        knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance', algorithm="kd_tree")
        
        n_folds = 10
        min_samples_per_class = np.min(np.bincount(y_train_labels))
        if min_samples_per_class < n_folds:
            print(f"Aviso Fitness: A menor classe possui {min_samples_per_class} amostras. "
                  f"Ajustando n_folds para {min_samples_per_class}.")
            n_folds = max(2, min_samples_per_class)
            
        # Configura o divisor da validação cruzada uma única vez
        cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

        def evaluate_fitness_configured(binary_feature_vector, *args, **kwargs):
            """
            Função de fitness interna que será chamada pelo otimizador.
            Reutiliza o knn_classifier e cv_splitter pré-configurados.
            
            kwargs aceita 'verbose_level' (mas não é usado aqui para velocidade).
            """
            selected_indices = np.where(binary_feature_vector == 1)[0]
            num_selected = len(selected_indices)
            total_features = len(binary_feature_vector)
            
            # Penalidade se estiver fora dos limites (MIN/MAX definidos globalmente)
            if not (MIN_FEATURES <= num_selected <= MAX_FEATURES):
                return {
                    'fitness': 1.0, # Pior fitness
                    'accuracy': 0.0,
                    'num_features': num_selected
                }

            X_train_selected = X_train_features[:, selected_indices]

            try:
                # Usa cross_val_score para eficiência com n_jobs=-1
                accuracies = cross_val_score(
                    knn_classifier,
                    X_train_selected,
                    y_train_labels,
                    cv=cv_splitter,
                    scoring="accuracy",
                    n_jobs=-1,  # Paraleliza a validação cruzada
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
        model_name,
        selected_features_vector,
        X_train_full_all_feat,
        y_train_full,
        X_test_all_feat,
        y_test,
        dnn_params,
        class_names,
        opt_fitness_score,
        plots_dir, # Necessário para salvar o plot
        save_plots # Necessário para salvar o plot
    ):
        """Treina e avalia o modelo DNN final com um subconjunto de features."""
        print(f"\n--- Treinamento e Avaliação Final: {model_name} ---")
        selected_indices = np.where(selected_features_vector == 1)[0]
        num_selected = len(selected_indices)

        if num_selected == 0:
            print(f"ERRO: {model_name} não selecionou nenhuma feature. Avaliação abortada.")
            return None, None

        print(f"{model_name}: Selecionou {num_selected} características.")

        X_train_full_selected = X_train_full_all_feat[:, selected_indices]
        X_test_selected = X_test_all_feat[:, selected_indices]
        
        tf.keras.backend.clear_session()
        final_model = ModelBuilder.build_dnn_model( # Chama o ModelBuilder
            num_selected_features=num_selected,
            num_classes=len(class_names),
            jit_compile_dnn=True,
        )
        if VERBOSE_OPTIMIZER_LEVEL > 0:
            print(f"Modelo final {model_name} construído com {num_selected} features.")
            final_model.summary()

        print(f"Iniciando treinamento final do modelo {model_name}...")
        early_stopping_final = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=dnn_params.get("patience", 30),
            restore_best_weights=True,
            verbose=1 if VERBOSE_OPTIMIZER_LEVEL > 0 else 0,
        )
        history = final_model.fit(
            X_train_full_selected,
            y_train_full,
            epochs=dnn_params.get("epochs", 150),
            batch_size=dnn_params.get("batch_size", 128),
            validation_split=0.15, # Usa parte do treino+val para early stopping
            callbacks=[early_stopping_final],
            verbose=1 if VERBOSE_OPTIMIZER_LEVEL > 0 else 0,
        )

        history_data = history.history
        Plotting.plot_dnn_training_history( # Chama o Plotting
            history_data,
            plots_dir,
            save_plots,
            title=f"Histórico de Treino Final - {model_name}",
            filename=f"final_dnn_history_{model_name.replace('+', '_').replace('-', '_')}.png",
        )

        print(f"\nAvaliando {model_name} no conjunto de teste...")
        y_pred_test_probs = final_model.predict(X_test_selected)
        y_pred_test = np.argmax(y_pred_test_probs, axis=1)

        metrics = Metrics.calculate_all_metrics(y_test, y_pred_test, class_names=class_names) # Chama o Metrics
        metrics["num_selected_features"] = num_selected
        metrics["selected_feature_indices"] = selected_indices.tolist()
        metrics["fitness_score_from_optimizer"] = opt_fitness_score

        model_save_path = os.path.join(
            RUN_RESULTS_DIR, f"{model_name.replace('+', '_').replace('-', '_')}_final_model.keras"
        )
        try:
            final_model.save(model_save_path)
        except Exception as e:
            print(f"Erro ao salvar o modelo {model_name}: {e}")

        del final_model
        gc.collect()
        return metrics, history_data

    @staticmethod
    def get_top_n_unique_solutions(history, n=20):
        """Processa o histórico de soluções para retornar as N melhores soluções únicas."""
        if not history:
            return []

        unique_solutions = {} # {sol_tuple: best_fitness}
        for fitness, sol in history:
            sol_tuple = tuple(sol)
            if sol_tuple not in unique_solutions or fitness < unique_solutions[sol_tuple]:
                unique_solutions[sol_tuple] = fitness

        # Ordena as soluções únicas pelo fitness (menor para maior)
        sorted_solutions = sorted(unique_solutions.items(), key=lambda item: item[1])

        # Retorna o (fitness, vetor_solucao)
        top_n = [(fit, np.array(sol)) for sol, fit in sorted_solutions[:n]]
        return top_n

    @staticmethod
    def get_all_unique_solutions_sorted(history):
        """Retorna TODAS as soluções únicas ordenadas por fitness."""
        if not history: return []
        unique_solutions = {}
        for fitness, sol in history:
            sol_tuple = tuple(sol)
            if sol_tuple not in unique_solutions or fitness < unique_solutions[sol_tuple]:
                unique_solutions[sol_tuple] = fitness
        sorted_solutions = sorted(unique_solutions.items(), key=lambda item: item[1])
        return [(fit, np.array(sol)) for sol, fit in sorted_solutions]

# --- Script Principal ---
if __name__ == "__main__":
    start_time_total = time.time()

    print("Iniciando Pipeline de Detecção de Epilepsia...")
    print(f"Usando TensorFlow versão: {tf.__version__}")
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

    # 1. Carregar Dados
    print("\n--- 1. Carregando Dados ---")
    try:
        raw_data, raw_labels = DataHandler.load_bonn_data(BASE_DATA_DIR)
    except Exception as e:
        print(f"Falha ao carregar dados: {e}. Verifique o caminho e formato do dataset.")
        sys.exit(1)

    if VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nPlotando exemplos de sinais EEG brutos...", flush=True)
        Plotting.plot_eeg_segments(
            {"Raw": raw_data},
            fs=FS,
            plots_dir=PLOTS_DIR,
            save_plots=SAVE_PLOTS,
            n_segments_to_plot=1,
            base_filename="eeg_raw_example",
        )

    # 2. Pré-processar Dados
    print("\n--- 2. Pré-processando Dados ---")
    data_processed = DataHandler.preprocess_eeg(
        raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
    )

    if VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nPlotando exemplos de sinais EEG pré-processados...", flush=True)
        Plotting.plot_eeg_segments(
            {"Processed": data_processed},
            fs=FS,
            plots_dir=PLOTS_DIR,
            save_plots=SAVE_PLOTS,
            n_segments_to_plot=1,
            base_filename="eeg_processed_example",
        )

    # 3. Dividir Dados
    print("\n--- 3. Dividindo Dados ---")
    # X_train_p, X_val_p, X_test_p são dados no domínio do tempo
    X_train_p, X_val_p, X_test_p, y_train_labels, y_val_labels, y_test_labels = (
        DataHandler.split_data( # Usando o método da classe DataHandler
            data_processed,
            raw_labels,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_SEED,
        )
    )

    # 4. Extrair Características SWT
    print("\n--- 4. Extraindo Características SWT ---")
    print("Extraindo features para o conjunto de TREINO (usado pelos otimizadores)...")
    X_train_feat_opt, feature_names = FeatureExtractor.extract_swt_features(
        X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )
    
    # Plotar coeficientes SWT de um segmento de exemplo
    if X_train_p.shape[0] > 0 and VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nPlotando coeficientes SWT de um segmento de treino de exemplo...", flush=True)
        slfs = X_train_p.shape[1] - (X_train_p.shape[1] % 2)
        example_signal_for_swt_plot_truncated = X_train_p[0, :slfs]

        # Recalcula SWT apenas para obter os coeficientes para plotagem
        swt_coeffs_arrays_example = pywt.swt(
            example_signal_for_swt_plot_truncated, wavelet=SWT_WAVELET, level=SWT_LEVEL, trim_approx=True, norm=True
        )

        example_coeffs_map_for_plot = {}
        if isinstance(swt_coeffs_arrays_example, list) and len(swt_coeffs_arrays_example) == (SWT_LEVEL + 1):
            example_coeffs_map_for_plot[f"A{SWT_LEVEL}"] = swt_coeffs_arrays_example[0]
            for k_idx_plot in range(SWT_LEVEL):
                detail_level_val_plot = SWT_LEVEL - k_idx_plot
                array_idx_plot = k_idx_plot + 1
                example_coeffs_map_for_plot[f"D{detail_level_val_plot}"] = (
                    swt_coeffs_arrays_example[array_idx_plot]
                )
            Plotting.plot_swt_coefficients(
                example_coeffs_map_for_plot,
                plots_dir=PLOTS_DIR,
                save_plots=SAVE_PLOTS,
                segment_idx=0,
                base_filename="swt_coeffs_train_example",
            )
        else:
            print("Não foi possível obter coeficientes SWT para plotagem de exemplo.", flush=True)
            
    print("Extraindo features para o conjunto de VALIDAÇÃO...")
    X_val_feat_combine, _ = FeatureExtractor.extract_swt_features(
        X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )

    print("Extraindo features para o conjunto de TESTE...")
    X_test_feat_final, _ = FeatureExtractor.extract_swt_features(
        X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )

    if not feature_names:
        print("ERRO: Nomes de features não foram gerados. Verifique a extração.")
        sys.exit(1)

    DIM_FEATURES = X_train_feat_opt.shape[1]
    print(f"Total de {DIM_FEATURES} características extraídas.")
    
    # O artigo afirma extrair 143 features.
    if DIM_FEATURES != 143:
        print(f"AVISO: Número de features extraídas ({DIM_FEATURES}) não corresponde a 143.")

    if VERBOSE_OPTIMIZER_LEVEL > 0 and X_train_p.shape[0] > 0:
        print("\nPlotando distribuição dos dados (PCA) após extração de features...", flush=True)
        X_datasets_for_pca_plot = {
            "Treino_Opt": X_train_feat_opt,
            "Val_Combine": X_val_feat_combine,
            "Teste_Final": X_test_feat_final,
        }
        y_datasets_for_pca_plot = {
            "Treino_Opt": y_train_labels,
            "Val_Combine": y_val_labels,
            "Teste_Final": y_test_labels,
        }
        Plotting.plot_data_distribution_pca(
            X_datasets_for_pca_plot,
            y_datasets_for_pca_plot,
            plots_dir=PLOTS_DIR,
            save_plots=SAVE_PLOTS,
            title="Distribuição dos Conjuntos de Features SWT (PCA)",
            filename="data_distribution_pca_swt_features.png",
            class_names=class_names,
        )

    all_results = {}
    all_convergence_curves = []
    convergence_labels = []
    all_candidate_solutions = {}

    # --- 5. Otimização com BDA ---
    print("\n\n--- 5. Otimização com Binary Dragonfly Algorithm (BDA) ---")
    start_time_bda_opt = time.time()
    
    # Cria a função de fitness otimizada (com KNN e CV embutidos)
    fitness_function_for_bda = PipelineHelpers.create_fitness_function_for_optimizer(
        X_train_feat_opt, y_train_labels, ALPHA_FITNESS, BETA_FITNESS, RANDOM_SEED
    )

    bda = BinaryDragonflyAlgorithm(
        N=N_AGENTS_OPTIMIZERS,
        T=T_MAX_ITER_OPTIMIZERS,
        dim=DIM_FEATURES,
        fitness_func=fitness_function_for_bda,
        X_train_feat=X_train_feat_opt,
        y_train=y_train_labels,
        alpha_fitness=ALPHA_FITNESS,
        beta_fitness=BETA_FITNESS,
        seed=RANDOM_SEED,
        verbose_optimizer_level=VERBOSE_OPTIMIZER_LEVEL,
        min_features=MIN_FEATURES,
        max_features=MAX_FEATURES,
    )
    Sf_bda, best_fitness_bda, convergence_bda, acc_curve_bda, nfeat_curve_bda, bda_history = bda.run()

    # Plot de diagnóstico do BDA
    bda_diagnostic_curves = {
        "Melhor Fitness": convergence_bda,
        "Acurácia do Melhor Agente (%)": np.array(acc_curve_bda) * 100,
        "Nº de Features do Melhor Agente": nfeat_curve_bda,
    }
    Plotting.plot_optimization_diagnostics(
        bda_diagnostic_curves,
        plots_dir=PLOTS_DIR,
        save_plots=SAVE_PLOTS,
        title="Diagnóstico da Otimização - BDA",
        filename="bda_diagnostics.png",
    )

    # Plot da fronteira de decisão do KNN para a melhor solução do BDA
    if Sf_bda is not None and np.sum(Sf_bda) > 1 and VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nGerando visualização da fronteira de decisão do KNN para a solução final do BDA...")
        Plotting.visualize_knn_decision_boundary(
            X_train_feat_opt,
            y_train_labels,
            Sf_bda,
            plots_dir=PLOTS_DIR,
            save_plots=SAVE_PLOTS,
            class_names=class_names,
            title="Fronteira de Decisão KNN (Solução Final BDA)",
            filename="bda_final_solution_knn_boundary.png",
        )

    all_results["bda_optimization"] = {
        "best_fitness": best_fitness_bda,
        "selected_features_vector": (Sf_bda.tolist() if isinstance(Sf_bda, np.ndarray) else Sf_bda),
        "num_selected_features": (int(np.sum(Sf_bda)) if isinstance(Sf_bda, np.ndarray) else 0),
        "convergence_curve": (convergence_bda.tolist() if isinstance(convergence_bda, np.ndarray) else convergence_bda),
    }
    all_convergence_curves.append(convergence_bda)
    convergence_labels.append("BDA")
    all_candidate_solutions["BDA"] = PipelineHelpers.get_all_unique_solutions_sorted(bda_history)
    
    print(f"BDA encontrou {len(all_candidate_solutions['BDA'])} soluções únicas.")
    print(f"Tempo de otimização BDA: {(time.time() - start_time_bda_opt)/60:.2f} minutos")

    gc.collect()
    
    # Plots comparativos da fase de otimização
    Plotting.plot_convergence_curves(all_convergence_curves, convergence_labels, plots_dir=PLOTS_DIR, save_plots=SAVE_PLOTS, title="Convergência dos Otimizadores", filename="optimizers_convergence.png")
    Plotting.plot_feature_count_distribution(all_candidate_solutions, plots_dir=PLOTS_DIR, save_plots=SAVE_PLOTS, filename="feature_count_distribution.png")

    # --- 7. Treinamento e Avaliação Final da DNN ---
    print("\n\n--- 7. Treinamento e Avaliação Final dos Modelos DNN ---")
    # Combina treino e validação para o treino final da DNN (conforme artigo)
    X_train_full_feat_final = np.concatenate((X_train_feat_opt, X_val_feat_combine), axis=0)
    y_train_full_labels_final = np.concatenate((y_train_labels, y_val_labels), axis=0)
    print(f"Usando {X_train_full_feat_final.shape[0]} amostras para o treinamento final da DNN.")

    for algo_name, candidate_solutions in all_candidate_solutions.items():
        print(f"\n--- Processando Candidatos de {algo_name} ---")
        
        filtered_candidates = candidate_solutions
        if ENABLE_FEATURE_COUNT_FILTER:
            print(f"Filtrando candidatos de {algo_name} para {TARGET_FEATURE_COUNT} features...")
            filtered_candidates = [(f, s) for f, s in candidate_solutions if np.sum(s) == TARGET_FEATURE_COUNT]
            print(f"Encontrados {len(filtered_candidates)} candidatos com o número de features alvo.")

        if not filtered_candidates:
            print(f"Nenhum candidato de {algo_name} sobrou após a filtragem.")
            continue

        # Loop de Treinamento com Limiar de Qualidade
        final_good_model_count = 0
        candidate_idx = 0
        
        while final_good_model_count < MAX_FINAL_MODELS_TO_KEEP and candidate_idx < len(filtered_candidates):
            fitness_score, solution_vector = filtered_candidates[candidate_idx]
            num_features = np.sum(solution_vector)
            model_rank = candidate_idx + 1 
            model_name = f"{algo_name}-F{num_features}-Rank{model_rank}" # Ex: BDA-F19-Rank1

            print(f"\n>>> Treinando modelo {final_good_model_count + 1}/{MAX_FINAL_MODELS_TO_KEEP} (Candidato Rank {model_rank} de {algo_name})...")
            
            metrics, history_data = PipelineHelpers.train_and_evaluate_final_model(
                model_name=model_name,
                selected_features_vector=solution_vector,
                X_train_full_all_feat=X_train_full_feat_final,
                y_train_full=y_train_full_labels_final,
                X_test_all_feat=X_test_feat_final,
                y_test=y_test_labels,
                dnn_params=DNN_TRAINING_PARAMS_FINAL,
                class_names=class_names,
                opt_fitness_score=fitness_score,
                plots_dir=PLOTS_DIR,
                save_plots=SAVE_PLOTS
            )
            
            candidate_idx += 1 # Avança para o próximo candidato

            if metrics and metrics.get("accuracy", 0) >= FINAL_MODEL_ACCURACY_THRESHOLD:
                print(f"+++ SUCESSO: Modelo {model_name} atingiu {metrics['accuracy']:.2%} de acurácia. Mantendo.")
                final_good_model_count += 1
                all_results[f"{model_name}_final_eval"] = metrics
            elif metrics:
                print(f"--- DESCARTADO: Modelo {model_name} com acurácia de {metrics['accuracy']:.2%}, abaixo do limiar de {FINAL_MODEL_ACCURACY_THRESHOLD:.0%}.")
            else:
                print(f"### FALHA: Treinamento para {model_name} não produziu métricas. Descartando.")

        if final_good_model_count < MAX_FINAL_MODELS_TO_KEEP:
            print(f"\nAVISO: Não foi possível encontrar {MAX_FINAL_MODELS_TO_KEEP} modelos para {algo_name} que satisfizessem o limiar. Encontrados: {final_good_model_count}.")

    # --- 8. Salvar Resultados Consolidados ---
    results_file_path = os.path.join(RUN_RESULTS_DIR, "all_pipeline_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        print(f"\nResultados consolidados salvos em: {results_file_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados consolidados: {e}")

    # --- 9. Tabela Comparativa ---
    print("\n\n--- Tabela Comparativa de Resultados (Conjunto de Teste) ---")
    print("------------------------------------------------------------------------------------------------------------------------------------------")
    print("| Modelo              | Fitness Opt. | Features Sel. | Acurácia (%) | Sens_Cl0 (%) | Sens_Cl1 (%) | Sens_Cl2 (%) | Esp_Cl0 (%) | Esp_Cl1 (%) | Esp_Cl2 (%) | F1_Macro (%) |")
    print("|---------------------|--------------|---------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|--------------|")

    # Função auxiliar local para imprimir linha da tabela
    def print_results_row_main(model_name, results_dict_eval):
        if not results_dict_eval: return
        
        fitness = results_dict_eval.get("fitness_score_from_optimizer", "N/A")
        num_feat = results_dict_eval.get("num_selected_features", "N/A")
        acc = results_dict_eval.get("accuracy", 0) * 100
        report = results_dict_eval.get("classification_report", {})
        sens_cl0 = report.get(class_names[0], {}).get("recall", 0) * 100
        sens_cl1 = report.get(class_names[1], {}).get("recall", 0) * 100
        sens_cl2 = report.get(class_names[2], {}).get("recall", 0) * 100
        f1_macro = report.get("macro avg", {}).get("f1-score", 0) * 100
        
        specificities = results_dict_eval.get("specificities", {})
        key_spec_cl0 = f"specificity_{class_names[0].replace(' ', '_').replace('(', '').replace(')', '')}"
        key_spec_cl1 = f"specificity_{class_names[1].replace(' ', '_').replace('(', '').replace(')', '')}"
        key_spec_cl2 = f"specificity_{class_names[2].replace(' ', '_').replace('(', '').replace(')', '')}"
        esp_cl0 = specificities.get(key_spec_cl0, 0) * 100
        esp_cl1 = specificities.get(key_spec_cl1, 0) * 100
        esp_cl2 = specificities.get(key_spec_cl2, 0) * 100
        
        fitness_str = f"{fitness:.4f}" if isinstance(fitness, (int, float)) else "N/A"

        print(
            f"| {model_name:<19} | {fitness_str:^12} | {str(num_feat):<13} | {acc:^12.2f} | {sens_cl0:^12.2f} | {sens_cl1:^12.2f} | {sens_cl2:^12.2f} | {esp_cl0:^11.2f} | {esp_cl1:^11.2f} | {esp_cl2:^11.2f} | {f1_macro:^12.2f} |"
        )
    
    # Ordena os resultados finais pelo nome para exibição
    sorted_results_keys = sorted([k for k in all_results if k.endswith('_final_eval')])
    for key in sorted_results_keys:
        model_name_display = key.replace('_final_eval', '')
        print_results_row_main(model_name_display, all_results[key])
    
    print("------------------------------------------------------------------------------------------------------------------------------------------")

    # --- 10. Plot Comparativo Final ---
    print("\n\n--- Gerando Gráficos Comparativos Finais ---")
    final_eval_results_for_plot = {k.replace('_final_eval', ''): v for k, v in all_results.items() if k.endswith('_final_eval')}

    if final_eval_results_for_plot:
        Plotting.plot_final_metrics_comparison_bars(
            final_eval_results_for_plot,
            class_labels=class_names,
            plots_dir=PLOTS_DIR,
            save_plots=SAVE_PLOTS,
            base_filename="final_model_metrics",
        )
    else:
        print("Nenhum resultado de avaliação final para plotar.")

    total_execution_time = time.time() - start_time_total
    print(f"\nTempo total de execução da pipeline: {total_execution_time/60:.2f} minutos")
    print("\n--- Fim da Execução ---")