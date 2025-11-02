# -*- coding: utf-8 -*-
"""
Módulo de Utilitários do Pipeline de Comparação.

Este arquivo centraliza todas as funções e classes compartilhadas
entre os diferentes pipelines (BDA-DNN, RHCB5, etc.) para garantir
consistência, especialmente no pré-processamento e avaliação.

Contém:
- Constantes Globais (Seeds, Parâmetros de Sinal, Nomes de Classes)
- NumpyEncoder: Para salvar resultados em JSON.
- DataHandler: Classe para carregar, pré-processar e dividir os dados.
- Metrics: Classe para calcular métricas de avaliação (incluindo especificidade).
- Plotting: Classe para gerar todos os gráficos (histórico, matrizes, boxplots).
"""

import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar matplotlib para não mostrar plots automaticamente
plt.ioff()  # Turn off interactive mode
matplotlib.interactive(False)  # Ensure non-interactive mode

# Desabilitar qualquer tentativa de mostrar plots
plt.switch_backend('Agg')  # Forçar backend Agg novamente
matplotlib.use('Agg', force=True)  # Forçar uso do backend Agg

# Configurar seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# --- Constantes Globais Compartilhadas ---
RANDOM_SEED_GLOBAL = 42 # Seed global para operações de setup
np.random.seed(RANDOM_SEED_GLOBAL)
tf.random.set_seed(RANDOM_SEED_GLOBAL)

# Nomes das classes
CLASS_NAMES = ["Normal (0)", "Interictal (1)", "Ictal (2)"]
NUM_CLASSES = len(CLASS_NAMES)

# Parâmetros do Dataset e Pré-processamento
FS = 173.61 # Frequência de amostragem
HIGHCUT_HZ = 40.0 # Frequência de corte do filtro passa-baixas
FILTER_ORDER = 4 # Ordem do filtro Butterworth
ORIGINAL_INPUT_LENGTH = 4097 # Comprimento original do sinal Bonn
TARGET_INPUT_LENGTH = 4096 # Comprimento alvo após remoção da 1ª amostra

# Parâmetros da Divisão de Dados
TEST_SIZE = 0.20 # Proporção do conjunto de teste
VAL_SIZE = 0.15 # Proporção do conjunto de validação

# Configuração de Hardware
USE_GPU = True # Flag para usar GPU (True) ou forçar CPU (False)

# Configuração de Análise
USE_XAI = True # Flag para executar análise XAI/SHAP (True) ou pular (False)

# --- Classes Auxiliares ---

class NumpyEncoder(json.JSONEncoder):
    """Codificador JSON customizado para lidar com tipos de dados NumPy."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.bool_, bool)):
            return bool(o)
        return super(NumpyEncoder, self).default(o)

# --- Classes Principais de Lógica ---

class DataHandler:
    """
    Classe unificada para manipulação de dados (carga, pré-processamento, divisão).
    Garante que todos os pipelines usem exatamente a mesma lógica.
    """
    
    @staticmethod
    def load_bonn_data(base_path):
        """
        Carrega os dados dos conjuntos A, D, E do dataset BONN.
        Ajusta de 4097 para 4096 amostras (removendo a primeira).
        """
        data_segments = []
        labels = []
        sets_labels = {'A': 0, 'D': 1, 'E': 2} # 0: Normal, 1: Interictal, 2: Ictal
        
        print("Iniciando carregamento dos dados...")
        for set_name, label in sets_labels.items():
            set_path = os.path.join(base_path, set_name)
            if not os.path.isdir(set_path):
                print(f"Aviso: Diretório não encontrado: {set_path}. Pulando.")
                continue

            fnames = sorted([f for f in os.listdir(set_path) if f.lower().endswith('.txt')])
            
            for fname in tqdm(fnames, desc=f"Lendo {set_name}", leave=False):
                file_path = os.path.join(set_path, fname)
                try:
                    # Carrega como float32 para economizar memória
                    segment_data = pd.read_csv(file_path, header=None, dtype=np.float32).values.flatten()

                    if len(segment_data) == ORIGINAL_INPUT_LENGTH:
                        # Ajuste crucial: remove a primeira amostra
                        adjusted_signal = segment_data[1:] 
                        data_segments.append(adjusted_signal)
                        labels.append(label)
                    else:
                        print(f"Aviso: Arquivo {fname} com tamanho inesperado {len(segment_data)}. Ignorando.")
                except Exception as e:
                    print(f"Erro ao carregar {fname}: {e}")

        if not data_segments:
            raise ValueError("Nenhum dado foi carregado. Verifique os caminhos e o formato dos arquivos.")

        dados_np = np.array(data_segments, dtype=np.float32)
        rotulos_np = np.array(labels, dtype=np.int32)
        print(f"Dados carregados: {dados_np.shape}, Rótulos: {rotulos_np.shape}")
        return dados_np, rotulos_np

    @staticmethod
    def preprocess_eeg(data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER):
        """
        Aplica filtro Butterworth passa-baixas e normalização Min-Max para [-1, 1].
        """
        processed_data = np.zeros_like(data)
        nyq = 0.5 * fs
        high = highcut_hz / nyq
        
        if high >= 1.0:
            raise ValueError(f"Frequência de corte ({highcut_hz} Hz) resulta em valor normalizado >= 1.0. Verifique FS e highcut_hz.")
            
        b, a = butter(order, high, btype='low', analog=False)
        scaler = MinMaxScaler(feature_range=(-1, 1))

        print("Iniciando pré-processamento (filtragem e normalização)...")
        for i in tqdm(range(data.shape[0]), desc="Pré-processando segmentos", leave=False):
            signal = data[i, :]
            
            # 1. Filtragem
            filtered_signal = filtfilt(b, a, signal)
            
            # 2. Normalização
            # Reshape para (n_amostras, 1) exigido pelo scaler
            reshaped_signal = filtered_signal.reshape(-1, 1) 
            normalized_signal = scaler.fit_transform(reshaped_signal).flatten()
            
            processed_data[i, :] = normalized_signal.astype(np.float32)
            
        print("Pré-processamento concluído.")
        return processed_data

    @staticmethod
    def split_data(data, labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED_GLOBAL):
        """
        Divide os dados em conjuntos de treino, validação e teste estratificados.
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

        print("Dados divididos (usando a mesma função para todos os pipelines):")
        print(f"  Treino:    {X_train.shape[0]} amostras")
        print(f"  Validação: {X_val.shape[0]} amostras")
        print(f"  Teste:     {X_test.shape[0]} amostras")
        return X_train, X_val, X_test, y_train, y_val, y_test

class Metrics:
    """Agrupa funções de cálculo de métricas de avaliação."""
    
    @staticmethod
    def calculate_specificity(y_true, y_pred, class_label):
        """
        Calcula a especificidade para uma classe específica (multiclasse).
        Especificidade = Verdadeiros Negativos / (Verdadeiros Negativos + Falsos Positivos)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Total de amostras
        total_sum = np.sum(cm)
        
        # Verdadeiros Positivos (TP) para a classe
        tp = cm[class_label, class_label]
        
        # Falsos Positivos (FP) para a classe (soma da coluna, exceto TP)
        fp = np.sum(cm[:, class_label]) - tp
        
        # Falsos Negativos (FN) para a classe (soma da linha, exceto TP)
        fn = np.sum(cm[class_label, :]) - tp
        
        # Verdadeiros Negativos (TN) para a classe (soma de tudo, exceto linha e coluna da classe)
        tn = total_sum - (tp + fp + fn)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, class_names=CLASS_NAMES):
        """
        Calcula e imprime acurácia, relatório de classificação e especificidade por classe.
        """
        # Garante que y_true e y_pred são arrays numpy
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }

        print(f"\nMatriz de Confusão (Run):\n{cm}")
        print(f"\nAcurácia Geral (Run): {acc:.4f}")
        print("\nRelatório de Classificação (Run):")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        specificities = {}
        print("Especificidade por classe (Run):", flush=True)
        
        for class_val in range(len(class_names)):
            spec = Metrics.calculate_specificity(y_true, y_pred, class_label=class_val)
            class_name_str = class_names[class_val]
            print(f"  - {class_name_str}: {spec:.4f}", flush=True)
            # Gera uma chave limpa para o JSON
            spec_key = f"specificity_{class_name_str.replace(' ', '_').replace('(', '').replace(')', '')}"
            specificities[spec_key] = spec
        
        metrics["specificities"] = specificities
        return metrics

class Plotting:
    """Agrupa todas as funções de plotagem (individuais e agregadas)."""
    
    @staticmethod
    def _handle_plot(fig, filename, plots_dir, save_plots, title=""):
        """Função auxiliar interna para salvar ou mostrar plot."""
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            filepath = os.path.join(plots_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight')
                print(f"Plot salvo em: {filepath}", flush=True)
            except Exception as e:
                print(f"Erro ao salvar plot {filepath}: {e}", flush=True)
        else:
            if title: fig.suptitle(title, fontsize=16)
        
        # Sempre fechar a figura para liberar memória, independentemente de salvar ou não
        plt.close(fig)

    # --- Funções de Plotagem de Execução Única ---

    @staticmethod
    def plot_dnn_training_history(history, plots_dir, save_plots, title, filename):
        """Plota o histórico de treinamento de um modelo Keras (loss e accuracy)."""
        if not history:
            print("Nenhum histórico de treinamento para plotar.", flush=True)
            return

        history_data = history if isinstance(history, dict) else history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if 'loss' in history_data and 'val_loss' in history_data:
            ax1.plot(history_data['loss'], label='Loss Treino')
            ax1.plot(history_data['val_loss'], label='Loss Validação')
            ax1.set_title('Loss do Modelo')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Época')
            ax1.legend(loc='upper right')
            ax1.grid(True)
        else:
            ax1.set_title('Dados de Loss Indisponíveis')

        if 'accuracy' in history_data and 'val_accuracy' in history_data:
            ax2.plot(history_data['accuracy'], label='Acurácia Treino')
            ax2.plot(history_data['val_accuracy'], label='Acurácia Validação')
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
    def plot_convergence_curves(curves, labels, plots_dir, save_plots, title, filename):
        """Plota múltiplas curvas de convergência (fitness vs. iteração)."""
        fig = plt.figure(figsize=(10, 6))
        for curve, label in zip(curves, labels):
            if curve is not None:
                plt.plot(curve, label=label)
        plt.title(title)
        plt.xlabel("Iteração")
        plt.ylabel("Melhor Fitness")
        plt.legend()
        plt.grid(True)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_optimization_diagnostics(curves_dict, plots_dir, save_plots, title, filename):
        """Plota métricas de diagnóstico do otimizador (fitness, acurácia, n_features)."""
        num_plots = len(curves_dict)
        if num_plots == 0: return
        
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
    def plot_single_confusion_matrix(cm, class_names, plots_dir, save_plots, title, filename):
        """Plota uma única matriz de confusão."""
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 14})
        plt.title(title, fontsize=18)
        plt.ylabel('Classe Verdadeira', fontsize=14)
        plt.xlabel('Classe Predita', fontsize=14)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    # --- Funções de Plotagem Agregada e Comparativa (para main.py) ---

    @staticmethod
    def _extract_metrics_for_plotting(results_dict_list, class_names):
        """Helper para extrair métricas de listas de resultados para um DataFrame."""
        plot_data = []
        for pipeline_name, results_list in results_dict_list.items():
            for run_idx, run_result in enumerate(results_list):
                if not run_result or 'final_metrics' not in run_result:
                    continue
                
                metrics = run_result['final_metrics']
                report = metrics.get('classification_report', {})
                specificities = metrics.get('specificities', {})
                
                # Métricas Gerais
                entry = {
                    'Pipeline': pipeline_name,
                    'run': run_idx + 1,
                    'accuracy': metrics.get('accuracy', np.nan),
                    'f1_macro': report.get('macro avg', {}).get('f1-score', np.nan),
                    'execution_time_min': run_result.get('execution_time_sec', np.nan) / 60.0,
                    'num_selected_features': run_result.get('num_selected_features', np.nan) # BDA-DNN
                }
                
                # Métricas por Classe (Recall, Specificity)
                for i, name in enumerate(class_names):
                    # Chave para o relatório (ex: "Normal (0)")
                    report_key = name
                    # Chave para especificidade (ex: "specificity_Normal_0")
                    spec_key = f"specificity_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
                    
                    entry[f'recall_{i}'] = report.get(report_key, {}).get('recall', np.nan)
                    entry[f'specificity_{i}'] = specificities.get(spec_key, np.nan)

                plot_data.append(entry)
                
        return pd.DataFrame(plot_data)

    @staticmethod
    def plot_comparison_boxplots(results_dict_list, class_names, plots_dir, save_plots):
        """
        Gera boxplots comparativos para Acurácia, F1-Macro, Tempo de Execução,
        e Recall/Especificidade por classe.
        
        Args:
            results_dict_list (dict): {'BDA_DNN': [run1_res, ...], 'RHCB5': [run1_res, ...]}
        """
        print("\nGerando boxplots comparativos...")
        df_plot = Plotting._extract_metrics_for_plotting(results_dict_list, class_names)
        
        if df_plot.empty:
            print("Nenhum dado para plotar nos boxplots comparativos.")
            return
            
        # 1. Métricas Gerais (Acurácia, F1-Macro, Tempo)
        general_metrics = ['accuracy', 'f1_macro', 'execution_time_min']
        fig_gen, axs_gen = plt.subplots(1, len(general_metrics), figsize=(8 * len(general_metrics), 7))
        if len(general_metrics) == 1: axs_gen = [axs_gen]
        
        for ax, metric in zip(axs_gen, general_metrics):
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="Set2", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.6)
            ax.set_title(f'Distribuição - {metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.split("_")[0].title())
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        Plotting._handle_plot(fig_gen, "comparison_boxplots_general.png", plots_dir, save_plots, "Boxplots Comparativos - Métricas Gerais")

        # 2. Número de Features (apenas BDA-DNN)
        df_bda = df_plot[df_plot['Pipeline'] == 'BDA_DNN']
        if not df_bda['num_selected_features'].isnull().all():
            fig_feat, ax_feat = plt.subplots(figsize=(8, 7))
            sns.boxplot(x='Pipeline', y='num_selected_features', data=df_bda, ax=ax_feat, hue='Pipeline', palette="Set2", legend=False)
            sns.stripplot(x='Pipeline', y='num_selected_features', data=df_bda, ax=ax_feat, color=".25", alpha=0.6)
            ax_feat.set_title('Distribuição - Nº de Features Selecionadas (BDA-DNN)')
            ax_feat.set_ylabel('Número de Features')
            ax_feat.grid(axis='y', linestyle='--', alpha=0.7)
            Plotting._handle_plot(fig_feat, "comparison_boxplot_features.png", plots_dir, save_plots, "Boxplot - Features BDA-DNN")
        
        # 3. Recall (Sensibilidade) por Classe
        fig_rec, axs_rec = plt.subplots(1, NUM_CLASSES, figsize=(8 * NUM_CLASSES, 7), sharey=True)
        if NUM_CLASSES == 1: axs_rec = [axs_rec]
        for i, ax in enumerate(axs_rec):
            metric = f'recall_{i}'
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="Set2", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.6)
            ax.set_title(f'Recall (Sensibilidade) - {class_names[i]}')
            ax.set_ylabel('Recall')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        Plotting._handle_plot(fig_rec, "comparison_boxplots_recall.png", plots_dir, save_plots, "Boxplots Comparativos - Recall por Classe")

        # 4. Especificidade por Classe
        fig_spec, axs_spec = plt.subplots(1, NUM_CLASSES, figsize=(8 * NUM_CLASSES, 7), sharey=True)
        if NUM_CLASSES == 1: axs_spec = [axs_spec]
        for i, ax in enumerate(axs_spec):
            metric = f'specificity_{i}'
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="Set2", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.6)
            ax.set_title(f'Especificidade - {class_names[i]}')
            ax.set_ylabel('Especificidade')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        Plotting._handle_plot(fig_spec, "comparison_boxplots_specificity.png", plots_dir, save_plots, "Boxplots Comparativos - Especificidade por Classe")

    @staticmethod
    def plot_class_distribution(labels, class_names, plots_dir, save_plots, title="Class Distribution", filename="class_distribution.png"):
        """Plota a distribuição das classes no dataset."""
        fig = plt.figure(figsize=(8, 6))
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(range(len(unique)), counts, tick_label=[class_names[i] for i in unique])
        plt.title(title, fontsize=16)
        plt.xlabel('Classes', fontsize=14)
        plt.ylabel('Número de Amostras', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_signal_histograms_per_class(data, labels, class_names, plots_dir, save_plots, title="Signal Histograms per Class", filename="signal_histograms_per_class.png", n_samples=100):
        """Plota histogramas das amplitudes dos sinais por classe."""
        fig, axs = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 5), sharey=True)
        if len(class_names) == 1: axs = [axs]
        
        for i, class_name in enumerate(class_names):
            class_data = data[labels == i]
            if len(class_data) > n_samples:
                indices = np.random.choice(len(class_data), n_samples, replace=False)
                class_data = class_data[indices]
            flat_signals = class_data.flatten()
            axs[i].hist(flat_signals, bins=50, alpha=0.7, edgecolor='black')
            axs[i].set_title(f'{class_name}', fontsize=14)
            axs[i].set_xlabel('Amplitude', fontsize=12)
            axs[i].grid(True, alpha=0.3)
        
        axs[0].set_ylabel('Frequência', fontsize=12)
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_sample_signals(data, labels, class_names, plots_dir, save_plots, title="Sample Signals", filename="sample_signals.png", n_samples_per_class=3):
        """Plota exemplos de sinais para cada classe."""
        fig, axs = plt.subplots(len(class_names), n_samples_per_class, figsize=(4*n_samples_per_class, 3*len(class_names)))
        if len(class_names) == 1: axs = axs.reshape(1, -1)
        
        for i, class_name in enumerate(class_names):
            class_data = data[labels == i]
            if len(class_data) < n_samples_per_class:
                n_samples_per_class = len(class_data)
            indices = np.random.choice(len(class_data), n_samples_per_class, replace=False)
            
            for j, idx in enumerate(indices):
                signal = class_data[idx]
                axs[i, j].plot(signal, linewidth=1)
                axs[i, j].set_title(f'{class_name} - Sample {j+1}', fontsize=10)
                axs[i, j].set_xlabel('Time (samples)', fontsize=8)
                axs[i, j].set_ylabel('Amplitude', fontsize=8)
                axs[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_original_vs_filtered_signals(original_data, filtered_data, plots_dir, save_plots, title="Original vs Filtered Signals", filename="original_vs_filtered.png", n_samples=3):
        """Compara sinais originais e filtrados."""
        fig, axs = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
        
        indices = np.random.choice(len(original_data), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Original
            axs[i, 0].plot(original_data[idx], linewidth=1, color='blue')
            axs[i, 0].set_title(f'Original Signal - Sample {i+1}', fontsize=12)
            axs[i, 0].set_xlabel('Time (samples)', fontsize=10)
            axs[i, 0].set_ylabel('Amplitude', fontsize=10)
            axs[i, 0].grid(True, alpha=0.3)
            
            # Filtered
            axs[i, 1].plot(filtered_data[idx], linewidth=1, color='red')
            axs[i, 1].set_title(f'Filtered Signal - Sample {i+1}', fontsize=12)
            axs[i, 1].set_xlabel('Time (samples)', fontsize=10)
            axs[i, 1].set_ylabel('Amplitude', fontsize=10)
            axs[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_power_spectral_density(data, fs, plots_dir, save_plots, title="Power Spectral Density", filename="psd.png", n_samples=50):
        """Plota a densidade espectral de potência dos sinais."""
        fig = plt.figure(figsize=(10, 6))
        
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = data[indices]
        
        for signal in data:
            freqs, psd = welch(signal, fs=fs, nperseg=1024)
            plt.semilogy(freqs, psd, alpha=0.5, linewidth=0.5)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power Spectral Density', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, fs/2)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_distributions(features, feature_names, plots_dir, save_plots, title="Feature Distributions", filename="feature_distributions.png", max_features=20):
        """Plota distribuições das características extraídas."""
        n_features = min(len(feature_names), max_features)
        fig, axs = plt.subplots((n_features + 4) // 5, 5, figsize=(15, 3*((n_features + 4) // 5)))
        axs = axs.flatten()
        
        for i in range(n_features):
            feature_data = features[:, i]
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                axs[i].hist(valid_data, bins=30, alpha=0.7, edgecolor='black')
                axs[i].set_title(feature_names[i], fontsize=10)
                axs[i].grid(True, alpha=0.3)
        
        for i in range(n_features, len(axs)):
            axs[i].set_visible(False)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_correlation_matrix(features, feature_names, plots_dir, save_plots, title="Feature Correlation Matrix", filename="feature_correlation.png", max_features=50):
        """Plota matriz de correlação das características."""
        n_features = min(features.shape[1], max_features)
        corr_matrix = np.corrcoef(features[:, :n_features].T)
        
        fig = plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                    xticklabels=feature_names[:n_features], yticklabels=feature_names[:n_features],
                    square=True, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_bda_population_fitness(population_fitness_history, plots_dir, save_plots, title="BDA Population Fitness Evolution", filename="bda_population_fitness.png"):
        """Plota a evolução do fitness da população BDA ao longo das iterações."""
        fig = plt.figure(figsize=(10, 6))
        
        # Plot fitness for each agent over iterations
        for agent in range(population_fitness_history.shape[0]):
            plt.plot(population_fitness_history[agent, :], alpha=0.3, linewidth=0.5, color='blue')
        
        # Plot best fitness
        plt.plot(np.min(population_fitness_history, axis=0), linewidth=2, color='red', label='Best Fitness')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_selection_heatmap(feature_selection_history, feature_names, plots_dir, save_plots, title="Feature Selection Frequency Heatmap", filename="feature_selection_heatmap.png"):
        """Plota mapa de calor da frequência de seleção de cada feature."""
        selection_freq = np.mean(feature_selection_history, axis=0)
        
        # Filtra features com frequência > 0
        if feature_names and len(feature_names) == len(selection_freq):
            filtered = [(name, freq) for name, freq in zip(feature_names, selection_freq) if freq > 0]
            if not filtered:
                print("Nenhuma feature foi selecionada em nenhuma execução.")
                return
            filtered_names, filtered_freq = zip(*filtered)
        else:
            filtered_freq = [freq for freq in selection_freq if freq > 0]
            filtered_names = [str(i) for i, freq in enumerate(selection_freq) if freq > 0]
            if not filtered_freq:
                print("Nenhuma feature foi selecionada em nenhuma execução.")
                return
        
        fig = plt.figure(figsize=(max(10, len(filtered_freq) * 0.3), 6))
        bars = plt.bar(range(len(filtered_freq)), filtered_freq, alpha=0.7, color='skyblue')
        plt.xticks(range(len(filtered_names)), filtered_names, rotation=90, fontsize=8)
        plt.title(title, fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Selection Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_roc_curves(results_dict_list, class_names, plots_dir, save_plots, title="ROC Curves Comparison", filename="roc_curves_comparison.png"):
        """Plota curvas ROC agregadas para comparação entre pipelines."""
        from sklearn.metrics import roc_curve, auc
        
        fig, axs = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 5))
        if len(class_names) == 1: axs = [axs]
        
        for pipeline_name, results_list in results_dict_list.items():
            all_y_true = []
            all_y_scores = []
            
            for run_result in results_list:
                if 'final_metrics' not in run_result:
                    continue
                # Note: This would require storing prediction probabilities, which we don't have yet
                # For now, skip this plot or modify to use available data
                pass
        
        # Placeholder - would need prediction probabilities stored
        for ax in axs:
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (Placeholder)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_precision_recall_curves(results_dict_list, class_names, plots_dir, save_plots, title="Precision-Recall Curves Comparison", filename="precision_recall_comparison.png"):
        """Plota curvas Precision-Recall agregadas para comparação entre pipelines."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, axs = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 5))
        if len(class_names) == 1: axs = [axs]
        
        # Similar to ROC, needs prediction probabilities
        for ax in axs:
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve (Placeholder)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_statistical_significance(results_dict_list, plots_dir, save_plots, title="Statistical Significance Tests", filename="statistical_significance.png"):
        """Visualiza testes de significância estatística entre pipelines."""
        from scipy import stats
        
        df_plot = Plotting._extract_metrics_for_plotting(results_dict_list, list(results_dict_list.keys())[0])  # Get class names from first pipeline
        
        metrics = ['accuracy', 'f1_macro']
        fig, axs = plt.subplots(1, len(metrics), figsize=(8*len(metrics), 6))
        if len(metrics) == 1: axs = [axs]
        
        for i, metric in enumerate(metrics):
            pipeline_names = list(results_dict_list.keys())
            data1 = df_plot[df_plot['Pipeline'] == pipeline_names[0]][metric].dropna()
            data2 = df_plot[df_plot['Pipeline'] == pipeline_names[1]][metric].dropna()
            
            # T-test
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            
            # Plot distributions
            axs[i].hist(data1, alpha=0.7, label=pipeline_names[0], bins=15)
            axs[i].hist(data2, alpha=0.7, label=pipeline_names[1], bins=15)
            axs[i].axvline(data1.mean(), color='blue', linestyle='--', label=f'{pipeline_names[0]} Mean')
            axs[i].axvline(data2.mean(), color='orange', linestyle='--', label=f'{pipeline_names[1]} Mean')
            
            axs[i].set_title(f'{metric.title()} Distribution\np-value: {p_val:.4f}', fontsize=14)
            axs[i].set_xlabel(metric.title())
            axs[i].set_ylabel('Frequency')
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_per_run_metrics(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Per-Run Metrics", filename="per_run_metrics.png"):
        """Plota métricas de cada execução individual."""
        df_plot = Plotting._extract_metrics_for_plotting({pipeline_name: results_list}, class_names)
        
        metrics = ['accuracy', 'f1_macro', 'execution_time_min']
        fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), sharex=True)
        if len(metrics) == 1: axs = [axs]
        
        for i, metric in enumerate(metrics):
            axs[i].plot(df_plot['run'], df_plot[metric], marker='o', linestyle='-', alpha=0.7)
            axs[i].set_title(f'{metric.replace("_", " ").title()} per Run', fontsize=14)
            axs[i].set_ylabel(metric.replace("_", " ").title())
            axs[i].grid(True, alpha=0.3)
        
        axs[-1].set_xlabel('Run Number')
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_per_run_confusion_matrices(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Per-Run Confusion Matrices", filename_prefix="cm_run"):
        """Plota matrizes de confusão para cada execução."""
        # Make filename prefix unique per pipeline
        unique_prefix = f"cm_run_{pipeline_name.lower().replace('_', '')}_"
        
        for run_result in results_list:
            if 'final_metrics' not in run_result or 'confusion_matrix' not in run_result['final_metrics']:
                continue
            
            run_id = run_result.get('run_id', 'unknown')
            cm = np.array(run_result['final_metrics']['confusion_matrix'])
            
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 12})
            plt.title(f'Confusion Matrix - {pipeline_name} Run {run_id}', fontsize=16)
            plt.ylabel('True Class', fontsize=14)
            plt.xlabel('Predicted Class', fontsize=14)
            
            filename = f"{unique_prefix}{run_id}.png"
            Plotting._handle_plot(fig, filename, plots_dir, save_plots, f"CM {pipeline_name} Run {run_id}")

    @staticmethod
    def plot_aggregated_confusion_matrix(results_list, pipeline_name, class_names, plots_dir, save_plots):
        """
        Soma as matrizes de confusão de todas as execuções e plota 
        uma matriz normalizada (por linha - "recall").
        """
        print(f"Gerando Matriz de Confusão Agregada para {pipeline_name}...")
        
        # Inicializa a matriz de confusão agregada
        aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        
        for run_result in results_list:
            if run_result and 'final_metrics' in run_result:
                cm = run_result['final_metrics'].get('confusion_matrix')
                if cm:
                    aggregated_cm += np.array(cm)
        
        if np.sum(aggregated_cm) == 0:
            print(f"Nenhuma matriz de confusão encontrada para {pipeline_name}.")
            return
            
        # Normaliza pelo total de amostras verdadeiras (por linha)
        row_sums = aggregated_cm.sum(axis=1)[:, np.newaxis]
        # Evita divisão por zero se uma classe nunca apareceu nos dados
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_cm = aggregated_cm.astype('float') / row_sums
            normalized_cm = np.nan_to_num(normalized_cm) # Substitui nan por 0

        # Cria anotações com a porcentagem e o total
        annotations = np.empty_like(normalized_cm, dtype=object)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                annotations[i, j] = f"{normalized_cm[i, j]:.2%}\n(n={aggregated_cm[i, j]})"
        
        # Plota o heatmap
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(normalized_cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12}, vmin=0.0, vmax=1.0)
        plt.title(f'Matriz de Confusão Agregada (Normalizada por Linha) - {pipeline_name}', fontsize=18)
        plt.ylabel('Classe Verdadeira', fontsize=14)
        plt.xlabel('Classe Predita', fontsize=14)
        
        filename = f"comparison_aggregated_cm_{pipeline_name}.png"
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, f"CM Agregada - {pipeline_name}")

    @staticmethod
    def plot_delta_accuracy_per_run(results_dict_list, plots_dir, save_plots, title="Delta Accuracy per Run", filename="delta_accuracy_per_run.png"):
        """Plota a diferença de acurácia (RHCB5 - BDA) para cada execução."""
        if len(results_dict_list) != 2:
            print("Erro: plot_delta_accuracy_per_run requer exatamente 2 pipelines.")
            return
        
        pipeline_names = list(results_dict_list.keys())
        results1 = results_dict_list[pipeline_names[0]]
        results2 = results_dict_list[pipeline_names[1]]
        
        # Assume RHCB5 é o segundo, BDA o primeiro
        bda_results = results1 if 'BDA' in pipeline_names[0] else results2
        rhcb5_results = results2 if 'RHCB5' in pipeline_names[1] else results1
        
        # Extrai acurácias por run_id
        bda_accs = []
        rhcb5_accs = []
        run_ids = []
        
        for run in bda_results:
            if 'final_metrics' in run and 'accuracy' in run['final_metrics']:
                run_id = run.get('run_id')
                if run_id is not None:
                    run_ids.append(run_id)
                    bda_accs.append(run['final_metrics']['accuracy'])
        
        for run in rhcb5_results:
            if 'final_metrics' in run and 'accuracy' in run['final_metrics']:
                rhcb5_accs.append(run['final_metrics']['accuracy'])
        
        if len(bda_accs) != len(rhcb5_accs):
            print("Erro: Número diferente de execuções válidas entre pipelines.")
            return
        
        diff_acc = np.array(rhcb5_accs) - np.array(bda_accs)
        
        fig = plt.figure(figsize=(12, 6))
        bars = plt.bar(run_ids, diff_acc, color=['red' if x < 0 else 'green' for x in diff_acc], alpha=0.7)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(title, fontsize=16)
        plt.xlabel('Run ID', fontsize=14)
        plt.ylabel('Delta Accuracy (RHCB5 - BDA)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Adiciona valores nas barras
        for bar, val in zip(bars, diff_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.001 if val >= 0 else -0.001), 
                     f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_confusion_matrix_std_heatmap(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Confusion Matrix Std Heatmap", filename="cm_std_heatmap.png"):
        """Plota heatmap do desvio padrão de cada célula da matriz de confusão ao longo das execuções."""
        cms = []
        for run_result in results_list:
            if 'final_metrics' in run_result and 'confusion_matrix' in run_result['final_metrics']:
                cm = np.array(run_result['final_metrics']['confusion_matrix'])
                cms.append(cm)
        
        if not cms:
            print(f"Nenhuma matriz de confusão encontrada para {pipeline_name}.")
            return
        
        cms_array = np.array(cms)  # Shape: (n_runs, n_classes, n_classes)
        std_cm = np.std(cms_array, axis=0)
        
        # Make filename unique per pipeline
        unique_filename = f"cm_std_heatmap_{pipeline_name.lower().replace('_', '')}.png"
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(std_cm, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12})
        plt.title(f'Desvio Padrão da Matriz de Confusão - {pipeline_name}', fontsize=16)
        plt.ylabel('Classe Verdadeira', fontsize=14)
        plt.xlabel('Classe Predita', fontsize=14)
        
        Plotting._handle_plot(fig, unique_filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_selection_frequency(bda_results_list, feature_names, plots_dir, save_plots, title="Feature Selection Frequency", filename="feature_selection_frequency.png"):
        """Plota a frequência de seleção de cada feature ao longo das execuções BDA."""
        selected_vectors = []
        for run_result in bda_results_list:
            if 'selected_features_vector' in run_result:
                vector = np.array(run_result['selected_features_vector'])
                selected_vectors.append(vector)

        if not selected_vectors:
            print("Nenhum vetor de features selecionadas encontrado.")
            return

        # Soma ao longo das runs (axis=0)
        selection_freq = np.sum(selected_vectors, axis=0)

        # Filtra features com frequência > 0
        if feature_names and len(feature_names) == len(selection_freq):
            filtered = [(name, freq) for name, freq in zip(feature_names, selection_freq) if freq > 0]
            if not filtered:
                print("Nenhuma feature foi selecionada em nenhuma execução.")
                return
            filtered_names, filtered_freq = zip(*filtered)
        else:
            filtered_freq = [freq for freq in selection_freq if freq > 0]
            filtered_names = [str(i) for i, freq in enumerate(selection_freq) if freq > 0]
            if not filtered_freq:
                print("Nenhuma feature foi selecionada em nenhuma execução.")
                return

        fig = plt.figure(figsize=(max(10, len(filtered_freq) * 0.3), 6))
        bars = plt.bar(filtered_names, filtered_freq, alpha=0.7, color='skyblue')
        plt.title(title, fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Selection Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=90, fontsize=8)

        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_distribution_plots(results_dict_list, plots_dir, save_plots, title="Distribution Plots", filename="distribution_plots.png"):
        """Plota distribuições das métricas principais."""
        df_plot = Plotting._extract_metrics_for_plotting(results_dict_list, list(results_dict_list.keys())[0])
        
        metrics = ['accuracy', 'f1_macro', 'execution_time_min']
        fig, axs = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1: axs = [axs]
        
        for ax, metric in zip(axs, metrics):
            for pipeline in df_plot['Pipeline'].unique():
                data = df_plot[df_plot['Pipeline'] == pipeline][metric].dropna()
                ax.hist(data, alpha=0.7, label=pipeline, bins=15)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_performance_vs_cost_scatter(results_dict_list, plots_dir, save_plots, title="Performance vs Cost Scatter Plot", filename="performance_vs_cost_scatter.png"):
        """
        Create a scatter plot of execution time vs F1-macro performance for both pipelines.
        
        Args:
            results_dict_list (dict): {'BDA_DNN': [run1_res, ...], 'RHCB5': [run1_res, ...]}
            plots_dir (str): Directory to save plots
            save_plots (bool): Whether to save plots
            title (str): Plot title
            filename (str): Output filename
        """
        print("Generating Performance vs Cost scatter plot...")
        
        fig = plt.figure(figsize=(10, 8))
        
        colors = {'BDA_DNN': 'blue', 'RHCB5': 'red'}
        labels = {'BDA_DNN': 'BDA-DNN', 'RHCB5': 'RHCB5'}
        
        for pipeline_name, results_list in results_dict_list.items():
            exec_times = []
            f1_scores = []
            
            for run_result in results_list:
                if 'final_metrics' in run_result and 'execution_time_sec' in run_result:
                    f1_macro = run_result['final_metrics'].get('classification_report', {}).get('macro avg', {}).get('f1-score')
                    exec_time = run_result['execution_time_sec']
                    
                    if f1_macro is not None and exec_time is not None:
                        f1_scores.append(f1_macro)
                        exec_times.append(exec_time)
            
            if exec_times and f1_scores:
                plt.scatter(exec_times, f1_scores, 
                          c=colors.get(pipeline_name, 'black'), 
                          label=labels.get(pipeline_name, pipeline_name),
                          alpha=0.7, s=50, edgecolors='black')
        
        plt.xlabel('Execution Time (seconds)', fontsize=14)
        plt.ylabel('F1-Macro Score', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some statistics in text
        for pipeline_name, results_list in results_dict_list.items():
            exec_times = []
            f1_scores = []
            
            for run_result in results_list:
                if 'final_metrics' in run_result and 'execution_time_sec' in run_result:
                    f1_macro = run_result['final_metrics'].get('classification_report', {}).get('macro avg', {}).get('f1-score')
                    exec_time = run_result['execution_time_sec']
                    
                    if f1_macro is not None and exec_time is not None:
                        f1_scores.append(f1_macro)
                        exec_times.append(exec_time)
            
            if exec_times and f1_scores:
                mean_time = np.mean(exec_times)
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                
                # Position text near the cluster center
                x_pos = mean_time
                y_pos = mean_f1 + 0.02
                
                plt.text(x_pos, y_pos, 
                        f'{labels.get(pipeline_name, pipeline_name)}\n'
                        f'F1: {mean_f1:.3f}±{std_f1:.3f}\n'
                        f'Time: {mean_time:.1f}s',
                        ha='center', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_sensitivity_analysis(bda_results_list, plots_dir, save_plots, title="Sensitivity Analysis - Number of Features", filename="sensitivity_analysis.png"):
        """Plota análise de sensibilidade: número de features vs acurácia e falsos positivos."""
        num_features_list = []
        accuracies = []
        false_positives_list = []
        
        for run_result in bda_results_list:
            if 'final_metrics' in run_result and 'num_selected_features' in run_result:
                num_feat = run_result['num_selected_features']
                acc = run_result['final_metrics']['accuracy']
                
                # Calculate false positives from confusion matrix
                cm = run_result['final_metrics'].get('confusion_matrix')
                if cm:
                    cm_array = np.array(cm)
                    # False positives = sum of off-diagonal elements
                    fp = np.sum(cm_array) - np.sum(np.diag(cm_array))
                    false_positives_list.append(fp)
                    num_features_list.append(num_feat)
                    accuracies.append(acc)
        
        if not num_features_list:
            print("Nenhum dado válido encontrado para análise de sensibilidade.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Num features vs Accuracy
        ax1.scatter(num_features_list, accuracies, alpha=0.7, s=50, edgecolors='black')
        ax1.set_xlabel('Number of Selected Features', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.set_title('Accuracy vs Number of Features', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(num_features_list) > 1:
            z = np.polyfit(num_features_list, accuracies, 1)
            p = np.poly1d(z)
            ax1.plot(num_features_list, p(num_features_list), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax1.legend()
        
        # Plot 2: Num features vs False Positives
        ax2.scatter(num_features_list, false_positives_list, alpha=0.7, s=50, edgecolors='black', color='red')
        ax2.set_xlabel('Number of Selected Features', fontsize=14)
        ax2.set_ylabel('False Positives', fontsize=14)
        ax2.set_title('False Positives vs Number of Features', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(num_features_list) > 1:
            z = np.polyfit(num_features_list, false_positives_list, 1)
            p = np.poly1d(z)
            ax2.plot(num_features_list, p(num_features_list), "b--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax2.legend()
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)
