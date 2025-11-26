# -*- coding: utf-8 -*-
"""
Módulo de Utilitários do Pipeline de Comparação.
(Versão para TCC UPF - Estilo Acadêmico em Português)

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar erros em servidores/scripts
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuração de Estilo Acadêmico (Template TCC) ---
def set_academic_style():
    """Configura o Matplotlib para gerar gráficos prontos para publicação/TCC."""
    # Tenta usar um estilo base limpo
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid') # Fallback para versões antigas
    
    # Configurações de Fonte e Tamanho para A4
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 12,              # Tamanho base
        'axes.labelsize': 12,         # Tamanho rótulos eixos
        'axes.titlesize': 14,         # Tamanho títulos
        'xtick.labelsize': 10,        # Tamanho ticks X
        'ytick.labelsize': 10,        # Tamanho ticks Y
        'legend.fontsize': 11,        # Tamanho legenda
        'figure.titlesize': 16,       # Título da figura
        'figure.dpi': 900,            # Alta resolução para impressão
        'savefig.dpi': 900,           # Alta resolução ao salvar
        'axes.grid': True,            # Grade ligada
        'grid.alpha': 0.3,            # Grade sutil
        'lines.linewidth': 2.0,       # Linhas mais grossas para visibilidade
        'lines.markersize': 8,        # Marcadores visíveis
    })
    
    # Paleta de cores acessível (Colorblind friendly)
    sns.set_palette("colorblind")

# Aplica o estilo imediatamente
set_academic_style()

# --- Constantes Globais Compartilhadas ---
RANDOM_SEED_GLOBAL = 42 # Seed global para operações de setup
np.random.seed(RANDOM_SEED_GLOBAL)
tf.random.set_seed(RANDOM_SEED_GLOBAL)

# Nomes das classes traduzidos para gráficos
CLASS_NAMES = ["Normal (0)", "Interictal (1)", "Ictal (2)"]
NUM_CLASSES = len(CLASS_NAMES)

# Parâmetros do Dataset e Pré-processamento
FS = 173.61 # Frequência de amostragem
HIGHCUT_HZ = 40.0 # Frequência de corte do filtro passa-baixas
FILTER_ORDER = 4 # Ordem do filtro Butterworth
ORIGINAL_INPUT_LENGTH = 4097 # Comprimento original do sinal Bonn
TARGET_INPUT_LENGTH = 4096 # Comprimento alvo após remoção da 1ª amostra

# Parâmetros da Divisão de Dados
TEST_SIZE = 0.15 # Proporção do conjunto de teste
VAL_SIZE = 0.15 # Proporção do conjunto de validação

# Configuração de Hardware e Análise
USE_GPU = True 
USE_XAI = True 

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
        Aplica filtro Butterworth passa-baixas aos sinais EEG.
        """
        processed_data = np.zeros_like(data)
        nyq = 0.5 * fs
        high = highcut_hz / nyq
        
        if high >= 1.0:
            raise ValueError(f"Frequência de corte ({highcut_hz} Hz) resulta em valor normalizado >= 1.0. Verifique FS e highcut_hz.")
            
        b, a = butter(order, high, btype='low', analog=False)

        print("Iniciando pré-processamento (filtragem)...")
        for i in tqdm(range(data.shape[0]), desc="Pré-processando segmentos", leave=False):
            signal = data[i, :]
            
            # 1. Filtragem
            filtered_signal = filtfilt(b, a, signal)
            
            processed_data[i, :] = filtered_signal.astype(np.float32)
            
        print("Pré-processamento concluído.")
        return processed_data

    @staticmethod
    def normalize_data_split(X_train, X_val, X_test):
        """
        Aplica Instance Standardization (Z-Score por amostra).
        
        Para Deep Learning em EEG, normalizar cada sinal individualmente 
        (subtrair sua própria média e dividir pelo seu próprio desvio padrão)
        geralmente funciona melhor que normalização global, pois remove
        variações de amplitude absoluta entre pacientes e foca na morfologia.
        """
        print("Aplicando Instance Normalization (por segmento)...")
        
        def instance_norm(X):
            # X shape: (n_samples, n_features)
            # Calcula média e std ao longo do eixo 1 (features/tempo)
            means = np.mean(X, axis=1, keepdims=True)
            stds = np.std(X, axis=1, keepdims=True)
            
            # Evita divisão por zero
            stds[stds == 0] = 1.0
            
            return (X - means) / stds

        # Aplica a função independentemente em cada conjunto
        # Como é por amostra, não há "fit" no treino para aplicar no teste.
        # Cada conjunto é tratado isoladamente, garantindo zero leakage.
        X_train_norm = instance_norm(X_train)
        X_val_norm = instance_norm(X_val)
        X_test_norm = instance_norm(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm

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
    """Agrupa todas as funções de plotagem (individuais e agregadas). Refatorado para PT-BR/ABNT."""
    
    @staticmethod
    def _handle_plot(fig, filename, plots_dir, save_plots, title=""):
        """Função auxiliar interna para salvar ou mostrar plot."""
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            filepath = os.path.join(plots_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=900)
                print(f"Plot salvo em: {filepath}", flush=True)
            except Exception as e:
                print(f"Erro ao salvar plot {filepath}: {e}", flush=True)
        # Fecha a figura para liberar memória
        plt.close(fig)

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

    # --- Funções de Plotagem ---

    @staticmethod
    def plot_dnn_training_history(history, plots_dir, save_plots, title, filename):
        """Plota o histórico de treinamento de um modelo Keras."""
        if not history:
            print("Nenhum histórico de treinamento para plotar.", flush=True)
            return

        history_data = history if isinstance(history, dict) else history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if 'loss' in history_data and 'val_loss' in history_data:
            ax1.plot(history_data['loss'], label='Perda (Treino)')
            ax1.plot(history_data['val_loss'], label='Perda (Validação)', linestyle='--')
            ax1.set_title('Função de Perda (Loss)')
            ax1.set_ylabel('Perda')
            ax1.set_xlabel('Época')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if 'accuracy' in history_data and 'val_accuracy' in history_data:
            ax2.plot(history_data['accuracy'], label='Acurácia (Treino)')
            ax2.plot(history_data['val_accuracy'], label='Acurácia (Validação)', linestyle='--')
            ax2.set_title('Evolução da Acurácia')
            ax2.set_ylabel('Acurácia')
            ax2.set_xlabel('Época')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_optimization_diagnostics(curves_dict, plots_dir, save_plots, title, filename):
        """Plota métricas de diagnóstico do otimizador BDA."""
        num_plots = len(curves_dict)
        if num_plots == 0: return
        
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
        if num_plots == 1: axs = [axs]

        label_map = {
            "Melhor Fitness": "Melhor Aptidão (Fitness)",
            "Acurácia KNN (CV)": "Acurácia KNN (Validação Cruzada)",
            "Nº de Features": "Nº de Atributos Selecionados"
        }

        iterations = range(len(next(iter(curves_dict.values()))))

        for ax, (metric_name, curve_data) in zip(axs, curves_dict.items()):
            pt_label = label_map.get(metric_name, metric_name)
            ax.plot(iterations, curve_data, marker=".", markersize=3)
            ax.set_title(f"Evolução: {pt_label}")
            ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3)

        axs[-1].set_xlabel("Iteração")
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

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

        metric_map = {
            'accuracy': 'Acurácia Global',
            'f1_macro': 'F1-Score (Macro)',
            'execution_time_min': 'Tempo de Execução (min)'
        }
            
        # 1. Métricas Gerais
        general_metrics = ['accuracy', 'f1_macro', 'execution_time_min']
        fig_gen, axs_gen = plt.subplots(1, len(general_metrics), figsize=(6 * len(general_metrics), 6))
        if len(general_metrics) == 1: axs_gen = [axs_gen]
        
        for ax, metric in zip(axs_gen, general_metrics):
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="colorblind", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.5, size=4)
            ax.set_title(metric_map.get(metric, metric))
            ax.set_ylabel(metric_map.get(metric, metric).split(" (")[0])
            ax.set_xlabel("Abordagem")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        Plotting._handle_plot(fig_gen, "comparison_boxplots_general.png", plots_dir, save_plots, "Comparativo Geral")

        # 2. Número de Features (Apenas BDA-DNN)
        df_bda = df_plot[df_plot['Pipeline'] == 'BDA_DNN']
        if not df_bda['num_selected_features'].isnull().all():
            fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Pipeline', y='num_selected_features', data=df_bda, ax=ax_feat, hue='Pipeline', palette="colorblind", legend=False)
            sns.stripplot(x='Pipeline', y='num_selected_features', data=df_bda, ax=ax_feat, color=".25", alpha=0.5)
            ax_feat.set_title('Nº de Atributos Selecionados (BDA-DNN)')
            ax_feat.set_ylabel('Quantidade')
            ax_feat.set_xlabel("Abordagem")
            ax_feat.grid(axis='y', linestyle='--', alpha=0.5)
            Plotting._handle_plot(fig_feat, "comparison_boxplot_features.png", plots_dir, save_plots, "Features BDA")
        
        # 3. Recall por Classe
        fig_rec, axs_rec = plt.subplots(1, NUM_CLASSES, figsize=(6 * NUM_CLASSES, 6), sharey=True)
        if NUM_CLASSES == 1: axs_rec = [axs_rec]
        for i, ax in enumerate(axs_rec):
            metric = f'recall_{i}'
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="colorblind", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.5, size=3)
            ax.set_title(f'Sensibilidade (Recall)\n{class_names[i]}')
            ax.set_ylabel('Recall')
            ax.set_xlabel("Abordagem")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        Plotting._handle_plot(fig_rec, "comparison_boxplots_recall.png", plots_dir, save_plots, "Recall por Classe")

        # 4. Especificidade por Classe
        fig_spec, axs_spec = plt.subplots(1, NUM_CLASSES, figsize=(6 * NUM_CLASSES, 6), sharey=True)
        if NUM_CLASSES == 1: axs_spec = [axs_spec]
        for i, ax in enumerate(axs_spec):
            metric = f'specificity_{i}'
            sns.boxplot(x='Pipeline', y=metric, data=df_plot, ax=ax, hue='Pipeline', palette="colorblind", legend=False)
            sns.stripplot(x='Pipeline', y=metric, data=df_plot, ax=ax, color=".25", alpha=0.5, size=3)
            ax.set_title(f'Especificidade\n{class_names[i]}')
            ax.set_ylabel('Especificidade')
            ax.set_xlabel("Abordagem")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        Plotting._handle_plot(fig_spec, "comparison_boxplots_specificity.png", plots_dir, save_plots, "Especificidade por Classe")

    @staticmethod
    def plot_original_vs_filtered_signals(original_data, filtered_data, plots_dir, save_plots, title="Original vs Filtrado", filename="original_vs_filtered.png", n_samples=3):
        """Compara sinais originais e filtrados."""
        fig, axs = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
        
        indices = np.random.choice(len(original_data), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Original
            axs[i, 0].plot(original_data[idx], linewidth=1, color='tab:blue')
            axs[i, 0].set_title(f'Sinal Original - Amostra {i+1}')
            axs[i, 0].set_ylabel('Amplitude')
            axs[i, 0].grid(True, alpha=0.3)
            
            # Filtered
            axs[i, 1].plot(filtered_data[idx], linewidth=1, color='tab:red')
            axs[i, 1].set_title(f'Sinal Filtrado - Amostra {i+1}')
            axs[i, 1].set_ylabel('Amplitude')
            axs[i, 1].grid(True, alpha=0.3)
        
        axs[-1, 0].set_xlabel('Tempo (amostras)')
        axs[-1, 1].set_xlabel('Tempo (amostras)')
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_power_spectral_density(data, fs, plots_dir, save_plots, title="Densidade Espectral de Potência (PSD)", filename="psd.png", n_samples=50):
        """Plota a densidade espectral de potência."""
        fig = plt.figure(figsize=(10, 6))
        
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = data[indices]
        
        for signal in data:
            freqs, psd = welch(signal, fs=fs, nperseg=1024)
            plt.semilogy(freqs, psd, alpha=0.4, linewidth=0.5)
        
        plt.title(title)
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Densidade Espectral de Potência (V²/Hz)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, fs/2)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_selection_heatmap(feature_selection_history, feature_names, plots_dir, save_plots, title="Mapa de Calor: Frequência de Seleção", filename="feature_selection_heatmap.png"):
        """Plota mapa de calor da frequência de seleção de cada feature."""
        selection_freq = np.mean(feature_selection_history, axis=0)
        
        if feature_names and len(feature_names) == len(selection_freq):
            filtered = [(name, freq) for name, freq in zip(feature_names, selection_freq) if freq > 0]
            if not filtered:
                print("Nenhuma feature foi selecionada.")
                return
            filtered_names, filtered_freq = zip(*filtered)
        else:
            filtered_freq = [freq for freq in selection_freq if freq > 0]
            filtered_names = [str(i) for i, freq in enumerate(selection_freq) if freq > 0]
            if not filtered_freq:
                print("Nenhuma feature foi selecionada.")
                return
        
        fig = plt.figure(figsize=(max(10, len(filtered_freq) * 0.3), 6))
        plt.bar(range(len(filtered_freq)), filtered_freq, alpha=0.7, color='tab:blue')
        plt.xticks(range(len(filtered_names)), filtered_names, rotation=90, fontsize=8)
        plt.title(title)
        plt.xlabel('Atributos (Features)')
        plt.ylabel('Frequência de Seleção (Média)')
        plt.grid(True, axis='y', alpha=0.3)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_aggregated_confusion_matrix(results_list, pipeline_name, class_names, plots_dir, save_plots):
        """Soma as matrizes de confusão e plota normalizada."""
        aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        
        for run_result in results_list:
            if run_result and 'final_metrics' in run_result:
                cm = run_result['final_metrics'].get('confusion_matrix')
                if cm: aggregated_cm += np.array(cm)
        
        if np.sum(aggregated_cm) == 0: return
            
        # Normaliza por linha (Recall)
        row_sums = aggregated_cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_cm = aggregated_cm.astype('float') / row_sums
            normalized_cm = np.nan_to_num(normalized_cm)

        annotations = np.empty_like(normalized_cm, dtype=object)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                annotations[i, j] = f"{normalized_cm[i, j]:.1%}\n(N={aggregated_cm[i, j]})"
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12}, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Proporção'})
        
        plt.title(f'Matriz de Confusão Agregada - {pipeline_name}')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        
        Plotting._handle_plot(fig, f"comparison_aggregated_cm_{pipeline_name}.png", plots_dir, save_plots)

    @staticmethod
    def plot_statistical_significance(results_dict_list, plots_dir, save_plots, title="Testes de Significância", filename="statistical_significance.png"):
        """Visualiza testes de significância (Histogramas comparativos)."""
        from scipy import stats
        df_plot = Plotting._extract_metrics_for_plotting(results_dict_list, list(results_dict_list.keys())[0]) # Class names dummy
        
        metrics = ['accuracy', 'f1_macro']
        metric_labels = {'accuracy': 'Acurácia', 'f1_macro': 'F1-Score'}
        
        fig, axs = plt.subplots(1, len(metrics), figsize=(14, 6))
        if len(metrics) == 1: axs = [axs]
        
        for i, metric in enumerate(metrics):
            pipeline_names = list(results_dict_list.keys())
            if len(pipeline_names) < 2: continue
            
            data1 = df_plot[df_plot['Pipeline'] == pipeline_names[0]][metric].dropna()
            data2 = df_plot[df_plot['Pipeline'] == pipeline_names[1]][metric].dropna()
            
            if len(data1) < 2 or len(data2) < 2: continue

            # T-test
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            
            axs[i].hist(data1, alpha=0.6, label=pipeline_names[0], bins=15, density=True)
            axs[i].hist(data2, alpha=0.6, label=pipeline_names[1], bins=15, density=True)
            
            # Linhas verticais nas médias
            axs[i].axvline(data1.mean(), color='C0', linestyle='--', linewidth=2, label=f'Média {pipeline_names[0]}')
            axs[i].axvline(data2.mean(), color='C1', linestyle='--', linewidth=2, label=f'Média {pipeline_names[1]}')
            
            axs[i].set_title(f'Distribuição - {metric_labels[metric]}\n(Teste-T p-valor: {p_val:.2e})')
            axs[i].set_xlabel(metric_labels[metric])
            axs[i].set_ylabel('Densidade de Frequência')
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_delta_accuracy_per_run(results_dict_list, plots_dir, save_plots, title="Diferença de Acurácia por Execução", filename="delta_accuracy_per_run.png"):
        """Plota a diferença de acurácia (RHCB5 - BDA) pareada."""
        if len(results_dict_list) != 2: return
        
        pipeline_names = list(results_dict_list.keys())
        # Tenta identificar qual é qual
        name_bda = next((name for name in pipeline_names if 'BDA' in name), pipeline_names[0])
        name_rhcb5 = next((name for name in pipeline_names if 'RHCB5' in name), pipeline_names[1])
        
        results_bda = results_dict_list[name_bda]
        results_rhcb5 = results_dict_list[name_rhcb5]
        
        # Extrai acurácias pareadas pelo ID
        acc_dict_bda = {r['run_id']: r['final_metrics']['accuracy'] for r in results_bda if 'final_metrics' in r}
        acc_dict_rhcb5 = {r['run_id']: r['final_metrics']['accuracy'] for r in results_rhcb5 if 'final_metrics' in r}
        
        common_ids = sorted(list(set(acc_dict_bda.keys()) & set(acc_dict_rhcb5.keys())))
        
        if not common_ids: return
            
        diff_acc = np.array([acc_dict_rhcb5[rid] - acc_dict_bda[rid] for rid in common_ids])
        
        fig = plt.figure(figsize=(12, 6))
        bars = plt.bar(common_ids, diff_acc, color=['tab:red' if x < 0 else 'tab:green' for x in diff_acc], alpha=0.7)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(title)
        plt.xlabel('ID da Execução (Run)')
        plt.ylabel(f'Diferença Acurácia\n({name_rhcb5} - {name_bda})')
        plt.grid(True, alpha=0.3)
        
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_performance_vs_cost_scatter(results_dict_list, plots_dir, save_plots, title="Dispersão: Desempenho vs Custo", filename="performance_vs_cost_scatter.png"):
        """Scatter plot de tempo vs F1-score."""
        fig = plt.figure(figsize=(10, 8))
        
        # Mapeamento de cores
        colors = {'BDA_DNN': 'tab:blue', 'RHCB5': 'tab:orange'}
        labels_legend = {'BDA_DNN': 'BDA-DNN (Pipeline)', 'RHCB5': 'RHCB5 (End-to-End)'}
        
        for pipeline_name, results_list in results_dict_list.items():
            exec_times = []
            f1_scores = []
            
            for run_result in results_list:
                if 'final_metrics' in run_result and 'execution_time_sec' in run_result:
                    f1 = run_result['final_metrics'].get('classification_report', {}).get('macro avg', {}).get('f1-score')
                    time_val = run_result['execution_time_sec']
                    if f1 and time_val:
                        f1_scores.append(f1)
                        exec_times.append(time_val)
            
            if exec_times:
                # Plot dos pontos
                plt.scatter(exec_times, f1_scores, 
                          c=colors.get(pipeline_name, 'black'), 
                          label=labels_legend.get(pipeline_name, pipeline_name),
                          alpha=0.6, s=60, edgecolors='white')
                
                # Plot da média (centróide)
                mean_time = np.mean(exec_times)
                mean_f1 = np.mean(f1_scores)
                plt.scatter([mean_time], [mean_f1], c=colors.get(pipeline_name, 'black'), 
                           s=200, marker='X', edgecolors='black', label=f'Média {pipeline_name}')

        plt.xlabel('Tempo de Execução (segundos)')
        plt.ylabel('F1-Score (Macro)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_sensitivity_analysis(bda_results_list, plots_dir, save_plots, title="Análise de Sensibilidade: Features vs Acurácia", filename="sensitivity_analysis.png"):
        """Plota número de features vs acurácia e falsos positivos."""
        num_features_list = []
        accuracies = []
        false_positives_list = []
        
        for run_result in bda_results_list:
            if 'final_metrics' in run_result and 'num_selected_features' in run_result:
                num_feat = run_result['num_selected_features']
                acc = run_result['final_metrics']['accuracy']
                cm = run_result['final_metrics'].get('confusion_matrix')
                if cm:
                    cm_array = np.array(cm)
                    fp = np.sum(cm_array) - np.sum(np.diag(cm_array))
                    false_positives_list.append(fp)
                    num_features_list.append(num_feat)
                    accuracies.append(acc)
        
        if not num_features_list: return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Num features vs Acurácia
        ax1.scatter(num_features_list, accuracies, alpha=0.7, s=50, edgecolors='black', color='tab:purple')
        ax1.set_xlabel('Nº Atributos Selecionados')
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Impacto na Acurácia')
        ax1.grid(True, alpha=0.3)
        
        # Tendência linear simples
        if len(num_features_list) > 1:
            z = np.polyfit(num_features_list, accuracies, 1)
            p = np.poly1d(z)
            ax1.plot(num_features_list, p(num_features_list), "k--", alpha=0.5, label='Tendência')
            ax1.legend()
        
        # Plot 2: Num features vs Falsos Positivos
        ax2.scatter(num_features_list, false_positives_list, alpha=0.7, s=50, edgecolors='black', color='tab:red')
        ax2.set_xlabel('Nº Atributos Selecionados')
        ax2.set_ylabel('Total Falsos Positivos')
        ax2.set_title('Impacto nos Erros (Falsos Positivos)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_per_run_metrics(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Métricas por Execução", filename="per_run_metrics.png"):
        """Plota métricas individuais ao longo das execuções."""
        df_plot = Plotting._extract_metrics_for_plotting({pipeline_name: results_list}, class_names)
        if df_plot.empty: return

        metrics = ['accuracy', 'f1_macro', 'execution_time_min']
        metric_labels = {'accuracy': 'Acurácia', 'f1_macro': 'F1-Score', 'execution_time_min': 'Tempo (min)'}
        
        fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), sharex=True)
        if len(metrics) == 1: axs = [axs]
        
        for i, metric in enumerate(metrics):
            axs[i].plot(df_plot['run'], df_plot[metric], marker='o', linestyle='-', alpha=0.7)
            axs[i].set_title(f'{metric_labels[metric]} por Execução')
            axs[i].set_ylabel(metric_labels[metric])
            axs[i].grid(True, alpha=0.3)
        
        axs[-1].set_xlabel('Número da Execução')
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_confusion_matrix_std_heatmap(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Desvio Padrão da Matriz de Confusão", filename="cm_std_heatmap.png"):
        """Plota desvio padrão das células da matriz de confusão."""
        cms = []
        for run_result in results_list:
            if 'final_metrics' in run_result and 'confusion_matrix' in run_result['final_metrics']:
                cms.append(np.array(run_result['final_metrics']['confusion_matrix']))
        
        if not cms: return
        
        cms_array = np.array(cms)
        std_cm = np.std(cms_array, axis=0)
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(std_cm, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12})
        
        plt.title(f'{title} - {pipeline_name}')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        
        unique_filename = f"cm_std_heatmap_{pipeline_name.lower().replace('_', '')}.png"
        Plotting._handle_plot(fig, unique_filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_per_run_confusion_matrices(results_list, pipeline_name, class_names, plots_dir, save_plots, title="Per-Run CM", filename_prefix="cm_run"):
        """Plota CMs individuais (opcional, gera muitos arquivos)."""
        unique_prefix = f"cm_run_{pipeline_name.lower().replace('_', '')}_"
        for run_result in results_list:
            if 'final_metrics' not in run_result: continue
            run_id = run_result.get('run_id', 'unknown')
            cm = np.array(run_result['final_metrics']['confusion_matrix'])
            
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Matriz de Confusão - {pipeline_name} Run {run_id}')
            plt.ylabel('Classe Verdadeira')
            plt.xlabel('Classe Predita')
            
            Plotting._handle_plot(fig, f"{unique_prefix}{run_id}.png", plots_dir, save_plots)
    
    @staticmethod
    def plot_distribution_plots(results_dict_list, plots_dir, save_plots, title="Distribuição das Métricas", filename="distribution_plots.png"):
        """Histogramas simples das métricas."""
        df_plot = Plotting._extract_metrics_for_plotting(results_dict_list, list(results_dict_list.keys())[0])
        metrics = ['accuracy', 'f1_macro']
        labels = {'accuracy': 'Acurácia', 'f1_macro': 'F1-Score'}

        fig, axs = plt.subplots(1, len(metrics), figsize=(12, 5))
        if len(metrics) == 1: axs = [axs]

        for ax, metric in zip(axs, metrics):
            for pipeline in df_plot['Pipeline'].unique():
                data = df_plot[df_plot['Pipeline'] == pipeline][metric].dropna()
                ax.hist(data, alpha=0.6, label=pipeline, bins=10)
            ax.set_xlabel(labels[metric])
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_distributions(features, feature_names, plots_dir, save_plots, title="Distribuição de Features", filename="feature_distributions.png", max_features=20):
        """Plota histograma de algumas features extraídas."""
        n_features = min(len(feature_names), max_features)
        cols = 5
        rows = (n_features + cols - 1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 3*rows))
        axs = axs.flatten()
        
        for i in range(n_features):
            valid_data = features[:, i][~np.isnan(features[:, i])]
            if len(valid_data) > 0:
                axs[i].hist(valid_data, bins=30, alpha=0.7, edgecolor='black', color='tab:cyan')
                axs[i].set_title(feature_names[i], fontsize=9)
                axs[i].grid(True, alpha=0.3)
        
        # Oculta eixos extras
        for i in range(n_features, len(axs)): axs[i].set_visible(False)
        
        plt.tight_layout()
        Plotting._handle_plot(fig, filename, plots_dir, save_plots, title)

    @staticmethod
    def plot_feature_selection_frequency(bda_results_list, feature_names, plots_dir, save_plots, title="Frequência de Seleção de Features", filename="feature_selection_frequency.png"):
        """Plota a frequência de seleção de features ao longo das execuções BDA."""
        # Coleta feature_selection_history de todas as runs válidas
        histories = []
        for run_result in bda_results_list:
            if 'feature_selection_history' in run_result:
                histories.append(np.array(run_result['feature_selection_history']))
        
        if not histories:
            print("Nenhum histórico de seleção de features encontrado.")
            return
        
        # Empilha os históricos (shape: (n_runs, n_iterations, n_features))
        histories_array = np.array(histories)
        
        # Calcula a frequência média de seleção por feature ao longo das iterações e runs
        # Média sobre runs e iterações
        selection_freq = np.mean(histories_array, axis=(0, 1))
        
        # Chama o método existente para plotar
        Plotting.plot_feature_selection_heatmap(
            histories_array.mean(axis=0),  # Média sobre runs para cada iteração
            feature_names, plots_dir, save_plots, title, filename
        )