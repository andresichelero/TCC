# -*- coding: utf-8 -*-
"""
Pipeline Completo para Detecção de Epilepsia com a arquitetura RHCB5.

Este script implementa o fluxo de trabalho de ponta a ponta para classificar
sinais de EEG do dataset da Universidade de Bonn em três classes:
Normal, Interictal e Ictal.

A metodologia substitui a extração e seleção manual de características (como SWT+BDA/BPSO)
por uma rede neural profunda híbrida (CNN + Bi-LSTM) que aprende as características
relevantes diretamente do sinal.

Passos do Pipeline:
1.  Carregamento e Preparação dos Dados:
    - Carrega os segmentos de EEG dos diretórios correspondentes.
    - Atribui os rótulos (0: Normal, 1: Interictal, 2: Ictal).
    - Ajusta o comprimento de cada sinal de 4097 para 4096 amostras, removendo o primeiro ponto.
2.  Pré-processamento do Sinal:
    - Aplica um filtro Butterworth passa-baixas para remover ruídos.
    - Normaliza os sinais para a faixa [-1, 1] usando Min-Max.
3.  Divisão dos Dados:
    - Separa os dados em conjuntos de treinamento (70%), validação (15%) e teste (15%) de forma estratificada.
4.  Construção do Modelo RHCB5, seguindo a arquitetura proposta por Augusto Maggioni (2023/2024):
    - Define a arquitetura da Rede Híbrida Convolucional Bidirecional adaptada para 4096 pontos de entrada.
5.  Treinamento do Modelo:
    - Compila o modelo com otimizador Adam e função de perda apropriada.
    - Utiliza EarlyStopping para prevenir overfitting e ModelCheckpoint para salvar o melhor modelo.
    - Treina a rede com os dados preparados.
6.  Avaliação de Desempenho:
    - Carrega o melhor modelo salvo durante o treinamento.
    - Avalia a performance no conjunto de teste.
    - Gera e exibe um relatório de classificação detalhado e a matriz de confusão.
7.  Visualização:
    - Plota as curvas de aprendizado (acurácia e perda) do treinamento.
    - Plota uma matriz de confusão visualmente clara.
"""
# --- 1. IMPORTAÇÕES ---
import os
import sys
import time
import datetime
import json

# Bibliotecas de manipulação de dados e computação científica
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

# Bibliotecas de Machine Learning e Deep Learning
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Bibliotecas de visualização
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

# Bibliotecas de Machine Learning e Deep Learning
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# --- 2. CONFIGURAÇÕES GLOBAIS ---

# Configuração de diretórios
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(current_dir, "data")
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_RESULTS_DIR = os.path.join(current_dir, "results")
RUN_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, f"run_rhcb5_{run_timestamp}")
PLOTS_DIR = os.path.join(RUN_RESULTS_DIR, "plots") # Diretório para salvar os plots
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Salvando resultados nesta execução em: {RUN_RESULTS_DIR}")

# Semente para reprodutibilidade
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configurações de plots e logs
SAVE_PLOTS = True # Mude para False para mostrar plots interativamente

# Parâmetros do Pipeline RHCB5
FS = 173.61
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
ORIGINAL_INPUT_LENGTH = 4097
TARGET_INPUT_LENGTH = 4096

# Parâmetros da divisão de dados
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Parâmetros de treino do modelo
NUM_EPOCHS = 100
BATCH_SIZE = 32
MODEL_SAVE_PATH = os.path.join(RUN_RESULTS_DIR, 'best_rhcb5_epilepsy_model.h5')

# Definição das classes e seus diretórios correspondentes
CLASSES = {
    'Normal': {'label': 0, 'folder': 'A'},
    'Interictal': {'label': 1, 'folder': 'D'},
    'Ictal': {'label': 2, 'folder': 'E'}
}
CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASSES)

# --- 3. DEFINIÇÃO DE CLASSES E FUNÇÕES ---

# Do ficheiro: src/utils.py
class NumpyEncoder(json.JSONEncoder):
    """ Classe helper para serializar objetos NumPy para JSON. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Do ficheiro: src/data_loader.py
def load_bonn_data(base_path, target_length=4096):
    """
    Carrega os dados dos conjuntos A, D, E do dataset BONN.
    Assume que os arquivos .txt estão dentro de subpastas 'A', 'D', 'E'.
    Remove a primeira amostra de cada sinal para ajustar o comprimento.
    """
    data_segments = []
    labels = []
    sets_labels = {'A': 0, 'D': 1, 'E': 2}
    original_segment_length = 4097

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
                if len(segment_data) == original_segment_length:
                    adjusted_signal = segment_data[1:]
                    data_segments.append(adjusted_signal)
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

# Do ficheiro: src/data_loader.py
def preprocess_eeg(data, fs=173.61, highcut_hz=40.0, order=4):
    """
    Aplica filtro Butterworth e normalização Min-Max para [-1, 1] aos sinais EEG.
    """
    processed_data = np.zeros_like(data)
    nyq = 0.5 * fs
    high = highcut_hz / nyq
    b, a = butter(order, high, btype='low', analog=False)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Iniciando pré-processamento (filtragem e normalização)...")
    for i in tqdm(range(data.shape[0]), desc="Pré-processando segmentos"):
        signal = data[i, :]
        filtered_signal = filtfilt(b, a, signal)
        reshaped_signal = filtered_signal.reshape(-1, 1)
        normalized_signal = scaler.fit_transform(reshaped_signal).flatten()
        processed_data[i, :] = normalized_signal
        
    print("Pré-processamento concluído.")
    return processed_data

# Do ficheiro: src/data_loader.py
def split_data(data, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Divide os dados em conjuntos de treino, validação e teste estratificados.
    """
    if not (0 < test_size < 1) or not (0 < val_size < 1) or not (0 < test_size + val_size < 1):
        raise ValueError("test_size e val_size devem estar entre 0 e 1, e sua soma deve ser menor que 1.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )

    print("Dados divididos em conjuntos de treino, validação e teste:")
    print(f"  Treino:    {X_train.shape[0]} amostras, {X_train.shape[1]} pontos de tempo")
    print(f"  Validação: {X_val.shape[0]} amostras")
    print(f"  Teste:     {X_test.shape[0]} amostras")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Do ficheiro: src/dnn_model.py
def build_rhcb5_model(input_shape, num_classes):
    """
    Constrói a arquitetura do modelo RHCB5, usando os artigos "Inteligência Artificial aplicada 
    a análise de Eletrocardiograma" e "DETECÇÃO DE ARRITMIAS CARDÍACAS: ABORDAGEM DA 
    DERIVAÇÃO I COM REDES NEURAIS HÍBRIDAS", de Augusto Felipe Maggioni.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=False))(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs, name="RHCB5_Epilepsy_Model")
    return model

# Do ficheiro: src/utils.py
def _handle_plot(fig, filename, title=""):
    """ Função auxiliar para salvar ou mostrar plots. """
    if SAVE_PLOTS:
        filepath = os.path.join(PLOTS_DIR, filename)
        fig.savefig(filepath)
        print(f"Plot salvo em: {filepath}", flush=True)
        plt.close(fig)
    else:
        if title: fig.suptitle(title, fontsize=16)
        plt.show()

# Do ficheiro: src/utils.py
def calculate_specificity(y_true, y_pred, class_label):
    """ Calcula a especificidade para uma classe específica. """
    cm = confusion_matrix(y_true, y_pred)
    tn = np.sum(cm) - (np.sum(cm[class_label, :]) + np.sum(cm[:, class_label]) - cm[class_label, class_label])
    fp = np.sum(cm[:, class_label]) - cm[class_label, class_label]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Do ficheiro: src/utils.py
def calculate_all_metrics(y_true, y_pred, class_names=None):
    """ Calcula e imprime acurácia, relatório de classificação e especificidade por classe. """
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": report_dict["accuracy"],
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    print(f"\nMatriz de Confusão:\n{cm}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    specificities = {}
    print("Especificidade por classe:", flush=True)
    for i, name in enumerate(class_names):
        spec = calculate_specificity(y_true, y_pred, class_label=i)
        print(f"  - {name}: {spec:.4f}", flush=True)
        specificities[f"specificity_{name.lower()}"] = spec
        
    metrics["specificities"] = specificities
    return metrics

# Do ficheiro: src/utils.py
def plot_dnn_training_history(history, title, filename):
    """ Plota o histórico de treinamento de um modelo Keras. """
    if not history: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['loss'], label='Loss Treino')
    ax1.plot(history['val_loss'], label='Loss Validação')
    ax1.set_title('Loss do Modelo')
    ax1.set_xlabel('Época'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['accuracy'], label='Acurácia Treino')
    ax2.plot(history['val_accuracy'], label='Acurácia Validação')
    ax2.set_title('Acurácia do Modelo')
    ax2.set_xlabel('Época'); ax2.set_ylabel('Acurácia')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    _handle_plot(fig, filename, title)

# Do ficheiro: src/utils.py
def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """ Plota uma matriz de confusão visualmente informativa. """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Matriz de Confusão no Conjunto de Teste', fontsize=18)
    plt.ylabel('Classe Verdadeira', fontsize=14)
    plt.xlabel('Classe Predita', fontsize=14)
    _handle_plot(fig, filename, "Matriz de Confusão")


def plot_eeg_segments(segments_dict, fs, n_segments_to_plot=1, base_filename="eeg_segment"):
    """
    Plota exemplos de segmentos EEG e seus espectros de frequência (PSD).
    """
    if not segments_dict:
        return

    num_types = len(segments_dict)
    keys = list(segments_dict.keys())
    
    min_available_segments = min(data.shape[0] for data in segments_dict.values() if data is not None and data.ndim > 1)
    if min_available_segments == 0:
        print("Aviso: Nenhum segmento disponível para plotar.")
        return

    segments_to_plot_actual = min(n_segments_to_plot, min_available_segments)

    for i in range(segments_to_plot_actual):
        fig, axs = plt.subplots(num_types, 2, figsize=(15, 5 * num_types), squeeze=False)

        for row_idx, key in enumerate(keys):
            segment = segments_dict[key][i, :]
            time_vector = np.arange(segment.size) / fs

            # Plot do sinal no domínio do tempo
            axs[row_idx, 0].plot(time_vector, segment)
            axs[row_idx, 0].set_title(f"{key} - Segmento {i} (Domínio do Tempo)")
            axs[row_idx, 0].set_xlabel("Tempo (s)")
            axs[row_idx, 0].set_ylabel("Amplitude")
            axs[row_idx, 0].grid(True)

            # Plot do espectro de frequência (usando Welch)
            frequencies, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
            axs[row_idx, 1].semilogy(frequencies, psd)
            axs[row_idx, 1].set_title(f"Espectro {key} - Segmento {i} (PSD)")
            axs[row_idx, 1].set_xlabel("Frequência (Hz)")
            axs[row_idx, 1].set_ylabel("PSD (V^2/Hz)")
            axs[row_idx, 1].grid(True)
            axs[row_idx, 1].set_xlim(0, fs / 2)

        plt.tight_layout()
        _handle_plot(fig, f"{base_filename}_example_{i}.png", f"Exemplo de Segmento EEG {i}")


def plot_data_distribution_pca(X_dict, y_dict, title, filename, class_names=None):
    """
    Plota a distribuição dos diferentes conjuntos de dados usando PCA.
    """
    if not X_dict or not y_dict: return

    combined_X, combined_y, set_labels = [], [], []
    for set_name, X_data in X_dict.items():
        # Remove a dimensão do 'canal' antes do PCA
        X_data_flat = X_data.squeeze(axis=-1)
        combined_X.append(X_data_flat)
        combined_y.append(y_dict[set_name])
        set_labels.extend([set_name] * X_data.shape[0])

    if not combined_X: return

    X_all = np.vstack(combined_X)
    y_all = np.concatenate(combined_y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    if X_scaled.shape[1] < 2:
        print("Aviso: PCA para 2D não é possível com menos de 2 features.")
        return

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_result = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df_pca['label'] = y_all
    df_pca['set'] = set_labels
    
    var_ratio = pca.explained_variance_ratio_
    print(f"PCA: Variância explicada pelos 2 componentes: {var_ratio.sum()*100:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(
        x="PC1", y="PC2", hue="label", style="set", data=df_pca,
        palette="plasma", s=50, alpha=0.7, ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel(f'Componente Principal 1 ({var_ratio[0]*100:.2f}%)')
    ax.set_ylabel(f'Componente Principal 2 ({var_ratio[1]*100:.2f}%)')
    ax.grid(True)
    
    handles, labels = ax.get_legend_handles_labels()
    # Lógica para separar legendas de cor (classe) e estilo (conjunto)
    num_classes = len(np.unique(y_all))
    hue_handles = handles[1:num_classes+1]
    hue_labels = [class_names[int(lbl)] for lbl in labels[1:num_classes+1]] if class_names else labels[1:num_classes+1]
    style_handles = handles[num_classes+2:]
    style_labels = labels[num_classes+2:]
    
    leg1 = ax.legend(hue_handles, hue_labels, title='Classe', loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg1)
    ax.legend(style_handles, style_labels, title='Conjunto', loc='lower left', bbox_to_anchor=(1.01, 0))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    _handle_plot(fig, filename, title)


# --- 4. SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    start_time_total = time.time()
    print("Iniciando Pipeline de Detecção de Epilepsia com RHCB5...")
    print(f"Usando TensorFlow versão: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs disponíveis: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")
    else:
        print("Nenhuma GPU encontrada. Usando CPU.")

    # Etapa 1: Carregar Dados
    print("\n--- Etapa 1: Carregando e Preparando os Dados ---")
    try:
        raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR, target_length=TARGET_INPUT_LENGTH)
    except Exception as e:
        print(f"Falha ao carregar dados: {e}. Verifique o caminho e formato do dataset.")
        sys.exit(1)

    # Etapa 2: Pré-processar Sinais
    print("\n--- Etapa 2: Pré-processando Sinais ---")
    data_processed = preprocess_eeg(raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER)
    X = np.expand_dims(data_processed, axis=-1)
    y = raw_labels
    print(f"Formato final dos dados de entrada (X): {X.shape}")

    # Etapa 3: Dividir Dados
    print("\n--- Etapa 3: Dividindo Dados ---")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # Etapa 4: Construir e Compilar o Modelo
    print("\n--- Etapa 4: Construindo o Modelo RHCB5 ---")
    input_shape = (TARGET_INPUT_LENGTH, 1)
    model = build_rhcb5_model(input_shape, NUM_CLASSES)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Etapa 5: Treinar o Modelo
    print("\n--- Etapa 5: Iniciando Treinamento do Modelo ---")
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    print("Treinamento concluído.")

    # Etapa 6: Avaliação Final
    print("\n--- Etapa 6: Avaliação Final no Conjunto de Teste ---")
    model.load_weights(MODEL_SAVE_PATH)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia no teste: {test_accuracy:.4f}")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    final_metrics = calculate_all_metrics(y_test, y_pred, class_names=CLASS_NAMES)

    # Etapa 7: Visualização e Salvamento
    print("\n--- Etapa 7: Gerando Visualizações e Salvando Resultados ---")
    plot_dnn_training_history(history.history, title="Histórico de Treino - RHCB5", filename="rhcb5_training_history.png")
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, filename="rhcb5_confusion_matrix.png")
    
    results_file_path = os.path.join(RUN_RESULTS_DIR, "final_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(final_metrics, f, indent=4, cls=NumpyEncoder)
        print(f"Resultados de avaliação salvos em: {results_file_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")

    total_execution_time = time.time() - start_time_total
    print(f"\nTempo total de execução do pipeline: {total_execution_time/60:.2f} minutos")
    print("\n--- Fim da Execução ---")