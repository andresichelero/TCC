# Para carregar e pré-processar dados
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_bonn_data(base_path):
    """
    Carrega os dados dos conjuntos A, D, E do dataset BONN.
    Assume que os arquivos .txt estão dentro de subpastas 'Set A', 'Set D', 'Set E'.
    Args:
        base_path (str): Caminho para o diretório contendo as pastas Set A, Set D, Set E.
    Returns:
        tuple: (dados_np, rotulos_np)
               dados_np: Array NumPy (n_segmentos, n_amostras)
               rotulos_np: Array NumPy (n_segmentos,)
    """
    data_segments = []
    labels = []
    # Mapeamento diretório
    # Conjunto A (Normal): 0
    # Conjunto D (Interictal): 1
    # Conjunto E (Ictal): 2
    sets_labels = {'A': 0, 'D': 1, 'E': 2}
    expected_segment_length = 4097 # Conforme artigo (23.6s * 173.61 Hz)

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


def preprocess_eeg(data, fs=173.61, lowcut_hz=None, highcut_hz=40.0, order=4):
    """
    Aplica filtro Butterworth e normalização Min-Max para [-1, 1] aos sinais EEG.
    Args:
        data (np.ndarray): Array de sinais EEG (n_segmentos, n_amostras).
        fs (float): Frequência de amostragem.
        lowcut_hz (float, optional): Frequência de corte inferior para passa-banda.
                                     Se None, aplica passa-baixas.
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
        if low <= 0: # Ajuste para garantir que low seja estritamente positivo para passa-banda
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
        print("########################## SEM FILTRO - Processamento será invalido! ################################")
        # Se nenhum filtro for aplicado, apenas prossegue para a normalização.
        # No entanto, o artigo especifica um filtro.
        # Para evitar erro se b,a não definidos:
        b, a = None, None


    print("Iniciando pré-processamento (filtragem e normalização)...")
    for i in tqdm(range(data.shape[0]), desc="Pré-processando segmentos"):
        signal = data[i, :]
        if b is not None and a is not None:
            filtered_signal = filtfilt(b, a, signal)
        else:
            filtered_signal = signal # Se nenhum filtro for para ser aplicado

        # Normaliza para [-1, 1]
        min_val = np.min(filtered_signal)
        max_val = np.max(filtered_signal)
        if (max_val - min_val) > 1e-6: # Evita divisão por zero
            normalized_signal = 2 * (filtered_signal - min_val) / (max_val - min_val) - 1
        else:
            normalized_signal = np.zeros_like(filtered_signal) # Sinal constante

        processed_data[i, :] = normalized_signal
    print("Pré-processamento concluído.")
    return processed_data


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

    # Primeiro divide em treino+validação e teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Calcula o tamanho da validação relativo ao conjunto temporário
    # (1.0 - test_size) é a proporção do conjunto temporário em relação ao total
    # val_size (do total) / (1.0 - test_size) (proporção do temp) = proporção da validação no temp
    relative_val_size = val_size / (1.0 - test_size)

    # Divide o conjunto temporário em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )

    print("Dados divididos em conjuntos de treino, validação e teste:")
    print(f"  Treino:    {X_train.shape[0]} amostras, {X_train.shape[1]} features (antes da extração SWT)")
    print(f"  Validação: {X_val.shape[0]} amostras")
    print(f"  Teste:     {X_test.shape[0]} amostras")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # BASE_DATA_DIR = '../data'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    BASE_DATA_DIR = os.path.join(project_root, 'data')

    print(f"Procurando dados em: {BASE_DATA_DIR}")

    try:
        raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
        print(f"\nForma dos dados brutos: {raw_data.shape}")
        print(f"Forma dos rótulos brutos: {raw_labels.shape}")
        print(f"Classes presentes: {np.unique(raw_labels, return_counts=True)}")

        # Pré-processamento - filtro passa-baixas 0-40 Hz
        data_processed = preprocess_eeg(raw_data, fs=173.61, highcut_hz=40.0, order=4)
        print(f"\nForma dos dados processados: {data_processed.shape}")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_processed, raw_labels)
        print(f"\nPrimeiras 5 amostras de y_train: {y_train[:5]}")

    except FileNotFoundError:
        print(f"ERRO: Diretório de dados não encontrado em {BASE_DATA_DIR}. "
              "Certifique-se de que o dataset BONN está corretamente localizado e nomeado.")
    except ValueError as ve:
        print(f"ERRO de Valor: {ve}")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")