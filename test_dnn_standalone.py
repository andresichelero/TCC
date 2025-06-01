# test_dnn_standalone.py
import os
import sys
import numpy as np
import tensorflow as tf
import time
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))


from src.data_loader import load_bonn_data, preprocess_eeg, split_data
from src.feature_extractor import extract_swt_features
from src.dnn_model import build_dnn_model
from src.utils import calculate_all_metrics, plot_dnn_training_history
import src.utils as utils_module

# --- Configurações ---
BASE_DATA_DIR = os.path.join(current_dir, 'data')
RESULTS_DIR = os.path.join(current_dir, 'results', 'dnn_standalone_tests')
PLOTS_DIR_STANDALONE = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR_STANDALONE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configurações de Plot
utils_module.SAVE_PLOTS = True
utils_module.PLOTS_DIR = PLOTS_DIR_STANDALONE

# Dataset e Pré-processamento
FS = 173.61
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
SWT_WAVELET = 'db4'
SWT_LEVEL = 4

# Divisão dos Dados
TEST_SIZE = 0.15
VAL_SIZE = 0.15

DNN_TRAINING_PARAMS_STANDALONE = {'epochs': 250, 'batch_size': 32, 'patience': 20}

# --- ESCOLHER SEU CONJUNTO FIXO DE CARACTERÍSTICAS ---
# Opção 1: Usar todas as características
USE_ALL_FEATURES = False # Mude para True para usar todas as características

# Opção 2: Definir um vetor binário específico de características
# Substitua pelo vetor desejado. Garanta que o comprimento corresponda a DIM_FEATURES.
# Exemplo para 45 características (igual à saída BDA do JSON):
FIXED_FEATURE_VECTOR_MANUAL = [
    0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 1
]

# Opção 3: Carregar do JSON de resultados
# Mude para True para carregar do JSON, forneça o caminho e o nome do otimizador
LOAD_FEATURES_FROM_JSON = True
RESULTS_JSON_PATH = os.path.join(current_dir, 'results/all_pipeline_results.json')
OPTIMIZER_FOR_FEATURES = "bpso" # "bda" ou "bpso" (usará as features encontradas por eles na ultima run)

def test_dnn_with_fixed_features(
    fixed_binary_vector,
    X_train_all_features, y_train_data,
    X_val_all_features, y_val_data,
    X_test_all_features, y_test_data,
    dnn_params, class_names, test_run_name="dnn_test"
):
    print(f"\n--- Testando DNN com Características Fixas: {test_run_name} ---")

    selected_indices = np.where(np.array(fixed_binary_vector) == 1)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        print("ERRO: Nenhuma característica selecionada pelo vetor fixo. Abortando.")
        return None

    print(f"Usando {num_selected} características pré-selecionadas.")

    X_train_selected = X_train_all_features[:, selected_indices]
    X_val_selected = X_val_all_features[:, selected_indices]
    X_test_selected = X_test_all_features[:, selected_indices]

    tf.keras.backend.clear_session()

    model = build_dnn_model(num_selected_features=num_selected, num_classes=len(class_names))
    print("\nResumo do Modelo DNN:")
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=dnn_params.get('patience', 20),
        restore_best_weights=True,
        verbose=1
    )

    print("\nIniciando treinamento da DNN...")
    start_train_time = time.time()
    history_obj = model.fit(
        X_train_selected, y_train_data,
        epochs=dnn_params.get('epochs', 100),
        batch_size=dnn_params.get('batch_size', 64),
        validation_data=(X_val_selected, y_val_data),
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_train_time
    print(f"Treinamento da DNN finalizado em {training_time:.2f} segundos.")

    plot_dnn_training_history(
        history_obj.history,
        title=f"Histórico de Treinamento DNN - {test_run_name}",
        filename=f"dnn_history_{test_run_name}.png"
    )

    print("\nAvaliando DNN no conjunto de teste...")
    y_pred_test = np.argmax(model.predict(X_test_selected), axis=1)

    metrics = calculate_all_metrics(y_test_data, y_pred_test, class_names=class_names)
    metrics['num_selected_features'] = num_selected
    metrics['training_time_seconds'] = training_time

    model_save_path = os.path.join(RESULTS_DIR, f"model_{test_run_name}.keras")
    try:
        model.save(model_save_path)
        print(f"Modelo DNN salvo em: {model_save_path}")
    except Exception as e:
        print(f"Erro ao salvar modelo DNN: {e}")

    print(f"\n--- Resultados para {test_run_name} ---")
    for key, value in metrics.items():
        if key == "classification_report":
            print(f"{key}:")
            for cat, report_vals in value.items():
                if isinstance(report_vals, dict):
                    print(f"  {cat}:")
                    for metric_name, metric_val in report_vals.items():
                        print(f"    {metric_name}: {metric_val:.4f}")
                else:
                     print(f"  {cat}: {report_vals:.4f}")
        elif key == "confusion_matrix":
            print(f"{key}:\n{np.array(value)}")
        else:
            print(f"{key}: {value}")
    return metrics

if __name__ == "__main__":
    print("Iniciando Script de Teste Autônomo da DNN...")

    # 1. Carregar Dados
    print("\n--- 1. Carregando Dados ---")
    raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
    class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]

    # 2. Pré-processar Dados
    print("\n--- 2. Pré-processando Dados ---")
    data_processed = preprocess_eeg(raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER)

    # 3. Dividir Dados (Treino, Validação, Teste para extração de características)
    # Estes conjuntos são necessários ANTES de aplicar o vetor fixo de características
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = split_data(
        data_processed, raw_labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # 4. Extrair Características
    print("\n--- 3. Extraindo Características (SWT) ---")
    print("Extraindo características para o conjunto de TREINO...")
    X_train_feat_all, feature_names = extract_swt_features(X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    print("Extraindo características para o conjunto de VALIDAÇÃO...")
    X_val_feat_all, _ = extract_swt_features(X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    print("Extraindo características para o conjunto de TESTE...")
    X_test_feat_all, _ = extract_swt_features(X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)

    DIM_FEATURES = X_train_feat_all.shape[1]
    print(f"Número total de características extraídas: {DIM_FEATURES}")

    # Determina o vetor de características a ser usado
    current_fixed_feature_vector = None
    run_name_suffix = "default"

    if USE_ALL_FEATURES:
        current_fixed_feature_vector = np.ones(DIM_FEATURES, dtype=int).tolist()
        print(f"Usando TODAS as {DIM_FEATURES} características para o teste.")
        run_name_suffix = "all_features"
    elif LOAD_FEATURES_FROM_JSON:
        try:
            with open(RESULTS_JSON_PATH, 'r') as f:
                results_data = json.load(f)
            opt_key = f"{OPTIMIZER_FOR_FEATURES}_optimization" # "bda_optimization" ou "bpso_optimization"
            if opt_key in results_data:
                current_fixed_feature_vector = results_data[opt_key]["selected_features_vector"]
                num_sel = np.sum(current_fixed_feature_vector) if current_fixed_feature_vector else 0
                run_name_suffix = f"fixed_{OPTIMIZER_FOR_FEATURES}_features"
                print(f"Carregadas características do {OPTIMIZER_FOR_FEATURES.upper()} ({num_sel} selecionadas) de {RESULTS_JSON_PATH}")
            else:
                print(f"AVISO: Chave '{opt_key}' não encontrada no JSON. Usando vetor manual pré-definido.")
                current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
                run_name_suffix = "fixed_manual_features_json_fallback"
        except Exception as e:
            print(f"ERRO ao carregar características do JSON: {e}. Usando vetor manual pré-definido.")
            current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
            run_name_suffix = "fixed_manual_features_error_fallback"
    else:
        current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
        num_sel = np.sum(current_fixed_feature_vector) if current_fixed_feature_vector else 0
        print(f"Usando vetor de características MANUALMENTE definido ({num_sel} selecionadas).")
        run_name_suffix = "fixed_manual_features"


    if current_fixed_feature_vector is None or len(current_fixed_feature_vector) != DIM_FEATURES:
        feat_len = len(current_fixed_feature_vector) if current_fixed_feature_vector is not None else "None"
        print(f"ERRO CRÍTICO: O vetor de características fixas é None ou seu comprimento ({feat_len}) "
              f"não corresponde à dimensão das características extraídas ({DIM_FEATURES}).")
        print("Verifique as configurações: USE_ALL_FEATURES, FIXED_FEATURE_VECTOR_MANUAL, ou o carregamento do JSON.")
        sys.exit(1)

    # Dê um nome único para a execução de teste, ex: baseado na modificação da DNN
    # Exemplo, se mudar o otimizador em dnn_model.py para SGD:
    # test_description = "dnn_com_otimizador_sgd"
    test_description = f"bpso_config_dnn_300_epochs_batch_size_32_{run_name_suffix}" # Nome padrão

    # test_description = "dnn_adam_lr0005_sigmoid_dropout02_ft_bda"

    # 5. Executa o teste da DNN
    test_dnn_with_fixed_features(
        fixed_binary_vector=current_fixed_feature_vector,
        X_train_all_features=X_train_feat_all, y_train_data=y_train,
        X_val_all_features=X_val_feat_all, y_val_data=y_val,
        X_test_all_features=X_test_feat_all, y_test_data=y_test,
        dnn_params=DNN_TRAINING_PARAMS_STANDALONE,
        class_names=class_names,
        test_run_name=test_description
    )

    print("\n--- Script de Teste Autônomo da DNN Finalizado ---")