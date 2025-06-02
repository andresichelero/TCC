# test_dnn_standalone.py
import os
import sys
import numpy as np
import tensorflow as tf
import time
import json
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.data_loader import load_bonn_data, preprocess_eeg, split_data
from src.feature_extractor import extract_swt_features
from src.dnn_model import build_dnn_model
from src.utils import calculate_all_metrics, plot_dnn_training_history
import src.utils as utils_module

# --- Configurações ---
BASE_DATA_DIR = os.path.join(current_dir, 'data')
RESULTS_DIR = os.path.join(current_dir, 'results', 'dnn_standalone_tests_series') # Novo diretório para séries
PLOTS_DIR_STANDALONE = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR_STANDALONE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

utils_module.SAVE_PLOTS = True
utils_module.PLOTS_DIR = PLOTS_DIR_STANDALONE

FS = 173.61
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
SWT_WAVELET = 'db4'
SWT_LEVEL = 4

TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- PARÂMETROS E CONFIGURAÇÕES DA DNN PARA TESTAR ---
# Listas de parâmetros de treinamento
EPOCHS_LIST = [200, 250]  # Ex: [150, 250, 300]
PATIENCE_LIST = [25, 30] # Ex: [15, 20, 30]
BATCH_SIZE_LIST = [32, 64] # Ex: [32, 64, 128]
#LEARNING_RATE_LIST

# Identificadores para diferentes configurações da DNN em dnn_model.py
DNN_CONFIG_IDS_TO_TEST = ["original", "config_A", "config_B"]


# ---CONJUNTO FIXO DE CARACTERÍSTICAS ---
USE_ALL_FEATURES = False
FIXED_FEATURE_VECTOR_MANUAL = [
    0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 1
]
LOAD_FEATURES_FROM_JSON = True
RESULTS_JSON_PATH = os.path.join(current_dir, 'results/all_pipeline_results.json')
OPTIMIZER_FOR_FEATURES = "bpso" # "bda" ou "bpso" (usará as features encontradas por eles na ultima run)

def test_dnn_with_fixed_features(
    fixed_binary_vector,
    X_train_all_features, y_train_data,
    X_val_all_features, y_val_data,
    X_test_all_features, y_test_data,
    dnn_training_params_override,
    dnn_config_id_override,
    class_names, test_run_name="dnn_test"
):
    print(f"\n--- Testando DNN com Características Fixas: {test_run_name} ---")
    print(f"--- Parâmetros de Treino: {dnn_training_params_override} ---")
    print(f"--- Configuração DNN ID: {dnn_config_id_override} ---")

    selected_indices = np.where(np.array(fixed_binary_vector) == 1)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        print("ERRO: Nenhuma característica selecionada. Abortando.")
        return None

    X_train_selected = X_train_all_features[:, selected_indices]
    X_val_selected = X_val_all_features[:, selected_indices]
    X_test_selected = X_test_all_features[:, selected_indices]

    tf.keras.backend.clear_session()

    model = build_dnn_model(
        num_selected_features=num_selected,
        num_classes=len(class_names),
        dnn_config_id=dnn_config_id_override
    )
    print("\nResumo do Modelo DNN:")
    model.summary(print_fn=lambda x: print(x, flush=True))


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=dnn_training_params_override.get('patience', 20),
        restore_best_weights=True,
        verbose=1
    )

    print("\nIniciando treinamento da DNN...")
    start_train_time = time.time()
    history_obj = model.fit(
        X_train_selected, y_train_data,
        epochs=dnn_training_params_override.get('epochs', 100),
        batch_size=dnn_training_params_override.get('batch_size', 64),
        validation_data=(X_val_selected, y_val_data),
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_train_time
    print(f"Treinamento da DNN finalizado em {training_time:.2f} segundos.")

    plot_dnn_training_history(
        history_obj.history,
        title=f"Histórico Treino DNN - {test_run_name}",
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
            print(f"{key}:", flush=True)
            for cat, report_vals in value.items():
                if isinstance(report_vals, dict):
                    print(f"  {cat}:", flush=True)
                    for metric_name, metric_val in report_vals.items():
                        print(f"    {metric_name}: {metric_val:.4f}", flush=True)
                else:
                     print(f"  {cat}: {report_vals:.4f}", flush=True)
        elif key == "confusion_matrix":
            print(f"{key}:\n{np.array(value)}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)
    return metrics

if __name__ == "__main__":
    print("Iniciando Script de Teste Autônomo da DNN...")
    overall_start_time = time.time()

    # 1. Carregar Dados
    print("\n--- 1. Carregando Dados ---", flush=True)
    raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
    class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]

    # 2. Pré-processar Dados
    print("\n--- 2. Pré-processando Dados ---", flush=True)
    data_processed = preprocess_eeg(raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER) #

    # 3. Dividir Dados
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = split_data(
        data_processed, raw_labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # 4. Extrair Características
    print("\n--- 3. Extraindo Características (SWT) ---", flush=True)
    X_train_feat_all, feature_names = extract_swt_features(X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    X_val_feat_all, _ = extract_swt_features(X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    X_test_feat_all, _ = extract_swt_features(X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)

    DIM_FEATURES = X_train_feat_all.shape[1]
    print(f"Número total de características extraídas: {DIM_FEATURES}", flush=True)

    current_fixed_feature_vector = None
    base_run_name_suffix = "default_feats"

    if USE_ALL_FEATURES:
        current_fixed_feature_vector = np.ones(DIM_FEATURES, dtype=int).tolist()
        base_run_name_suffix = "all_features"
    elif LOAD_FEATURES_FROM_JSON:
        try:
            with open(RESULTS_JSON_PATH, 'r') as f:
                results_data = json.load(f) #
            opt_key = f"{OPTIMIZER_FOR_FEATURES}_optimization"
            if opt_key in results_data:
                current_fixed_feature_vector = results_data[opt_key]["selected_features_vector"]
                base_run_name_suffix = f"feats_from_{OPTIMIZER_FOR_FEATURES}"
            else:
                print(f"AVISO: Chave '{opt_key}' não encontrada. Usando manual.", flush=True)
                current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
                base_run_name_suffix = "manual_feats_fallback_json"
        except Exception as e:
            print(f"ERRO ao carregar do JSON: {e}. Usando manual.", flush=True)
            current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
            base_run_name_suffix = "manual_feats_error_fallback"
    else:
        current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL
        base_run_name_suffix = "manual_features"

    if current_fixed_feature_vector is None or len(current_fixed_feature_vector) != DIM_FEATURES:
        feat_len = len(current_fixed_feature_vector) if current_fixed_feature_vector is not None else "None"
        print(f"ERRO CRÍTICO: Vetor de características inválido (comprimento {feat_len}, esperado {DIM_FEATURES}).", flush=True)
        sys.exit(1)
    print(f"Usando conjunto de características: {base_run_name_suffix} ({np.sum(current_fixed_feature_vector)} selecionadas)", flush=True)

    all_series_results = []
    total_combinations = len(EPOCHS_LIST) * len(PATIENCE_LIST) * len(BATCH_SIZE_LIST) * len(DNN_CONFIG_IDS_TO_TEST)
    current_run_count = 0

    for epochs_val in EPOCHS_LIST:
        for patience_val in PATIENCE_LIST:
            for batch_size_val in BATCH_SIZE_LIST:
                for dnn_id_val in DNN_CONFIG_IDS_TO_TEST:
                    current_run_count += 1
                    # Parâmetros de treino para esta iteração
                    current_dnn_training_params = {
                        'epochs': epochs_val,
                        'batch_size': batch_size_val,
                        'patience': patience_val
                        # outros parâmetros (learning_rate)
                    }

                    # Nome descritivo para esta execução específica
                    test_run_name_current = f"dnn_{dnn_id_val}_ep{epochs_val}_pat{patience_val}_bs{batch_size_val}_{base_run_name_suffix}"

                    print(f"\n\n===== INICIANDO TESTE ({current_run_count}/{total_combinations}): {test_run_name_current} =====", flush=True)

                    run_metrics = test_dnn_with_fixed_features(
                        fixed_binary_vector=current_fixed_feature_vector,
                        X_train_all_features=X_train_feat_all, y_train_data=y_train,
                        X_val_all_features=X_val_feat_all, y_val_data=y_val,
                        X_test_all_features=X_test_feat_all, y_test_data=y_test,
                        dnn_training_params_override=current_dnn_training_params,
                        dnn_config_id_override=dnn_id_val,
                        class_names=class_names,
                        test_run_name=test_run_name_current
                    )

                    if run_metrics:
                        result_entry = {
                            "test_name": test_run_name_current,
                            "dnn_config_id": dnn_id_val,
                            "epochs": epochs_val,
                            "patience": patience_val,
                            "batch_size": batch_size_val,
                            "feature_set_type": base_run_name_suffix,
                            "num_selected_features_in_set": int(np.sum(current_fixed_feature_vector)),
                            "accuracy_test": run_metrics.get('accuracy'),
                            "f1_macro_test": run_metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score'),
                            "training_time_seconds": run_metrics.get('training_time_seconds'),
                            "full_metrics_test": run_metrics
                        }
                        all_series_results.append(result_entry)
                    else:
                         print(f"AVISO: Nenhuma métrica retornada para o teste: {test_run_name_current}", flush=True)
                    
                    print(f"===== TESTE ({current_run_count}/{total_combinations}) FINALIZADO: {test_run_name_current} =====", flush=True)
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()


    summary_file_path = os.path.join(RESULTS_DIR, "dnn_series_test_summary.json")
    with open(summary_file_path, 'w') as f_summary:
        json.dump(all_series_results, f_summary, indent=4)
    print(f"\n\nResumo de todos os testes da série salvo em: {summary_file_path}", flush=True)

    if all_series_results:
        try:
            df_results = pd.DataFrame(all_series_results)
            if 'full_metrics_test' in df_results.columns:
                df_results_simplified = df_results.drop(columns=['full_metrics_test'])
            else:
                df_results_simplified = df_results
            csv_summary_path = os.path.join(RESULTS_DIR, "dnn_series_test_summary.csv")
            df_results_simplified.to_csv(csv_summary_path, index=False)
            print(f"Resumo da série em CSV salvo em: {csv_summary_path}", flush=True)
        except Exception as e:
            print(f"Erro ao salvar resumo em CSV: {e}", flush=True)
    else:
        print("Nenhum resultado coletado na série para salvar em CSV.", flush=True)


    overall_execution_time = time.time() - overall_start_time
    print(f"\n--- Tempo Total de Execução da Série de Testes da DNN: {overall_execution_time/60:.2f} minutos ({overall_execution_time:.2f} segundos) ---", flush=True)
    print("--- Script de Teste Autônomo da DNN em Série Finalizado ---", flush=True)