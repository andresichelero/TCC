# test_dnn_standalone.py
import os
import sys
import numpy as np
import tensorflow as tf
import time
import json
import pandas as pd
import gc 

# Configuração de Caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src')) 

from src.data_loader import load_bonn_data, preprocess_eeg, split_data
from src.feature_extractor import extract_swt_features
from src.dnn_model import build_dnn_model 
from src.utils import calculate_all_metrics, plot_dnn_training_history
import src.utils as utils_module

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier 
from tensorflow.keras.callbacks import EarlyStopping

# --- Configurações Globais ---
BASE_DATA_DIR = os.path.join(current_dir, 'data')
RESULTS_DIR_BASE = os.path.join(current_dir, 'results', 'dnn_gridsearch_custom_params_nopatience') 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

utils_module.SAVE_PLOTS = True 

# Parâmetros do Dataset e Pré-processamento
FS = 173.61
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
SWT_WAVELET = 'db4'
SWT_LEVEL = 4
TEST_SIZE = 0.15
VAL_SIZE = 0.15 

# --- HIPERPARÂMETROS PARA GridSearchCV ---
EPOCHS_LIST = [200, 250, 300]
BATCH_SIZE_LIST = [16, 32, 64]
LEARNING_RATE_LIST = [0.001, 0.0005, 0.0001]
DROPOUT_RATE_LIST = [0.1]
FIXED_PATIENCE_ES = 25

OPTIMIZER_PARAMS = [
    {'model__optimizer_name': ['adam'], 'model__learning_rate': LEARNING_RATE_LIST},
    {'model__optimizer_name': ['sgd'], 'model__learning_rate': LEARNING_RATE_LIST, 'model__momentum': [0.9]},
    {'model__optimizer_name': ['rmsprop'], 'model__learning_rate': LEARNING_RATE_LIST},
]

REGULARIZER_PARAMS = [
    {'model__kernel_regularizer_type': [None]}, 
    {'model__kernel_regularizer_type': ['l1'], 'model__kernel_regularizer_strength': [0.01, 0.001]},
    {'model__kernel_regularizer_type': ['l2'], 'model__kernel_regularizer_strength': [0.01, 0.001]},
]

# --- FONTES DE FEATURES PARA TESTAR ---
FEATURE_SOURCES_TO_TEST = ["bda", "bpso"] 
LOAD_FEATURES_FROM_JSON = True 
RESULTS_JSON_PATH_MAIN_PIPELINE = os.path.join(current_dir, 'results/all_pipeline_results.json')
FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE = [1]*45 


def train_evaluate_best_model(
    best_params_from_grid,
    fixed_binary_vector,
    X_train_all_features, y_train_data,
    X_val_all_features, y_val_data, 
    X_test_all_features, y_test_data,
    class_names, 
    fixed_es_patience,
    run_name_prefix="best_dnn_model", current_plots_dir="plots"
):
    utils_module.PLOTS_DIR = current_plots_dir 

    selected_indices = np.where(np.array(fixed_binary_vector) == 1)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        print(f"ERRO: Nenhuma feature selecionada para {run_name_prefix}. Avaliação abortada.")
        return None, 0.0

    X_train_selected = X_train_all_features[:, selected_indices]
    X_val_selected = X_val_all_features[:, selected_indices] 
    X_test_selected = X_test_all_features[:, selected_indices]

    best_lr = best_params_from_grid.get('model__learning_rate', 0.001)
    best_optimizer = best_params_from_grid.get('model__optimizer_name', 'adam')
    best_momentum = best_params_from_grid.get('model__momentum') 
    best_reg_type = best_params_from_grid.get('model__kernel_regularizer_type')
    best_reg_strength = best_params_from_grid.get('model__kernel_regularizer_strength', 0.01)
    best_dropout1 = best_params_from_grid.get('model__dropout_rate1', 0.1)
    best_dropout2 = best_params_from_grid.get('model__dropout_rate2', 0.1)
    best_dropout3 = best_params_from_grid.get('model__dropout_rate3', 0.1)

    best_epochs = best_params_from_grid.get('epochs', 200) 
    best_batch_size = best_params_from_grid.get('batch_size', 64) 
    current_patience_for_final_model = fixed_es_patience

    run_name = (f"{run_name_prefix}_opt-{best_optimizer}_lr-{best_lr:.0e}_"
                f"reg-{str(best_reg_type).lower()}_str-{best_reg_strength if best_reg_type else 0:.0e}_"
                f"drp-{best_dropout1:.1f}-{best_dropout2:.1f}-{best_dropout3:.1f}_"
                f"ep-{best_epochs}_bs-{best_batch_size}_patFIX-{current_patience_for_final_model}").replace('.', '_')

    print(f"\n--- Treinando e Avaliando Melhor Modelo Final: {run_name} ---")
    print(f"   Com Parâmetros (patience é fixa em {current_patience_for_final_model}): {best_params_from_grid}")
    tf.keras.backend.clear_session()
    gc.collect()

    final_model = build_dnn_model(
        num_selected_features=num_selected,
        num_classes=len(class_names),
        learning_rate=best_lr,
        optimizer_name=best_optimizer,
        momentum=best_momentum,
        kernel_regularizer_type=best_reg_type,
        kernel_regularizer_strength=best_reg_strength,
        dropout_rate1=best_dropout1,
        dropout_rate2=best_dropout2,
        dropout_rate3=best_dropout3,
        jit_compile_dnn=False 
    )
    final_model.summary(print_fn=lambda x: print(x, flush=True))

    early_stopping_final = EarlyStopping(
        monitor='val_loss', 
        patience=current_patience_for_final_model,
        restore_best_weights=True, 
        verbose=1
    )

    start_train_time = time.time()
    history = final_model.fit(
        X_train_selected, y_train_data, 
        epochs=best_epochs, 
        batch_size=best_batch_size,
        validation_data=(X_val_selected, y_val_data), 
        callbacks=[early_stopping_final], 
        verbose=1
    )
    training_time = time.time() - start_train_time
    print(f"Treinamento finalizado em {training_time:.2f} segundos.")

    plot_dnn_training_history(history.history, title=f"Histórico Treino Final - {run_name}", filename=f"final_dnn_history_{run_name}.png")

    y_pred_test = np.argmax(final_model.predict(X_test_selected), axis=1)
    metrics = calculate_all_metrics(y_test_data, y_pred_test, class_names=class_names)
    metrics['num_selected_features'] = num_selected
    metrics['training_time_seconds'] = training_time
    metrics['best_params_from_gridsearch'] = best_params_from_grid 
    metrics['fixed_early_stopping_patience'] = current_patience_for_final_model

    model_save_path = os.path.join(current_plots_dir, "..", f"model_{run_name}.keras") 
    try:
        final_model.save(model_save_path)
        print(f"Modelo final salvo em: {model_save_path}")
    except Exception as e:
        print(f"Erro ao salvar modelo final: {e}")

    del final_model
    gc.collect()
    return metrics, training_time


if __name__ == "__main__":
    print("Iniciando Script de Teste com GridSearchCV Extendido para DNN...")
    overall_script_start_time = time.time()

    print("\n--- 1. Carregando Dados ---", flush=True)
    raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
    class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]

    print("\n--- 2. Pré-processando Dados ---", flush=True)
    data_processed = preprocess_eeg(raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER)

    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = split_data(
        data_processed, raw_labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    print("\n--- 4. Extraindo Características (SWT) ---", flush=True)
    X_train_feat_all, feature_names_swt = extract_swt_features(X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    X_val_feat_all, _ = extract_swt_features(X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    X_test_feat_all, _ = extract_swt_features(X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)

    DIM_FEATURES = X_train_feat_all.shape[1]
    print(f"Número total de características extraídas: {DIM_FEATURES}", flush=True)
    
    if len(FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE) != DIM_FEATURES:
        print(f"Aviso: FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE não tem o comprimento correto ({len(FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE)} vs {DIM_FEATURES}). Ajustando para todas as features.")
        FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE = np.ones(DIM_FEATURES, dtype=int).tolist()

    all_gridsearch_runs_master_results = []

    for feature_source_name in FEATURE_SOURCES_TO_TEST:
        print(f"\n\n===== INICIANDO GRIDSEARCH PARA FEATURES DE: {feature_source_name.upper()} =====")

        current_results_dir = os.path.join(RESULTS_DIR_BASE, f"features_{feature_source_name}")
        current_plots_dir = os.path.join(current_results_dir, 'plots')
        os.makedirs(current_plots_dir, exist_ok=True)
        utils_module.PLOTS_DIR = current_plots_dir 

        current_fixed_feature_vector = None
        run_name_suffix_for_features = f"feats_from_{feature_source_name}"

        if feature_source_name.lower() in ["bda", "bpso"] and LOAD_FEATURES_FROM_JSON:
            try:
                with open(RESULTS_JSON_PATH_MAIN_PIPELINE, 'r') as f: data = json.load(f)
                opt_key = f"{feature_source_name.lower()}_optimization"
                if opt_key in data and "selected_features_vector" in data[opt_key]:
                    current_fixed_feature_vector = data[opt_key]["selected_features_vector"]
                    if len(current_fixed_feature_vector) != DIM_FEATURES:
                        print(f"Erro: Vetor de features de '{opt_key}' tem comprimento {len(current_fixed_feature_vector)}, esperado {DIM_FEATURES}. Usando manual.", flush=True)
                        current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE
                        run_name_suffix_for_features = "feats_manual_fallback_len_mismatch"
                    else:
                         print(f"Carregadas {int(np.sum(current_fixed_feature_vector))} features de '{opt_key}'.", flush=True)
                else:
                    print(f"Aviso: Chave '{opt_key}' não encontrada no JSON. Usando manual.", flush=True)
                    current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE
                    run_name_suffix_for_features = "feats_manual_fallback_no_key"
            except Exception as e:
                print(f"ERRO ao carregar features do JSON para {feature_source_name}: {e}. Usando manual.", flush=True)
                current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE
                run_name_suffix_for_features = "feats_manual_fallback_exception"
        elif feature_source_name.lower() == "all_features":
            current_fixed_feature_vector = np.ones(DIM_FEATURES, dtype=int).tolist()
            run_name_suffix_for_features = "feats_all"
            print(f"Usando todas as {DIM_FEATURES} features.", flush=True)
        elif feature_source_name.lower() == "manual":
            current_fixed_feature_vector = FIXED_FEATURE_VECTOR_MANUAL_EXAMPLE
            run_name_suffix_for_features = "feats_manual_defined"
            print(f"Usando vetor de features manual definido: {int(np.sum(current_fixed_feature_vector))} features.", flush=True)
        else:
            print(f"Fonte de feature '{feature_source_name}' desconhecida. Pulando.", flush=True)
            continue

        if current_fixed_feature_vector is None: 
            print(f"ERRO: Vetor de características não pôde ser determinado para {feature_source_name}. Pulando.", flush=True)
            continue
            
        num_selected_initial = int(np.sum(current_fixed_feature_vector))
        if num_selected_initial == 0:
            print(f"ERRO CRÍTICO: Nenhuma feature selecionada para {feature_source_name} (num_selected_initial = 0). Pulando esta fonte.", flush=True)
            continue
        print(f"Para {feature_source_name.upper()}: Usando {num_selected_initial} features para o GridSearchCV.", flush=True)

        selected_indices = np.where(np.array(current_fixed_feature_vector) == 1)[0]
        X_train_selected_for_gs = X_train_feat_all[:, selected_indices]
        X_val_selected_for_gs = X_val_feat_all[:, selected_indices] 

        param_grid_list = []
        
        fit_params_for_grid = {
            'batch_size': BATCH_SIZE_LIST,
            'epochs': EPOCHS_LIST,
        }
        
        common_model_params_for_grid = {
            'model__dropout_rate1': DROPOUT_RATE_LIST,
            'model__dropout_rate2': DROPOUT_RATE_LIST,
            'model__dropout_rate3': DROPOUT_RATE_LIST,
        }

        for opt_config_template in OPTIMIZER_PARAMS: 
            for reg_config_template in REGULARIZER_PARAMS: 
                current_grid_config = {
                    **fit_params_for_grid, 
                    **common_model_params_for_grid,
                    **opt_config_template, 
                    **reg_config_template
                }
                param_grid_list.append(current_grid_config)
        
        print(f"GridSearch param_grid construído com {len(param_grid_list)} dicionários de configuração.", flush=True)
        if len(param_grid_list) > 0:
             print(f"Exemplo de primeira configuração no grid (patience do ES é fixa em {FIXED_PATIENCE_ES}): {param_grid_list[0]}", flush=True)

        early_stopping_gs_callback = EarlyStopping(
            monitor='val_loss', 
            patience=FIXED_PATIENCE_ES,
            restore_best_weights=True, 
            verbose=1 
        )
        
        keras_clf = KerasClassifier(
            model=build_dnn_model, 
            model__num_selected_features=num_selected_initial, 
            model__num_classes=len(class_names),          
            model__jit_compile_dnn=False,                 
            verbose=0, 
            callbacks=[early_stopping_gs_callback]
        )

        grid_search = GridSearchCV(
            estimator=keras_clf, 
            param_grid=param_grid_list, 
            cv=2, 
            scoring='accuracy', 
            verbose=1,        
            refit=True,       
            n_jobs=1,
            error_score='raise'      
        )

        print(f"\n--- 5. Iniciando GridSearchCV para {feature_source_name.upper()} ---", flush=True)
        gs_start_time = time.time()
        grid_search.fit(X_train_selected_for_gs, y_train, validation_data=(X_val_selected_for_gs, y_val))
        
        gs_total_time_min = (time.time() - gs_start_time) / 60
        print(f"GridSearchCV para {feature_source_name.upper()} finalizado em {gs_total_time_min:.2f} minutos.", flush=True)
        print(f"Melhores Parâmetros (CV): {grid_search.best_params_}")
        print(f"Melhor Acurácia (CV): {grid_search.best_score_:.4f}")

        best_model_metrics_final, final_model_train_time_sec = train_evaluate_best_model(
            best_params_from_grid=grid_search.best_params_,
            fixed_binary_vector=current_fixed_feature_vector,
            X_train_all_features=X_train_feat_all, y_train_data=y_train,
            X_val_all_features=X_val_feat_all, y_val_data=y_val, 
            X_test_all_features=X_test_feat_all, y_test_data=y_test,
            class_names=class_names,
            fixed_es_patience=FIXED_PATIENCE_ES,
            run_name_prefix=f"gs_best_{run_name_suffix_for_features}",
            current_plots_dir=current_plots_dir
        )

        if best_model_metrics_final:
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            relevant_cols = [
                col for col in cv_results_df.columns if 
                'param_model__' in col or 
                col in ['param_epochs', 'param_batch_size', 'mean_test_score', 'std_test_score', 'rank_test_score']
            ]
            cv_results_summary = cv_results_df[relevant_cols].sort_values(by='rank_test_score').to_dict('records')

            run_result_entry = {
                "feature_source": feature_source_name,
                "num_input_features_for_gs": num_selected_initial,
                "best_cv_score_grid": grid_search.best_score_,
                "best_params_grid": grid_search.best_params_,
                "fixed_early_stopping_patience": FIXED_PATIENCE_ES,
                "grid_search_duration_minutes": gs_total_time_min,
                "final_model_train_duration_seconds": final_model_train_time_sec,
                "final_model_test_metrics": best_model_metrics_final,
                "grid_cv_results_summary_top10": cv_results_summary[:10] 
            }
            all_gridsearch_runs_master_results.append(run_result_entry)
        
        del grid_search, keras_clf, cv_results_df
        tf.keras.backend.clear_session()
        gc.collect()

    master_summary_file_path = os.path.join(RESULTS_DIR_BASE, "gridsearch_ALL_SOURCES_summary.json")
    try:
        class NpEncoder(json.JSONEncoder): 
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(master_summary_file_path, 'w') as f:
            json.dump(all_gridsearch_runs_master_results, f, indent=4, cls=NpEncoder)
        print(f"\nResumo mestre de todos os GridSearchCV salvo em: {master_summary_file_path}", flush=True)
    except Exception as e: 
        print(f"Erro ao salvar resumo mestre JSON: {e}", flush=True)

    try:
        master_df_list = []
        for record in all_gridsearch_runs_master_results:
            flat_record = record.copy()
            if 'best_params_grid' in flat_record and isinstance(flat_record['best_params_grid'], dict):
                for k,v in flat_record['best_params_grid'].items(): 
                    param_key = k.replace('model__', '')
                    flat_record[f"best_param_{param_key}"] = v
                del flat_record['best_params_grid']
            if 'final_model_test_metrics' in flat_record and isinstance(flat_record['final_model_test_metrics'], dict):
                for k,v in flat_record['final_model_test_metrics'].items():
                    if not isinstance(v, (dict, list)): flat_record[f"final_metric_{k}"] = v
                del flat_record['final_model_test_metrics']
            if 'grid_cv_results_summary_top10' in flat_record: 
                del flat_record['grid_cv_results_summary_top10'] 
            master_df_list.append(flat_record)

        master_df = pd.DataFrame(master_df_list)
        master_csv_path = os.path.join(RESULTS_DIR_BASE, "gridsearch_ALL_SOURCES_summary.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"Resumo mestre em CSV salvo em: {master_csv_path}", flush=True)
    except Exception as e:
        print(f"Erro ao salvar resumo mestre CSV: {e}", flush=True)

    overall_script_duration_min = (time.time() - overall_script_start_time) / 60
    print(f"\n--- Tempo Total de Execução do Script GridSearchCV Extendido: {overall_script_duration_min:.2f} minutos ---")
    print("--- Script Finalizado ---")
