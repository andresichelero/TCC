# main.py
import os
import numpy as np
import time
import tensorflow as tf
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.data_loader import load_bonn_data, preprocess_eeg, split_data
from src.feature_extractor import extract_swt_features
from src.dnn_model import build_dnn_model
from src.fitness_function import evaluate_fitness
from src.bda import BinaryDragonflyAlgorithm
from src.bpso import BinaryPSO
from src.utils import calculate_all_metrics, plot_convergence_curves

# --- Configurações Globais ---
BASE_DATA_DIR = os.path.join(current_dir, 'data')
RESULTS_DIR = os.path.join(current_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Parâmetros do Dataset e Pré-processamento
FS = 173.61
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
SWT_WAVELET = 'db4'
SWT_LEVEL = 4

# Parâmetros da Divisão de Dados
TEST_SIZE = 0.15
VAL_SIZE = 0.15 # Usado dentro da função de fitness e para otimização

# Parâmetros da DNN para Fitness e Treino Final
DNN_TRAINING_PARAMS_FITNESS = {'epochs': 200, 'batch_size': 128, 'patience': 15} # Para fitness (mais rápido)
DNN_TRAINING_PARAMS_FINAL = {'epochs': 400, 'batch_size': 256, 'patience': 20} # Para treino final (mais robusto)
# Se patience do final for usado com EarlyStopping, separar uma pequena porção do X_train_full para validação interna do treino final.
# Ou treinar por um número fixo de épocas no conjunto treino+validação. O artigo menciona ambas.
# Foi usado EarlyStopping com val_split para o treino final.

# Parâmetros dos Otimizadores
N_AGENTS_OPTIMIZERS = 10 # População/Partículas
T_MAX_ITER_OPTIMIZERS = 50 # Iterações (Artigo sugere 100)

# Parâmetros Fitness
ALPHA_FITNESS = 0.99
BETA_FITNESS = 0.01

# --- Funções Auxiliares para o Main ---
def train_and_evaluate_final_model(model_name, selected_features_vector,
                                   X_train_full_all_feat, y_train_full,
                                   X_test_all_feat, y_test,
                                   dnn_params, class_names):
    print(f"\n--- Treinamento e Avaliação Final: {model_name} ---")
    
    selected_indices = np.where(selected_features_vector == 1)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        print(f"ERRO: {model_name} não selecionou nenhuma feature. Avaliação abortada.")
        return None, None

    print(f"{model_name}: Selecionou {num_selected} características.")

    X_train_full_selected = X_train_full_all_feat[:, selected_indices]
    X_test_selected = X_test_all_feat[:, selected_indices]
    
    # Limpa sessão Keras antes de construir novo modelo
    tf.keras.backend.clear_session()
    final_model = build_dnn_model(num_selected_features=num_selected, num_classes=len(class_names),
                              jit_compile_dnn=True)
    print(f"Modelo final {model_name} construído com {num_selected} features.")
    final_model.summary()

    # Treinamento Final
    # O artigo sugere treinar no conjunto combinado (treino+validação)
    # Pode-se usar EarlyStopping com uma fração do X_train_full_selected para validação interna
    print(f"Iniciando treinamento final do modelo {model_name}...")
    early_stopping_final = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitora a perda na fração de validação do treino final
        patience=dnn_params.get('patience', 10),
        restore_best_weights=True,
        verbose=1
    )
    
    history = final_model.fit(
        X_train_full_selected, y_train_full,
        epochs=dnn_params.get('epochs', 100),
        batch_size=dnn_params.get('batch_size', 32),
        validation_split=0.1, # Usa 10% do X_train_full_selected para validação interna do treino final
        callbacks=[early_stopping_final],
        verbose=1
    )

    # Avaliação no Conjunto de Teste
    print(f"\nAvaliando {model_name} no conjunto de teste...")
    y_pred_test = np.argmax(final_model.predict(X_test_selected), axis=1)
    
    metrics = calculate_all_metrics(y_test, y_pred_test, class_names=class_names)
    metrics['num_selected_features'] = num_selected
    metrics['selected_feature_indices'] = selected_indices.tolist() # Para referência

    # Salvar o modelo treinado para review futuro se quiser
    model_save_path = os.path.join(RESULTS_DIR, f"{model_name.replace('+', '_')}_final_model.keras")
    try:
        final_model.save(model_save_path)
        print(f"Modelo final {model_name} salvo em: {model_save_path}")
    except Exception as e:
        print(f"Erro ao salvar o modelo {model_name}: {e}")


    return metrics, history.history # Retorna também o histórico para possível plotagem

# --- Script Principal ---
if __name__ == "__main__":
    start_time_total = time.time()

    print("Iniciando Pipeline de Detecção de Epilepsia...")
    print(f"Usando TensorFlow versão: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
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
        raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
    except Exception as e:
        print(f"Falha ao carregar dados: {e}. Verifique o caminho e formato do dataset.")
        sys.exit(1)
    
    class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]

    # 2. Pré-processar Dados
    print("\n--- 2. Pré-processando Dados ---")
    data_processed = preprocess_eeg(raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER)

    # 3. Dividir Dados
    print("\n--- 3. Dividindo Dados ---")
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = split_data(
        data_processed, raw_labels, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # 4. Extrair Características SWT
    print("\n--- 4. Extraindo Características SWT ---")
    print("Extraindo features para o conjunto de TREINO...")
    X_train_feat, feature_names = extract_swt_features(X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    print("Extraindo features para o conjunto de VALIDAÇÃO...")
    X_val_feat, _ = extract_swt_features(X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    print("Extraindo features para o conjunto de TESTE...")
    X_test_feat, _ = extract_swt_features(X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL)
    
    DIM_FEATURES = X_train_feat.shape[1]
    print(f"Total de {DIM_FEATURES} características extraídas.")
    
    all_results = {}
    all_convergence_curves = []
    convergence_labels = []

    # --- 5. Otimização com BDA ---
    print("\n\n--- 5. Otimização com Binary Dragonfly Algorithm (BDA) ---")
    start_time_bda_opt = time.time()
    bda = BinaryDragonflyAlgorithm(
        N=N_AGENTS_OPTIMIZERS, T=T_MAX_ITER_OPTIMIZERS, dim=DIM_FEATURES,
        fitness_func=evaluate_fitness,
        X_train_feat=X_train_feat, y_train=y_train,
        X_val_feat=X_val_feat, y_val=y_val,
        dnn_params=DNN_TRAINING_PARAMS_FITNESS,
        # Valores de exemplo para BDA, ajustar conforme necessário
        s=0.1, a=0.05, c_cohesion=0.6, f_food=1.0, e_enemy=0.8, w_inertia=0.9,
        tau_min=1.0, tau_max=6.0, # Parâmetros para V-shaped
        alpha_fitness=ALPHA_FITNESS, beta_fitness=BETA_FITNESS,
        verbose_fitness=0, # 0 para silenciar Keras na fitness, 1 para debug
        seed=RANDOM_SEED
    )
    Sf_bda, best_fitness_bda, convergence_bda = bda.run()
    all_results['bda_optimization'] = {
        'best_fitness': best_fitness_bda,
        'selected_features_vector': Sf_bda.tolist(),
        'num_selected_features': int(np.sum(Sf_bda)),
        'convergence_curve': convergence_bda.tolist()
    }
    all_convergence_curves.append(convergence_bda)
    convergence_labels.append("BDA")
    print(f"Tempo de otimização BDA: {(time.time() - start_time_bda_opt)/60:.2f} minutos")

    # --- 6. Otimização com BPSO ---
    print("\n\n--- 6. Otimização com Binary Particle Swarm Optimization (BPSO) ---")
    start_time_bpso_opt = time.time()
    bpso = BinaryPSO(
        N=N_AGENTS_OPTIMIZERS, T=T_MAX_ITER_OPTIMIZERS, dim=DIM_FEATURES,
        fitness_func=evaluate_fitness,
        X_train_feat=X_train_feat, y_train=y_train,
        X_val_feat=X_val_feat, y_val=y_val,
        dnn_params=DNN_TRAINING_PARAMS_FITNESS,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, Vmax=4.0,
        alpha_fitness=ALPHA_FITNESS, beta_fitness=BETA_FITNESS,
        verbose_fitness=0, # 0 para silenciar Keras na fitness, 1 para debug
        seed=RANDOM_SEED
    )
    Sf_bpso, best_fitness_bpso, convergence_bpso = bpso.run()
    all_results['bpso_optimization'] = {
        'best_fitness': best_fitness_bpso,
        'selected_features_vector': Sf_bpso.tolist(),
        'num_selected_features': int(np.sum(Sf_bpso)),
        'convergence_curve': convergence_bpso.tolist()
    }
    all_convergence_curves.append(convergence_bpso)
    convergence_labels.append("BPSO")
    print(f"Tempo de otimização BPSO: {(time.time() - start_time_bpso_opt)/60:.2f} minutos")

    # Plotar curvas de convergência dos otimizadores
    if all_convergence_curves:
        plot_convergence_curves(all_convergence_curves, convergence_labels, "Convergência dos Otimizadores (Fitness na Validação)")


    # --- 7. Treinamento e Avaliação Final ---
    print("\n\n--- 7. Treinamento e Avaliação Final dos Modelos ---")
    # Combinar dados de treino e validação para o treino final
    X_train_full_feat = np.concatenate((X_train_feat, X_val_feat), axis=0)
    y_train_full = np.concatenate((y_train, y_val), axis=0)

    # BDA+DNN
    metrics_bda_dnn, history_bda_dnn = train_and_evaluate_final_model(
        "BDA+DNN", Sf_bda,
        X_train_full_feat, y_train_full,
        X_test_feat, y_test,
        DNN_TRAINING_PARAMS_FINAL, class_names
    )
    if metrics_bda_dnn: all_results['bda_dnn_final_eval'] = metrics_bda_dnn

    # BPSO+DNN
    metrics_bpso_dnn, history_bpso_dnn = train_and_evaluate_final_model(
        "BPSO+DNN", Sf_bpso,
        X_train_full_feat, y_train_full,
        X_test_feat, y_test,
        DNN_TRAINING_PARAMS_FINAL, class_names
    )
    if metrics_bpso_dnn: all_results['bpso_dnn_final_eval'] = metrics_bpso_dnn
    
    # --- 8. Salvar Resultados Consolidados ---
    results_file_path = os.path.join(RESULTS_DIR, "all_pipeline_results.json")
    try:
        with open(results_file_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResultados consolidados salvos em: {results_file_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados consolidados: {e}")

    print("\n\n--- Tabela Comparativa de Resultados (Conjunto de Teste) ---")
    print("-----------------------------------------------------------------------------------------------------------------")
    print("| Algoritmo | Features Sel. | Acurácia (%) | Sens_Cl0 (%) | Sens_Cl1 (%) | Sens_Cl2 (%) | Esp_Cl0 (%) | Esp_Cl1 (%) | Esp_Cl2 (%) | F1_Macro (%) |")
    print("|-----------|---------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|--------------|")

    def print_results_row(algo_name, results_dict):
        if not results_dict:
            print(f"| {algo_name:<9} | N/A           | N/A          | N/A          | N/A          | N/A          | N/A         | N/A         | N/A         | N/A          |")
            return

        num_feat = results_dict.get('num_selected_features', 'N/A')
        acc = results_dict.get('accuracy', 0) * 100
        report = results_dict.get('classification_report', {})
        sens_cl0 = report.get('Normal (0)', {}).get('recall', 0) * 100
        sens_cl1 = report.get('Interictal (1)', {}).get('recall', 0) * 100
        sens_cl2 = report.get('Ictal (2)', {}).get('recall', 0) * 100
        f1_macro = report.get('macro avg', {}).get('f1-score', 0) * 100
        
        specificities = results_dict.get('specificities', {})
        esp_cl0 = specificities.get('specificity_class_0', 0) * 100
        esp_cl1 = specificities.get('specificity_class_1', 0) * 100
        esp_cl2 = specificities.get('specificity_class_2', 0) * 100

        print(f"| {algo_name:<9} | {num_feat:<13} | {acc:^12.2f} | {sens_cl0:^12.2f} | {sens_cl1:^12.2f} | {sens_cl2:^12.2f} | {esp_cl0:^11.2f} | {esp_cl1:^11.2f} | {esp_cl2:^11.2f} | {f1_macro:^12.2f} |")

    if 'bda_dnn_final_eval' in all_results:
        print_results_row("BDA+DNN", all_results['bda_dnn_final_eval'])
    if 'bpso_dnn_final_eval' in all_results:
        print_results_row("BPSO+DNN", all_results['bpso_dnn_final_eval'])
    print("-----------------------------------------------------------------------------------------------------------------")


    total_execution_time = time.time() - start_time_total
    print(f"\nTempo total de execução da pipeline: {total_execution_time/60:.2f} minutos ({total_execution_time:.2f} segundos)")
    print("\n--- Fim da Execução ---")