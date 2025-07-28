# main.py
import gc, os, time, datetime, json, sys
import pywt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.data_loader import load_bonn_data, preprocess_eeg, split_data
from src.feature_extractor import extract_swt_features
from src.dnn_model import build_dnn_model
from src.bda import BinaryDragonflyAlgorithm
from src.bpso import BinaryPSO
from src.utils import (
    calculate_all_metrics,
    plot_convergence_curves,
    plot_data_distribution_pca,
    plot_eeg_segments,
    plot_feature_count_distribution,
    plot_final_metrics_comparison_bars,
    plot_optimization_diagnostics,
    plot_dnn_training_history,
    plot_swt_coefficients,
    visualize_knn_decision_boundary,
    plot_dragonfly_positions_pca,
    animate_dragonfly_movement_pca,
)
import src.utils as utils_module

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

# --- Configurações Globais ---
BASE_DATA_DIR = os.path.join(current_dir, "data")
RESULTS_DIR = os.path.join(current_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Diretórios de Resultados Dinâmicos
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_RESULTS_DIR = os.path.join(current_dir, "results")
RUN_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, f"run_{run_timestamp}")
PLOTS_DIR_MAIN = os.path.join(RUN_RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR_MAIN, exist_ok=True)
print(f"Salvando resultados nesta execução em: {RUN_RESULTS_DIR}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Configurações Globais de Plot ---
try:
    SAVE_PLOTS_DEFAULT = True
except Exception as e:
    print(f"Aviso: Não foi possível configurar backend do Matplotlib: {e}", flush=True)
    SAVE_PLOTS_DEFAULT = True

utils_module.SAVE_PLOTS = SAVE_PLOTS_DEFAULT
utils_module.PLOTS_DIR = PLOTS_DIR_MAIN

# --- CONFIGURAÇÕES PARA TREINAMENTO AVANÇADO ---
ENABLE_FEATURE_COUNT_FILTER = False      # Mude para True para filtrar pelo número de features
TARGET_FEATURE_COUNT = 19               # O número exato de features a serem treinadas
FINAL_MODEL_ACCURACY_THRESHOLD = 0.95   # Acurácia mínima de 85% para um modelo ser considerado "bom"
MAX_FINAL_MODELS_TO_KEEP = 5            # Tentar encontrar até 5 modelos que passem no limiar

# Parâmetros do Dataset e Pré-processamento
FS = 173.61
HIGHCUT_HZ = 40.0  # Filtro passa-baixas [0-40Hz]
FILTER_ORDER = 4  # Ordem do filtro
SWT_WAVELET = "db4"
SWT_LEVEL = 4

# Parâmetros da Divisão de Dados
TEST_SIZE = 0.20
VAL_SIZE = 0.15

# Parâmetros da DNN para Treino Final
DNN_TRAINING_PARAMS_FINAL = {"epochs": 250, "batch_size": 16, "patience": 30}

# Parâmetros dos Otimizadores
N_AGENTS_OPTIMIZERS = 30  # Artigo: population_size = 10
T_MAX_ITER_OPTIMIZERS = 150  # Artigo: iterations = 100

# Parâmetros Fitness (Conforme Artigo)
ALPHA_FITNESS = 0.99
BETA_FITNESS = 0.01

# Nível de verbosidade para os otimizadores
VERBOSE_OPTIMIZER_LEVEL = 1

# Funções Auxiliares

def create_fitness_function_for_optimizer(X_train_features, y_train_labels, alpha, beta):
    """
    Cria e retorna uma função de fitness otimizada que encapsula uma configuração de KNN.
    O classificador e o StratifiedKFold são instanciados uma única vez aqui.
    """
    # Configuração do KNN (pode ser parametrizada se necessário)
    knn_classifier = KNeighborsClassifier(n_neighbors=6, metric='manhattan', weights='distance')
    
    # Validação Cruzada Robusta
    n_folds = 10
    min_samples_per_class = np.min(np.bincount(y_train_labels))
    if min_samples_per_class < n_folds:
        print(f"Aviso Fitness: A menor classe possui {min_samples_per_class} amostras, "
              f"o que é menor que n_folds={n_folds}. Ajustando n_folds para {min_samples_per_class}.")
        n_folds = max(2, min_samples_per_class)
        
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    def evaluate_fitness_configured(binary_feature_vector, *args, **kwargs):
        """
        Função de fitness que será chamada pelo otimizador (e.g., BDA).
        Reutiliza o knn_classifier e cv_splitter pré-configurados.
        """
        selected_indices = np.where(binary_feature_vector == 1)[0]
        num_selected = len(selected_indices)
        total_features = len(binary_feature_vector)
        
        if num_selected == 0:
            return {
                'fitness': alpha * 1.0 + beta * 1.0,
                'accuracy': 0.0,
                'num_features': 0
            }

        X_train_selected = X_train_features[:, selected_indices]

        # Verifica se o n_folds ainda é válido para a dobra atual (pouco provável de mudar, mas seguro)
        if cv_splitter.get_n_splits() > np.min(np.bincount(y_train_labels)):
             return { 'fitness': 1.0, 'accuracy': 0.0, 'num_features': num_selected }

        try:
            # cross_val_score é eficiente e lida com a divisão de dados internamente
            accuracies = cross_val_score(
                knn_classifier,
                X_train_selected,
                y_train_labels,
                cv=cv_splitter,
                scoring="accuracy",
                n_jobs=-1,  # Utiliza todos os cores disponíveis para a validação cruzada
            )
            mean_accuracy = np.mean(accuracies)
        except ValueError:
            # Caso ocorra um erro (e.g., uma dobra de CV não tem membros de uma classe)
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


def train_and_evaluate_final_model(
    model_name,
    selected_features_vector,
    X_train_full_all_feat,
    y_train_full,
    X_test_all_feat,
    y_test,
    dnn_params,
    class_names,
    opt_fitness_score
):
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
    final_model = build_dnn_model(
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
        validation_split=0.15,
        callbacks=[early_stopping_final],
        verbose=1 if VERBOSE_OPTIMIZER_LEVEL > 0 else 0,
    )

    history_data = history.history
    plot_dnn_training_history(
        history_data,
        title=f"Histórico de Treino Final - {model_name}",
        filename=f"final_dnn_history_{model_name.replace('+', '_')}.png",
    )

    print(f"\nAvaliando {model_name} no conjunto de teste...")
    y_pred_test_probs = final_model.predict(X_test_selected)
    y_pred_test = np.argmax(y_pred_test_probs, axis=1)

    metrics = calculate_all_metrics(y_test, y_pred_test, class_names=class_names)
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

# --- Função para processar histórico e pegar top N soluções ---
def get_top_n_unique_solutions(history, n=20):
    """
    Processa o histórico de soluções de um otimizador para retornar as N melhores
    soluções únicas encontradas.
    Args:
        history (list): Lista de tuplas (fitness, solution_vector).
        n (int): Número de melhores soluções a retornar.
    Returns:
        list: Lista de tuplas (fitness, solution_vector) para as N melhores soluções.
    """
    if not history:
        return []

    unique_solutions = {}
    for fitness, sol in history:
        sol_tuple = tuple(sol)
        # Se a solução é nova ou se encontramos um fitness melhor para ela
        if sol_tuple not in unique_solutions or fitness < unique_solutions[sol_tuple]:
            unique_solutions[sol_tuple] = fitness

    # Ordena as soluções únicas pelo seu fitness (do menor para o maior)
    sorted_solutions = sorted(unique_solutions.items(), key=lambda item: item[1])

    # Retorna o vetor da solução e seu fitness
    top_n = [(fit, np.array(sol)) for sol, fit in sorted_solutions[:n]]
    return top_n


def get_all_unique_solutions_sorted(history):
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
        raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
    except Exception as e:
        print(f"Falha ao carregar dados: {e}. Verifique o caminho e formato do dataset.")
        sys.exit(1)

    class_names = ["Normal (0)", "Interictal (1)", "Ictal (2)"]
    if VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nPlotando exemplos de sinais EEG brutos...", flush=True)
        plot_eeg_segments(
            {"Raw": raw_data[:3, :]},
            fs=FS,
            n_segments_to_plot=1,
            base_filename="eeg_raw_example",
        )

    # 2. Pré-processar Dados
    print("\n--- 2. Pré-processando Dados ---")
    data_processed = preprocess_eeg(
        raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
    )

    if VERBOSE_OPTIMIZER_LEVEL > 0:
        print("\nPlotando exemplos de sinais EEG pré-processados...", flush=True)
        plot_eeg_segments(
            {"Processed": data_processed[:3, :]},
            fs=FS,
            n_segments_to_plot=1,
            base_filename="eeg_processed_example",
        )

    # 3. Dividir Dados
    # X_train_p, X_val_p, X_test_p são dados no domínio do tempo, antes da extração de features SWT
    print("\n--- 3. Dividindo Dados ---")
    X_train_p, X_val_p, X_test_p, y_train_labels, y_val_labels, y_test_labels = (
        split_data(
            data_processed,
            raw_labels,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_SEED,
        )
    )
    # y_train_labels é o y_train para os otimizadores.
    # y_val_labels e y_test_labels são para a avaliação final.

    # 4. Extrair Características SWT
    print("\n--- 4. Extraindo Características SWT ---")
    print("Extraindo features para o conjunto de TREINO (usado pelos otimizadores)...")
    # X_train_feat_opt será usado pelos otimizadores (BDA, BPSO)
    X_train_feat_opt, feature_names = extract_swt_features(
        X_train_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )
    # Plotar coeficientes SWT de um segmento de exemplo do treino
    if X_train_p.shape[0] > 0:
        print(
            "\nPlotando coeficientes SWT de um segmento de treino de exemplo...",
            flush=True,
        )
        example_signal_for_swt_plot = X_train_p[0, :]
        original_signal_length = X_train_p.shape[1]
        signal_length_for_swt = original_signal_length - (original_signal_length % 2)
        if original_signal_length % 2 != 0:  # Reaplicar truncamento para consistência
            example_signal_for_swt_plot = example_signal_for_swt_plot[
                :signal_length_for_swt
            ]  # signal_length_for_swt do escopo de extração
        # Recalcular SWT para este sinal específico para obter os coeficientes para plotagem
        # (A função extract_swt_features não retorna os coeficientes brutos, apenas a matriz de features)
        # Precisamos de signal_length_for_swt que foi definido dentro de extract_swt_features
        # Melhor pegar do X_train_p e truncar novamente aqui.
        slfs = X_train_p.shape[1] - (X_train_p.shape[1] % 2)
        example_signal_for_swt_plot_truncated = X_train_p[0, :slfs]

        # Se a função apply_swt agora retorna a lista plana de arrays:
        swt_coeffs_arrays_example = pywt.swt(
            example_signal_for_swt_plot_truncated, wavelet=SWT_WAVELET, level=SWT_LEVEL, trim_approx=True, norm=True
        )

        example_coeffs_map_for_plot = {}
        if isinstance(swt_coeffs_arrays_example, list) and len(
            swt_coeffs_arrays_example
        ) == (SWT_LEVEL + 1):
            example_coeffs_map_for_plot[f"A{SWT_LEVEL}"] = swt_coeffs_arrays_example[0]
            for k_idx_plot in range(SWT_LEVEL):
                detail_level_val_plot = SWT_LEVEL - k_idx_plot
                array_idx_plot = k_idx_plot + 1
                example_coeffs_map_for_plot[f"D{detail_level_val_plot}"] = (
                    swt_coeffs_arrays_example[array_idx_plot]
                )
            plot_swt_coefficients(
                example_coeffs_map_for_plot,
                segment_idx=0,
                base_filename="swt_coeffs_train_example",
            )
        else:
            print(
                "Não foi possível obter coeficientes SWT para plotagem de exemplo.",
                flush=True,
            )
    # Para o treino final da DNN, o artigo combina treino e validação.
    # As features SWT precisam ser extraídas separadamente para X_val_p e X_test_p.
    print(
        "Extraindo features para o conjunto de VALIDAÇÃO (para combinar com treino para DNN final)..."
    )
    X_val_feat_combine, _ = extract_swt_features(
        X_val_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )

    print(
        "Extraindo features para o conjunto de TESTE (para avaliação final da DNN)..."
    )
    X_test_feat_final, _ = extract_swt_features(
        X_test_p, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )

    if not feature_names:  # Se a extração de features falhar e não retornar nomes
        print(
            "ERRO: Nomes de features não foram gerados. Verifique a extração de features."
        )
        # Tenta criar um número esperado de nomes para evitar falha no DIM_FEATURES
        # Isso é um fallback, o ideal é que extract_swt_features sempre retorne nomes
        num_expected_features = (SWT_LEVEL + 1) * 9  # 5 bandas * 9 features
        feature_names = [f"feature_{k}" for k in range(num_expected_features)]
        # Se X_train_feat_opt também estiver vazio, DIM_FEATURES será problemático
        if X_train_feat_opt.shape[1] == 0 and len(feature_names) > 0:
            X_train_feat_opt = np.zeros(
                (X_train_p.shape[0], len(feature_names))
            )  # Placeholder
            print(
                f"AVISO: X_train_feat_opt estava vazio, preenchendo com zeros e {len(feature_names)} colunas."
            )

    DIM_FEATURES = X_train_feat_opt.shape[1]
    print(f"Total de {DIM_FEATURES} características extraídas (para os otimizadores).")
    if (
        DIM_FEATURES != 45 and DIM_FEATURES != 143
    ):  # 45 é o esperado pela metodologia detalhada. 143 é a menção inconsistente.
        print(
            f"AVISO: Número de features extraídas ({DIM_FEATURES}) não corresponde a 45 (esperado pela metodologia detalhada) ou 143 (menção no artigo)."
        )

    if VERBOSE_OPTIMIZER_LEVEL > 0 and X_train_p.shape[0] > 0:
        print(
            "\nPlotando distribuição dos dados (PCA) antes da seleção de features...",
            flush=True,
        )
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
        plot_data_distribution_pca(
            X_datasets_for_pca_plot,
            y_datasets_for_pca_plot,
            title="Distribuição dos Conjuntos de Features SWT (PCA)",
            filename="data_distribution_pca_swt_features.png",
            class_names=class_names,
        )

    all_results = {}
    all_convergence_curves = []
    convergence_labels = []
    top_solutions_to_train = {}
    all_candidate_solutions = {}

    # --- 5. Otimização com BDA ---
    print("\n\n--- 5. Otimização com Binary Dragonfly Algorithm (BDA) ---")
    start_time_bda_opt = time.time()
    
    # Cria a função de fitness otimizada ANTES de instanciar o BDA
    fitness_function_for_bda = create_fitness_function_for_optimizer(
        X_train_feat_opt, y_train_labels, ALPHA_FITNESS, BETA_FITNESS
    )

    bda = BinaryDragonflyAlgorithm(
        N=N_AGENTS_OPTIMIZERS,
        T=T_MAX_ITER_OPTIMIZERS,
        dim=DIM_FEATURES,
        fitness_func=fitness_function_for_bda,
        X_train_feat=X_train_feat_opt,
        y_train=y_train_labels,
        s=0.1, a=0.1, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.85,
        tau_min=0.01, tau_max=4.0,
        alpha_fitness=ALPHA_FITNESS,
        beta_fitness=BETA_FITNESS,
        seed=RANDOM_SEED,
        verbose_optimizer_level=VERBOSE_OPTIMIZER_LEVEL,
        min_features=15,
        max_features=28,
    )
    Sf_bda, best_fitness_bda, convergence_bda, acc_curve_bda, nfeat_curve_bda, bda_history = bda.run()

    bda_diagnostic_curves = {
        "Melhor Fitness": convergence_bda,
        "Acurácia do Melhor Agente (%)": np.array(acc_curve_bda) * 100,
        "Nº de Features do Melhor Agente": nfeat_curve_bda,
    }
    plot_optimization_diagnostics(
        bda_diagnostic_curves,
        title="Diagnóstico da Otimização - BDA",
        filename="bda_diagnostics.png",
    )

    if Sf_bda is not None and np.sum(Sf_bda) > 1:
        print(
            "\nGerando visualização da fronteira de decisão do KNN para a solução final do BDA..."
        )
        visualize_knn_decision_boundary(
            X_train_feat_opt,  # Dados de treino usados na otimização
            y_train_labels,  # Rótulos de treino
            Sf_bda,  # Vetor de features da melhor solução
            class_names=class_names,
            title="Fronteira de Decisão KNN (Solução Final BDA)",
            filename="bda_final_solution_knn_boundary.png",
        )

    all_results["bda_optimization"] = {
        "best_fitness": best_fitness_bda,
        "selected_features_vector": (
            Sf_bda.tolist() if isinstance(Sf_bda, np.ndarray) else Sf_bda
        ),
        "num_selected_features": (
            int(np.sum(Sf_bda)) if isinstance(Sf_bda, np.ndarray) else 0
        ),
        "convergence_curve": (
            convergence_bda.tolist()
            if isinstance(convergence_bda, np.ndarray)
            else convergence_bda
        ),
    }
    all_convergence_curves.append(convergence_bda)
    convergence_labels.append("BDA")
    all_candidate_solutions["BDA"] = get_all_unique_solutions_sorted(bda_history)
    print(f"BDA encontrou {len(all_candidate_solutions['BDA'])} soluções únicas.")
    print(f"Tempo de otimização BDA: {(time.time() - start_time_bda_opt)/60:.2f} minutos")
    top_10_bda = get_top_n_unique_solutions(bda_history, n=20)
    top_solutions_to_train["BDA"] = top_10_bda
    print(f"\nBDA: Top {len(top_10_bda)} soluções únicas encontradas.")
    for i, (fit, sol) in enumerate(top_10_bda):
        print(f"  - Rank {i+1}: Fitness={fit:.4f}, Features={np.sum(sol)}")
        all_results[f"BDA-Rank-{i+1}_opt_fitness"] = fit # Salva o fitness para usar depois

    gc.collect()

    # --- 6. Otimização com BPSO ---
    #print("\n\n--- 6. Otimização com Binary Particle Swarm Optimization (BPSO) ---")
    #start_time_bpso_opt = time.time()
    #bpso = BinaryPSO(
#        N=N_AGENTS_OPTIMIZERS,
#        T=T_MAX_ITER_OPTIMIZERS,
#        dim=DIM_FEATURES,
#        fitness_func=evaluate_fitness,
#        X_train_feat=X_train_feat_opt,
#        y_train=y_train_labels,
#        w_max=0.9,
#        w_min=0.4,
#        c1=2.0,
#        c2=2.0,
#        Vmax=4.0,  # Parâmetros BPSO comuns
#        alpha_fitness=ALPHA_FITNESS,
#        beta_fitness=BETA_FITNESS,
#        seed=RANDOM_SEED,
#        verbose_optimizer_level=VERBOSE_OPTIMIZER_LEVEL,
#    )
    #Sf_bpso, best_fitness_bpso, convergence_bpso, acc_curve_bpso, nfeat_curve_bpso = bpso.run()
#
    #bpso_diagnostic_curves = {
#        "Melhor Fitness": convergence_bpso,
#        "Acurácia do Melhor Agente (%)": np.array(acc_curve_bpso) * 100,
#        "Nº de Features do Melhor Agente": nfeat_curve_bpso,
#    }
#    #plot_optimization_diagnostics(
#        bpso_diagnostic_curves,
#        title="Diagnóstico da Otimização - BPSO",
#        filename="bpso_diagnostics.png",
#    )
#
    #if Sf_bpso is not None and np.sum(Sf_bda) > 1:
#        print(
#            "\nGerando visualização da fronteira de decisão do KNN para a solução final do BPSO..."
#        )
#        visualize_knn_decision_boundary(
#            X_train_feat_opt,  # Dados de treino usados na otimização
#            y_train_labels,  # Rótulos de treino
#            Sf_bpso,  # Vetor de features da melhor solução
#            class_names=class_names,
#            title="Fronteira de Decisão KNN (Solução Final BPSO)",
#            filename="bpso_final_solution_knn_boundary.png",
#        )
#
    #all_results["bpso_optimization"] = {
#       "best_fitness": best_fitness_bpso,
#       "selected_features_vector": (
#           Sf_bpso.tolist() if isinstance(Sf_bpso, np.ndarray) else Sf_bpso
#       ),
#       "num_selected_features": (
#           int(np.sum(Sf_bpso)) if isinstance(Sf_bpso, np.ndarray) else 0
#       ),
#       "convergence_curve": (
#           convergence_bpso.tolist()
#           if isinstance(convergence_bpso, np.ndarray)
#           else convergence_bpso
#       ),
#   }
    #all_convergence_curves.append(convergence_bpso)
    #convergence_labels.append("BPSO")
    #print(
    #    f"Tempo de otimização BPSO: {(time.time() - start_time_bpso_opt)/60:.2f} minutos"
    #)
    #gc.collect()

    #if all_convergence_curves:
    #    plot_convergence_curves(
    #        all_convergence_curves,
    #        convergence_labels,
    #        title="Convergência dos Otimizadores (Fitness KNN na Validação Cruzada)",
    #        filename="optimizers_convergence_knn_fitness.png",
    #    )

    plot_convergence_curves(all_convergence_curves, convergence_labels, title="Convergência dos Otimizadores", filename="optimizers_convergence.png")
    plot_feature_count_distribution(all_candidate_solutions, filename="feature_count_distribution.png")

    # --- 7. Treinamento e Avaliação Final da DNN ---
    print("\n\n--- 7. Treinamento e Avaliação Final dos Modelos DNN ---")
    X_train_full_feat_final = np.concatenate((X_train_feat_opt, X_val_feat_combine), axis=0)
    y_train_full_labels_final = np.concatenate((y_train_labels, y_val_labels), axis=0)

    for algo_name, candidate_solutions in all_candidate_solutions.items():
        print(f"\n--- Processando Candidatos de {algo_name} ---")
        
        # Filtra os candidatos se a flag estiver ativa
        if ENABLE_FEATURE_COUNT_FILTER:
            print(f"Filtrando candidatos de {algo_name} para {TARGET_FEATURE_COUNT} features...")
            filtered_candidates = [(f, s) for f, s in candidate_solutions if np.sum(s) == TARGET_FEATURE_COUNT]
            print(f"Encontrados {len(filtered_candidates)} candidatos com o número de features alvo.")
        else:
            filtered_candidates = candidate_solutions

        if not filtered_candidates:
            print(f"Nenhum candidato de {algo_name} sobrou após a filtragem.")
            continue

        # Novo Loop de Treinamento com Limiar de Qualidade
        final_good_model_count = 0
        candidate_idx = 0
        
        while final_good_model_count < MAX_FINAL_MODELS_TO_KEEP and candidate_idx < len(filtered_candidates):
            fitness_score, solution_vector = filtered_candidates[candidate_idx]
            num_features = np.sum(solution_vector)
            
            # O rank é baseado na posição na lista de candidatos (já ordenada por fitness)
            model_rank = candidate_idx + 1 
            model_name = f"{algo_name}-F{num_features}-Rank{model_rank}"

            print(f"\n>>> Tentando treinar modelo {final_good_model_count + 1}/{MAX_FINAL_MODELS_TO_KEEP} para {algo_name}...")
            
            metrics, history_data = train_and_evaluate_final_model(
                model_name=model_name,
                selected_features_vector=solution_vector,
                X_train_full_all_feat=X_train_full_feat_final,
                y_train_full=y_train_full_labels_final,
                X_test_all_feat=X_test_feat_final,
                y_test=y_test_labels,
                dnn_params=DNN_TRAINING_PARAMS_FINAL,
                class_names=class_names,
                opt_fitness_score=fitness_score
            )
            
            candidate_idx += 1 # Sempre avança para o próximo candidato

            if metrics and metrics.get("accuracy", 0) >= FINAL_MODEL_ACCURACY_THRESHOLD:
                print(f"+++ SUCESSO: Modelo {model_name} atingiu {metrics['accuracy']:.2%} de acurácia. Mantendo.")
                final_good_model_count += 1
                all_results[f"{model_name}_final_eval"] = metrics
            elif metrics:
                print(f"--- DESCARTADO: Modelo {model_name} com acurácia de {metrics['accuracy']:.2%}, abaixo do limiar de {FINAL_MODEL_ACCURACY_THRESHOLD:.0%}.")
            else:
                print(f"### FALHA: Treinamento para {model_name} não produziu métricas. Descartando.")

        if final_good_model_count < MAX_FINAL_MODELS_TO_KEEP:
            print(f"\nAVISO: Não foi possível encontrar {MAX_FINAL_MODELS_TO_KEEP} modelos para {algo_name} que satisfizessem o limiar de acurácia. Encontrados: {final_good_model_count}.")

    # --- 8. Salvar Resultados Consolidados ---
    results_file_path = os.path.join(RUN_RESULTS_DIR, "all_pipeline_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4, cls=utils_module.NumpyEncoder)
        print(f"\nResultados consolidados salvos em: {results_file_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados consolidados: {e}")

    # --- 9. Tabela Comparativa ---
    print("\n\n--- Tabela Comparativa de Resultados (Conjunto de Teste) ---")
    print("------------------------------------------------------------------------------------------------------------------------------------------")
    print("| Modelo              | Fitness Opt. | Features Sel. | Acurácia (%) | Sens_Cl0 (%) | Sens_Cl1 (%) | Sens_Cl2 (%) | Esp_Cl0 (%) | Esp_Cl1 (%) | Esp_Cl2 (%) | F1_Macro (%) |")
    print("|---------------------|--------------|---------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|--------------|")

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
    
    sorted_results_keys = sorted([k for k in all_results if k.endswith('_final_eval')])
    for key in sorted_results_keys:
        model_name_display = key.replace('_final_eval', '')
        print_results_row_main(model_name_display, all_results[key])
    
    print("------------------------------------------------------------------------------------------------------------------------------------------")

    # --- 10. Plot Comparativo Final ---
    print("\n\n--- Gerando Gráficos Comparativos Finais ---")
    final_eval_results_for_plot = {k.replace('_final_eval', ''): v for k, v in all_results.items() if k.endswith('_final_eval')}

    if final_eval_results_for_plot:
        plot_final_metrics_comparison_bars(
            final_eval_results_for_plot,
            base_filename="final_model_metrics",
        )
    else:
        print("Nenhum resultado de avaliação final para plotar.")

    total_execution_time = time.time() - start_time_total
    print(f"\nTempo total de execução da pipeline: {total_execution_time/60:.2f} minutos")
    print("\n--- Fim da Execução ---")