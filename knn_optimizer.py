# knn_optimizer.py
import os
import sys
import time
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss,
)

from src.utils import generate_aggregate_run_plots

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from data_loader import load_bonn_data, preprocess_eeg
    from feature_extractor import extract_swt_features
    from bda import BinaryDragonflyAlgorithm
except ImportError as e:
    print(f"Erro ao importar módulos de 'src': {e}")
    print(
        "Certifique-se de que este script está no diretório raiz e o restante do código está em 'src/'."
    )
    sys.exit(1)
except Exception:
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from data_loader import load_bonn_data, preprocess_eeg
    from feature_extractor import extract_swt_features
    from bda import BinaryDragonflyAlgorithm


# CONFIGURAÇÕES GERAIS

# Estilo dos gráficos para publicações científicas
plt.style.use("seaborn-v0_8-paper")
sns.set_theme(
    style="ticks",
    palette="viridis",
)
sns.color_palette("rocket_r", as_cmap=True)

# CONFIGURAÇÕES DA OTIMIZAÇÃO
TESTING_PIPELINE = False
PARALLEL_PROCESSES = 4
N_RUNS_PER_CONFIG = 1  # Número de execuções de BDA por configuração de KNN
if TESTING_PIPELINE:
    # Dicionário para testes rápidos do pipeline
    KNN_PARAM_GRID = {
        "n_neighbors": [1, 2],
        "metric": ["manhattan", "minkowski"],
        "weights": ["distance", "uniform"],
        "algorithm": ["ball_tree"],
        "leaf_size": [30, 50],
    }
else:
    KNN_PARAM_GRID = {
        "n_neighbors": [4, 5, 6, 7, 8],
        "metric": ["manhattan"],
        "weights": ["distance"],
        "algorithm": ["auto"]
    }

#  Configurações do BDA e do Dataset
RANDOM_SEED = 42
N_AGENTS_BDA = 50
MAX_ITER_BDA = 125
ALPHA_FITNESS = 0.99
BETA_FITNESS = 0.01
BASE_DATA_DIR = os.path.join(current_dir, "data")
SWT_WAVELET = "db4"
SWT_LEVEL = 4

#  Diretório de Resultados
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(current_dir, "results", f"knn_optimization_{run_timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Salvando resultados e gráficos em: {RESULTS_DIR}")


def generate_knn_configs(param_grid):
    """
    Gera uma lista de configurações de KNN, lidando com a dependência
    do parâmetro 'p' com a 'metric'.
    """
    configs = []
    keys = param_grid.keys()
    all_combinations = list(itertools.product(*param_grid.values()))

    for combo in all_combinations:
        config_dict = dict(zip(keys, combo))

        # Ignora parâmetros irrelevantes para manter a lista de configs mais limpa
        if config_dict["metric"] != "minkowski":
            config_dict.pop("p", None)

        if config_dict["algorithm"] in ["auto", "brute"]:
            config_dict.pop("leaf_size", None)

        if config_dict not in configs:
            configs.append(config_dict)

    print(f"Geradas {len(configs)} configurações únicas de KNN para teste.")
    return configs


def create_fitness_function(knn_config, X_train_features, y_train_labels):
    """
    Cria uma função de fitness que encapsula uma configuração do KNN.
    """

    knn_classifier = KNeighborsClassifier(**knn_config)
    n_folds = 10
    min_samples_per_class = np.min(np.bincount(y_train_labels))
    if min_samples_per_class < n_folds:
        n_folds = max(2, min_samples_per_class)    
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    def evaluate_fitness_configured(binary_feature_vector, *args, **kwargs):
        """
        Função de fitness que será chamada pelo BDA.
        """
        selected_indices = np.where(binary_feature_vector == 1)[0]
        num_selected = len(selected_indices)
        total_features = len(binary_feature_vector)
        X_train_selected = X_train_features[:, selected_indices]

        if (num_selected == 0 or X_train_selected.shape[1] == 0):
             return { "fitness": 1.0, "accuracy": 0.0, "num_features": 0 }

        try:
            accuracies = cross_val_score(
                knn_classifier,
                X_train_selected,
                y_train_labels,
                cv=cv_splitter,
                scoring="accuracy",
                n_jobs=1,
            )
            mean_accuracy = np.mean(accuracies)
        except ValueError:
            mean_accuracy = 0.0

        error_rate = 1.0 - mean_accuracy
        feature_ratio = num_selected / total_features
        fitness = ALPHA_FITNESS * error_rate + BETA_FITNESS * feature_ratio

        return {
            "fitness": fitness,
            "accuracy": mean_accuracy,
            "num_features": num_selected,
        }

    return evaluate_fitness_configured


def get_full_metrics(fitness_func, solution_vector, X_train, y_train):
    """
    Roda a avaliação uma vez fora do BDA para obter métricas detalhadas.
    """
    selected_indices = np.where(solution_vector == 1)[0]
    if len(selected_indices) == 0:
        return 0.0, 0, 1.0

    results_dict = fitness_func(solution_vector)
    accuracy = results_dict.get("accuracy", 0.0)
    num_features = results_dict.get("num_features", 0)
    fitness = results_dict.get("fitness", 1.0)
    return accuracy, num_features, fitness


def run_optimization_pipeline():
    """
    Executa todo o pipeline de otimização, rodando múltiplas execuções por configuração.
    """
    print(" 1. Carregando e Pré-processando Dados ")
    try:
        raw_data, raw_labels = load_bonn_data(BASE_DATA_DIR)
        processed_data = preprocess_eeg(raw_data, fs=173.61, highcut_hz=40.0)
    except Exception as e:
        print(f"Falha ao carregar/processar dados: {e}")
        return

    print("\n 2. Extraindo Características (Features) ")
    all_features, feature_names = extract_swt_features(
        processed_data, wavelet=SWT_WAVELET, level=SWT_LEVEL
    )

    X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        all_features,
        raw_labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=raw_labels,
    )
    ######### TESTING NEW CODE #########

    # Step 1: Run 10 BDA optimizations and store best feature sets
    print("\n 3. Executando 20 otimizações BDA para seleção de features...")
    bda_feature_sets = []
    bda_fitnesses = []
    bda_convergence_curves = []
    for bda_run_idx in range(20):
        print(f"BDA Run {bda_run_idx+1}/20...")
        bda = BinaryDragonflyAlgorithm(
            N=N_AGENTS_BDA,
            T=MAX_ITER_BDA,
            dim=all_features.shape[1],
            fitness_func=create_fitness_function({
                "n_neighbors": 5,
                "metric": "manhattan",
                "weights": "distance",
                "algorithm": "auto"
            }, X_train_feat, y_train),
            X_train_feat=X_train_feat,
            y_train=y_train,
            seed=RANDOM_SEED + bda_run_idx * 100,
            verbose_optimizer_level=0,
            min_features=15,
            max_features=28,
        )
        best_solution, best_fitness, convergence_curve, _, _, _ = bda.run()
        bda_feature_sets.append(best_solution)
        bda_fitnesses.append(best_fitness)
        bda_convergence_curves.append(convergence_curve)

    print(f"\nBDA feature selection complete. {len(bda_feature_sets)} sets stored.")

    # Step 2: For each KNN config, run KNN tests using each BDA feature set, 3 times per config
    knn_configurations = generate_knn_configs(KNN_PARAM_GRID)
    total_configs = len(knn_configurations)
    print(f"\n 4. Testando cada configuração de KNN ({total_configs} configs), cada uma com 10 conjuntos de features BDA, 3 execuções cada...")

    def knn_test_run(config_id, config, bda_idx, feature_vector, run_idx):
        selected_indices = np.where(feature_vector == 1)[0]
        if len(selected_indices) == 0:
            return None
        knn = KNeighborsClassifier(**config)
        knn.fit(X_train_feat[:, selected_indices], y_train)
        y_pred = knn.predict(X_test_feat[:, selected_indices])
        y_prob = knn.predict_proba(X_test_feat[:, selected_indices])
        acc = np.mean(y_pred == y_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        return {
            "config_id": config_id,
            "bda_idx": bda_idx,
            "run_idx": run_idx,
            **config,
            "num_selected_features": len(selected_indices),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
            "log_loss": ll,
        }

    all_knn_results = []
    for config_id, config in enumerate(knn_configurations):
        print(f"\nConfiguração {config_id+1}/{total_configs}: {config}")
        # Prepare all jobs for this config
        jobs = []
        for bda_idx, feature_vector in enumerate(bda_feature_sets):
            for run_idx in range(1):
                jobs.append((config_id, config, bda_idx, feature_vector, run_idx))
        # Run in parallel
        run_results = Parallel(n_jobs=PARALLEL_PROCESSES, verbose=11)(
            delayed(knn_test_run)(*job) for job in jobs
        )
        # Filter out None results
        all_knn_results.extend([r for r in run_results if r is not None])

    # Step 3: Aggregate results and save
    all_knn_df = pd.DataFrame(all_knn_results)
    all_knn_df.to_csv(os.path.join(RESULTS_DIR, "knn_all_runs_results.csv"), index=False)

    # Aggregate by config
    agg_df = all_knn_df.groupby(["config_id", "n_neighbors", "metric", "weights", "algorithm"]).agg({
        "accuracy": ["mean", "std"],
        "balanced_accuracy": ["mean", "std"],
        "cohen_kappa": ["mean", "std"],
        "mcc": ["mean", "std"],
        "log_loss": ["mean", "std"],
        "num_selected_features": ["mean", "std"],
    }).reset_index()
    agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns.values]
    agg_df.to_csv(os.path.join(RESULTS_DIR, "knn_aggregate_results.csv"), index=False)

    ########### END OF TESTING NEW CODE ###########

    print(f"Total de {all_features.shape[1]} features extraídas.")
    print(f"Dados de treino: {X_train_feat.shape}, Dados de teste: {X_test_feat.shape}")

    knn_configurations = generate_knn_configs(KNN_PARAM_GRID)
    total_configs = len(knn_configurations)
    print(f"\n 3. Iniciando Otimização em Lote ({total_configs} configurações de KNN, {N_RUNS_PER_CONFIG} execuções cada) ")

    # Lista para armazenar resultados detalhados de cada execução
    all_runs_results = []
    # Lista para armazenar o melhor resultado de cada configuração
    best_per_config = []

    for config_id, config in enumerate(knn_configurations):
        print(f"\nConfiguração {config_id+1}/{total_configs}: {config}")
        # Executa N_RUNS_PER_CONFIG vezes em paralelo para cada configuração
        run_results = Parallel(n_jobs=PARALLEL_PROCESSES, verbose=11)(
            delayed(process_single_config)(
                config_id, config, X_train_feat, y_train, all_features.shape[1], run_idx
            ) for run_idx in range(N_RUNS_PER_CONFIG)
        )
        # Salva todos os resultados detalhados
        all_runs_results.extend(run_results)
        # Seleciona o melhor resultado desta configuração, ignorando execuções None
        valid_runs = [r for r in run_results if r is not None]
        if valid_runs:
            best_run = min(valid_runs, key=lambda x: x["best_fitness"])
            best_per_config.append(best_run)

    # DataFrame com todos os resultados de todas as execuções
    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_df.to_csv(os.path.join(RESULTS_DIR, "optimization_all_runs_results.csv"), index=False)

    # DataFrame com o melhor de cada configuração (como era antes)
    results_df = pd.DataFrame(best_per_config)
    results_df = results_df.sort_values(by="best_fitness", ascending=True)
    df_to_save = results_df.drop(columns=["best_solution_vector", "convergence_curve"])
    df_to_save.to_csv(
        os.path.join(RESULTS_DIR, "optimization_summary_results.csv"), index=False
    )
    print("\n\n 4. Análise e Visualização dos Resultados ")
    print("Top 5 melhores configurações encontradas (ordenadas por fitness):")
    print(df_to_save.head().to_markdown(index=False))

    results_df_for_similarity = results_df.sort_index()

    # Gera relatórios e gráficos para o melhor de cada configuração
    generate_reports_and_plots(
        results_df,
        results_df_for_similarity,
        X_train_feat,
        y_train,
        X_test_feat,
        y_test,
        feature_names,
    )

    # Gera relatórios e gráficos agregados considerando todas as execuções
    generate_aggregate_run_plots(
        all_runs_df,
        knn_configurations,
        feature_names,
        X_train_feat,
        y_train,
        X_test_feat,
        y_test,
    )


def process_single_config(config_id, config, X_train_feat, y_train, feature_dim, run_idx=0):
    """
    Executa o pipeline de otimização para uma única configuração de KNN.
    Esta função será chamada em paralelo e múltiplas vezes por configuração.
    """
    start_run_time = time.time()
    fitness_function_for_bda = create_fitness_function(config, X_train_feat, y_train)

    # Semente diferente para cada execução
    bda = BinaryDragonflyAlgorithm(
        N=N_AGENTS_BDA,
        T=MAX_ITER_BDA,
        dim=feature_dim,
        fitness_func=fitness_function_for_bda,
        X_train_feat=X_train_feat,
        y_train=y_train,
        seed=RANDOM_SEED + config_id * 1000 + run_idx,
        verbose_optimizer_level=0,
        min_features=15,
        max_features=28,
    )

    best_solution, best_fitness, convergence_curve, _, _, _ = bda.run()

    mean_accuracy, num_features, _ = get_full_metrics(
        fitness_function_for_bda, best_solution, X_train_feat, y_train
    )
    elapsed_time = time.time() - start_run_time
    print(f"Config {config_id} Run {run_idx+1} finalizou em {elapsed_time:.2f}s. Accuracy: {mean_accuracy:.4f}, Features: {num_features}, Fitness: {best_fitness:.4f}")
    return {
        "config_id": config_id,
        "run_idx": run_idx,
        **config,
        "best_fitness": best_fitness,
        "num_selected_features": num_features,
        "mean_accuracy_cv": mean_accuracy,
        "execution_time_sec": elapsed_time,
        "best_solution_vector": best_solution,
        "convergence_curve": convergence_curve,
    }


def generate_reports_and_plots(
    df_sorted_by_fitness,
    df_sorted_by_id,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
):
    """
    Gera todos os gráficos de análise e relatórios de texto.
    """
    if df_sorted_by_fitness.empty:
        print("DataFrame de resultados vazio. Nenhum gráfico será gerado.")
        return

    print("\nGerando relatórios e gráficos para publicação...")

    df = df_sorted_by_fitness.copy()
    best_config_row = df.iloc[0]
    best_knn_params = {
        key: best_config_row[key]
        for key in KNN_PARAM_GRID.keys()
        if key in best_config_row and pd.notna(best_config_row[key])
    }
    if 'leaf_size' in best_knn_params:
        best_knn_params['leaf_size'] = int(best_knn_params['leaf_size'])
    best_feature_vector = best_config_row["best_solution_vector"]
    selected_indices = np.where(best_feature_vector == 1)[0]
    class_names = [f"Classe {i}" for i in sorted(np.unique(y_train))]
    class_labels = sorted(np.unique(y_train))

    #  Avaliação do Melhor Modelo e Cálculo de Métricas
    if len(selected_indices) > 0:
        final_knn = KNeighborsClassifier(**best_knn_params)
        final_knn.fit(X_train[:, selected_indices], y_train)
        y_pred = final_knn.predict(X_test[:, selected_indices])
        y_prob = final_knn.predict_proba(X_test[:, selected_indices])

        report_str = classification_report(y_test, y_pred, target_names=class_names)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        ll = log_loss(y_test, y_prob)

    #  Relatório de Texto
    report_path = os.path.join(RESULTS_DIR, "best_model_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Relatório de Desempenho do Melhor Modelo no Conjunto de Teste\n")
        f.write("=" * 70 + "\n")
        f.write(f"Melhor Configuração KNN: {best_knn_params}\n")
        f.write(
            f"Número de Features Selecionadas: {len(selected_indices)} de {len(feature_names)}\n"
        )
        f.write(f"Fitness (CV no Treino): {best_config_row['best_fitness']:.6f}\n")
        f.write(f"Acurácia (CV no Treino): {best_config_row['mean_accuracy_cv']:.4f}\n")
        f.write("-" * 70 + "\n")

        if len(selected_indices) > 0:
            f.write("Relatório de Classificação Padrão (Teste):\n\n")
            f.write(report_str)
            f.write("\n" + "-" * 70 + "\n")
            f.write("Métricas de Avaliação Adicionais (Teste):\n\n")
            f.write(f"Acurácia Balanceada: {bal_acc:.4f}\n")
            f.write(f"Cohen's Kappa: {kappa:.4f}\n")
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
            f.write(f"Log Loss: {ll:.4f}\n")
        else:
            f.write(
                "Nenhuma feature selecionada, não foi possível gerar o relatório de teste.\n"
            )
    print(f"Salvo: Relatório de Classificação Detalhado em {report_path}")

    #  Matriz de Confusão
    if len(selected_indices) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f"Matriz de Confusão da Melhor Configuração KNN\nParâmetros: {best_knn_params}",
            fontsize=16,
        )
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp1.plot(ax=ax1, cmap="Blues", values_format="d")
        ax1.set_title("Contagens Absolutas", fontsize=14)
        disp2 = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=class_names, cmap="Greens", normalize="true"
        )
        disp2.plot(ax=ax2, values_format=".2%")
        ax2.set_title("Normalizada por Linha (Recall)", fontsize=14)
        plt.tight_layout(rect=(0, 0.03, 1, 0.94))
        plt.savefig(
            os.path.join(RESULTS_DIR, "confusion_matrix_best_config.png"), dpi=300
        )
        plt.close(fig)
        print("Salvo: Matriz de Confusão")

    if df_sorted_by_fitness.empty:
        print("DataFrame de resultados vazio. Nenhum gráfico será gerado.")
        return

    print("\nGerando relatórios e gráficos para publicação...")

    df = df_sorted_by_fitness.copy()
    best_config_row = df.iloc[0]
    best_knn_params = {
        key: best_config_row[key]
        for key in KNN_PARAM_GRID.keys()
        if key in best_config_row and pd.notna(best_config_row[key])
    }
    if 'leaf_size' in best_knn_params:
        best_knn_params['leaf_size'] = int(best_knn_params['leaf_size'])
    best_feature_vector = best_config_row["best_solution_vector"]
    selected_indices = np.where(best_feature_vector == 1)[0]
    class_names = [f"Classe {i}" for i in sorted(np.unique(y_train))]
    class_labels = sorted(np.unique(y_train))

    #  Avaliação do Melhor Modelo e Cálculo de Métricas
    y_pred = None
    y_prob = None
    report_str = ""
    cm = None
    bal_acc = None
    kappa = None
    mcc = None
    ll = None
    if len(selected_indices) > 0:
        final_knn = KNeighborsClassifier(**best_knn_params)
        final_knn.fit(X_train[:, selected_indices], y_train)
        y_pred = final_knn.predict(X_test[:, selected_indices])
        y_prob = final_knn.predict_proba(X_test[:, selected_indices])
        report_str = classification_report(y_test, y_pred, target_names=class_names)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        ll = log_loss(y_test, y_prob)

    #  Relatório de Texto
    report_path = os.path.join(RESULTS_DIR, "best_model_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Relatório de Desempenho do Melhor Modelo no Conjunto de Teste\n")
        f.write("=" * 70 + "\n")
        f.write(f"Melhor Configuração KNN: {best_knn_params}\n")
        f.write(
            f"Número de Features Selecionadas: {len(selected_indices)} de {len(feature_names)}\n"
        )
        f.write(f"Fitness (CV no Treino): {best_config_row['best_fitness']:.6f}\n")
        f.write(f"Acurácia (CV no Treino): {best_config_row['mean_accuracy_cv']:.4f}\n")
        f.write("-" * 70 + "\n")
        if len(selected_indices) > 0 and y_pred is not None:
            f.write(report_str)
            f.write(f"Acurácia Balanceada: {bal_acc:.4f}\n")
            f.write(f"Cohen's Kappa: {kappa:.4f}\n")
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
            f.write(f"Log Loss: {ll:.4f}\n")
        else:
            f.write("Nenhuma feature selecionada ou erro na predição.\n")
    print(f"Salvo: Relatório de Classificação Detalhado em {report_path}")

    #  Matriz de Confusão
    if len(selected_indices) > 0 and cm is not None and y_pred is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f"Matriz de Confusão da Melhor Configuração KNN\nParâmetros: {best_knn_params}",
            fontsize=16,
        )
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp1.plot(ax=ax1, cmap="Blues", values_format="d")
        ax1.set_title("Contagens Absolutas", fontsize=14)
        disp2 = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=class_names, cmap="Greens", normalize="true"
        )
        disp2.plot(ax=ax2, values_format=".2%")
        plt.tight_layout(rect=(0, 0.03, 1, 0.94))
        plt.savefig(
            os.path.join(RESULTS_DIR, "confusion_matrix_best_config.png"), dpi=300
        )
        plt.close(fig)
        print("Salvo: Matriz de Confusão")

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="execution_time_sec",
        y="best_fitness",
        hue="n_neighbors",
        style="metric",
        palette="magma",
        s=100,
    )
    plt.title("Fitness Final vs. Tempo de Execução da Otimização", fontsize=16)
    plt.xlabel("Tempo de Execução (segundos)", fontsize=12)
    plt.ylabel("Melhor Fitness Encontrado (Menor é Melhor)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Parâmetros", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scatter_fitness_vs_time.png"), dpi=300)
    plt.close()
    print("Salvo: Gráfico de Fitness vs Tempo de Execução")

    #  Gráfico de Dispersão com FACETAS para visualizar o parâmetro 'algorithm'
    g = sns.relplot(
        data=df,
        x="num_selected_features",
        y="mean_accuracy_cv",
        hue="n_neighbors",
        style="metric",
        size="weights",
        col="algorithm",
        col_wrap=2,
        height=6,
        aspect=1.2,
        palette="bright",
    )
    g.fig.suptitle(
        "Trade-off: Acurácia vs. Features (dividido por Algoritmo)", y=1.03, fontsize=16
    )
    g.set_xlabels("Número de Features Selecionadas", fontsize=12)
    g.set_ylabels("Acurácia Média (CV)", fontsize=12)
    plt.savefig(
        os.path.join(RESULTS_DIR, "scatter_accuracy_vs_features_faceted.png"), dpi=300
    )
    plt.close()
    print("Salvo: Gráfico de Trade-off com Facetas por Algoritmo")

    #  Curvas de Convergência
    plt.figure(figsize=(14, 9))
    top_n = len(df)
    for i in range(top_n):
        row = df.iloc[i]
        params_to_show = {
            k: row.get(k)
            for k in ["n_neighbors", "metric", "weights", "algorithm", "leaf_size", "p"]
            if pd.notna(row.get(k))
        }
        config_str = ", ".join([f"{k}={v}" for k, v in params_to_show.items()])
        label = f"Rank {i+1}: {config_str} (Fit: {row['best_fitness']:.4f})"
        plt.plot(row["convergence_curve"], label=label, linewidth=2.0, alpha=0.9)
    plt.title(
        f"Curvas de Convergência do BDA para as Top {top_n} Configurações", fontsize=16
    )
    plt.xlabel("Iteração do BDA", fontsize=12)
    plt.ylabel("Melhor Fitness Encontrado", fontsize=12)
    plt.legend(title="Configuração KNN (Melhores Resultados)", frameon=True, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_convergence_curves.png"), dpi=300)
    plt.close()
    print("Salvo: Curvas de Convergência")

    #  Frequência de Seleção de Features
    if not df["best_solution_vector"].empty:
        all_solutions = np.vstack(df["best_solution_vector"].values)
        feature_counts = np.sum(all_solutions, axis=0)
        feat_freq_df = pd.DataFrame(
            {"feature_name": feature_names, "count": feature_counts}
        ).sort_values("count", ascending=False)
        plt.figure(figsize=(20, 10))
        top_n_features = min(40, len(feature_names))
        sns.barplot(
            x="feature_name",
            y="count",
            data=feat_freq_df.head(top_n_features),
            palette="mako",
        )
        plt.title(
            f"Top {top_n_features} Features Mais Selecionadas em Todas as Execuções",
            fontsize=16,
        )
        plt.xlabel("Nome da Feature", fontsize=12)
        plt.ylabel("Contagem de Seleção", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, "barplot_feature_selection_frequency.png"),
            dpi=300,
        )
        plt.close()
        print("Salvo: Frequência de Seleção de Features")

    #  Boxplots de Fitness por Hiperparâmetro
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    fig.suptitle("Distribuição do Fitness por Hiperparâmetro do KNN", fontsize=18)
    sns.boxplot(ax=axes[0], data=df, x="n_neighbors", y="best_fitness", palette="crest")
    axes[0].set_title("por Número de Vizinhos (k)", fontsize=14)
    axes[0].set_xlabel("Número de Vizinhos (k)", fontsize=12)
    axes[0].set_ylabel("Melhor Fitness (Menor é Melhor)", fontsize=12)
    sns.boxplot(ax=axes[1], data=df, x="metric", y="best_fitness", palette="flare")
    axes[1].set_title("por Métrica de Distância", fontsize=14)
    axes[1].set_xlabel("Métrica", fontsize=12)
    axes[1].set_ylabel("")
    sns.boxplot(ax=axes[2], data=df, x="weights", y="best_fitness", palette="magma")
    axes[2].set_title("por tipo de Peso", fontsize=14)
    axes[2].set_xlabel("Peso", fontsize=12)
    axes[2].set_ylabel("")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, "boxplots_fitness_by_param.png"), dpi=300)
    plt.close()
    print("Salvo: Boxplots de Fitness")

    # Relação entre Tempo de Execução e Algoritmo
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df, x='algorithm', y='execution_time_sec', palette='rocket')
    sns.stripplot(data=df, x='algorithm', y='execution_time_sec', color=".25", alpha=0.6)
    plt.title('Distribuição do Tempo de Execução por Algoritmo', fontsize=16)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel('Tempo de Execução (segundos)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot_time_vs_algorithm.png"), dpi=300)
    plt.close()
    print("Salvo: Gráfico de Tempo de Execução vs. Algoritmo")

    # Swarm Plot para Acurácia vs. Hiperparâmetros
    try:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
        fig.suptitle("Distribuição da Acurácia por Hiperparâmetro (Swarm Plot)", fontsize=18)
        
        sns.swarmplot(ax=axes[0], data=df, x="n_neighbors", y="mean_accuracy_cv", palette="crest")
        axes[0].set_title("por Número de Vizinhos (k)", fontsize=14)
        axes[0].set_xlabel("Número de Vizinhos (k)", fontsize=12)
        axes[0].set_ylabel("Acurácia Média (CV)", fontsize=12)

        sns.swarmplot(ax=axes[1], data=df, x="algorithm", y="mean_accuracy_cv", palette="flare")
        axes[1].set_title("por Algoritmo", fontsize=14)
        axes[1].set_xlabel("Algoritmo", fontsize=12)
        axes[1].set_ylabel("")

        sns.swarmplot(ax=axes[2], data=df, x="weights", y="mean_accuracy_cv", palette="magma")
        axes[2].set_title("por Tipo de Peso", fontsize=14)
        axes[2].set_xlabel("Peso", fontsize=12)
        axes[2].set_ylabel("")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(RESULTS_DIR, "swarmplot_accuracy_by_param.png"), dpi=300)
        plt.close()
        print("Salvo: Swarm Plots de Acurácia")
    except Exception as e:
        print(f"Não foi possível gerar o Swarm Plot de Acurácia: {e}")

    # Violin Plot para Fitness vs. Hiperparâmetros
    try:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
        fig.suptitle("Distribuição do Fitness por Hiperparâmetro (Violin Plot)", fontsize=18)

        sns.violinplot(ax=axes[0], data=df, x="n_neighbors", y="best_fitness", palette="crest", inner="quartile")
        axes[0].set_title("por Número de Vizinhos (k)", fontsize=14)
        axes[0].set_xlabel("Número de Vizinhos (k)", fontsize=12)
        axes[0].set_ylabel("Melhor Fitness (Menor é Melhor)", fontsize=12)

        sns.violinplot(ax=axes[1], data=df, x="algorithm", y="best_fitness", palette="flare", inner="quartile")
        axes[1].set_title("por Algoritmo", fontsize=14)
        axes[1].set_xlabel("Algoritmo", fontsize=12)
        axes[1].set_ylabel("")

        sns.violinplot(ax=axes[2], data=df, x="weights", y="best_fitness", palette="magma", inner="quartile")
        axes[2].set_title("por Tipo de Peso", fontsize=14)
        axes[2].set_xlabel("Peso", fontsize=12)
        axes[2].set_ylabel("")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(RESULTS_DIR, "violinplot_fitness_by_param.png"), dpi=300)
        plt.close()
        print("Salvo: Violin Plots de Fitness")
    except Exception as e:
        print(f"Não foi possível gerar o Violin Plot de Fitness: {e}")
        
    # Gráfico de Coordenadas Paralelas
    try:
        plt.figure(figsize=(18, 10))
        df_parallel = df.copy()
        cols_to_plot = [
            'n_neighbors', 'algorithm', 'leaf_size',
            'mean_accuracy_cv', 'num_selected_features', 'best_fitness'
        ]
        df_parallel = df_parallel[cols_to_plot].dropna()
        
        for col in df_parallel.select_dtypes(include=['object']).columns:
            df_parallel[col] = df_parallel[col].astype('category').cat.codes
            
        for col in df_parallel.columns:
            if col != 'best_fitness':
                 df_parallel[col] = (df_parallel[col] - df_parallel[col].min()) / (df_parallel[col].max() - df_parallel[col].min())

        parallel_coordinates(
            df_parallel,
            class_column='best_fitness',
            colormap=plt.get_cmap("viridis_r"),
            linewidth=1.5,
            alpha=0.6
        )
        plt.title('Análise de Hiperparâmetros com Coordenadas Paralelas', fontsize=18)
        plt.xlabel('Hiperparâmetros e Métricas', fontsize=12)
        plt.ylabel('Valor Normalizado', fontsize=12)
        plt.xticks(rotation=15)
        # Adicionar uma colorbar para indicar a escala do fitness
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis_r"), norm=plt.Normalize(vmin=df['best_fitness'].min(), vmax=df['best_fitness'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Fitness (Menor é Melhor)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "parallel_coordinates_plot.png"), dpi=300)
        plt.close()
        print("Salvo: Gráfico de Coordenadas Paralelas")
    except Exception as e:
        print(f"Não foi possível gerar o Gráfico de Coordenadas Paralelas: {e}")

    # Heatmap de Correlação Numérica
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_to_exclude = ['config_id']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": 10},
            vmin=-1, vmax=1
        )
        plt.title("Heatmap de Correlação entre Métricas e Hiperparâmetros Numéricos", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "heatmap_numeric_correlation.png"), dpi=300)
        plt.close()
        print("Salvo: Heatmap de Correlação Numérica")
    except Exception as e:
        print(f"Não foi possível gerar o Heatmap de Correlação: {e}")

    # Gráfico de Densidade 2D (KDE) - Acurácia vs. Nº de Features
    try:
        plt.figure(figsize=(12, 9))
        sns.kdeplot(
            data=df,
            x="num_selected_features",
            y="mean_accuracy_cv",
            fill=True,
            thresh=0.05,
            levels=10,
            cmap="mako_r"
        )
        sns.scatterplot(
            data=df.iloc[[0]],
            x="num_selected_features",
            y="mean_accuracy_cv",
            marker='*',
            color='red',
            s=300,
            label=f'Melhor Solução (Fit: {df.iloc[0]["best_fitness"]:.4f})',
            edgecolor='black'
        )
        plt.title("Densidade da Relação Acurácia vs. Nº de Features", fontsize=16)
        plt.xlabel("Número de Features Selecionadas", fontsize=12)
        plt.ylabel("Acurácia Média na Validação Cruzada", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "kdeplot_accuracy_vs_features.png"), dpi=300)
        plt.close()
        print("Salvo: Gráfico de Densidade 2D (Acurácia vs. Features)")
    except Exception as e:
        print(f"Não foi possível gerar o Gráfico de Densidade 2D: {e}")


    print("\nTodos os relatórios e gráficos foram gerados com sucesso.")


if __name__ == "__main__":
    total_start_time = time.time()
    run_optimization_pipeline()
    total_end_time = time.time()
    print(
        f"\nProcesso de otimização completo. Tempo total: {(total_end_time - total_start_time) / 60:.2f} minutos."
    )
