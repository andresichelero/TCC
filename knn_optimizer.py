# optimizer.py
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
    jaccard_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss,
)

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

#  CONFIGURAÇÕES DA OTIMIZAÇÃO

KNN_PARAM_GRID = {
    "n_neighbors": [3, 4, 5, 6, 7, 8],
    "metric": [
        "manhattan"
    ],
    "weights": ["distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "leaf_size": [10, 30, 50],
}

#  Configurações do BDA e do Dataset
RANDOM_SEED = 42
N_AGENTS_BDA = 30
MAX_ITER_BDA = 50
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
    Cria uma função de fitness que encapsula uma configuração específica do KNN.
    """

    def evaluate_fitness_configured(binary_feature_vector, *args, **kwargs):
        """
        Função de fitness que será chamada pelo BDA.
        """
        selected_indices = np.where(binary_feature_vector == 1)[0]
        num_selected = len(selected_indices)
        total_features = len(binary_feature_vector)

        if num_selected == 0:
            return {
                "fitness": 1.0,
                "accuracy": 0.0,
                "num_features": 0,
            }

        X_train_selected = X_train_features[:, selected_indices]
        knn = KNeighborsClassifier(**knn_config)
        n_folds = 10
        min_samples_per_class = np.min(np.bincount(y_train_labels))
        if min_samples_per_class < n_folds:
            n_folds = max(2, min_samples_per_class)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

        try:
            accuracies = cross_val_score(
                knn,
                X_train_selected,
                y_train_labels,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
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
    Executa todo o pipeline de otimização.
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

    print(f"Total de {all_features.shape[1]} features extraídas.")
    print(f"Dados de treino: {X_train_feat.shape}, Dados de teste: {X_test_feat.shape}")

    knn_configurations = generate_knn_configs(KNN_PARAM_GRID)
    total_runs = len(knn_configurations)

    print(
        f"\n 3. Iniciando Otimização em Lote ({total_runs} configurações de KNN) "
    )

    all_results = Parallel(n_jobs=10, verbose=11)(delayed(process_single_config)(i, config, X_train_feat, y_train, all_features.shape[1]) for i, config in enumerate(knn_configurations))

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="best_fitness", ascending=True)

    df_to_save = results_df.drop(columns=["best_solution_vector", "convergence_curve"])
    df_to_save.to_csv(
        os.path.join(RESULTS_DIR, "optimization_summary_results.csv"), index=False
    )
    print("\n\n 4. Análise e Visualização dos Resultados ")
    print("Top 5 melhores configurações encontradas (ordenadas por fitness):")
    print(df_to_save.head().to_markdown(index=False))

    results_df_for_similarity = results_df.sort_index()

    generate_reports_and_plots(
        results_df,
        results_df_for_similarity,
        X_train_feat,
        y_train,
        X_test_feat,
        y_test,
        feature_names,
    )


def process_single_config(config_id, config, X_train_feat, y_train, feature_dim):
    """
    Executa o pipeline de otimização para uma única configuração de KNN.
    Esta função será chamada em paralelo.
    """
    start_run_time = time.time()
    fitness_function_for_bda = create_fitness_function(config, X_train_feat, y_train)

    bda = BinaryDragonflyAlgorithm(
        N=N_AGENTS_BDA,
        T=MAX_ITER_BDA,
        dim=feature_dim,
        fitness_func=fitness_function_for_bda,
        X_train_feat=X_train_feat,
        y_train=y_train,
        seed=RANDOM_SEED + config_id,
        verbose_optimizer_level=0,
    )

    best_solution, best_fitness, convergence_curve, _, _, _ = bda.run()

    mean_accuracy, num_features, _ = get_full_metrics(
        fitness_function_for_bda, best_solution, X_train_feat, y_train
    )
    elapsed_time = time.time() - start_run_time
    print(f"Config {config_id} finalizou em {elapsed_time:.2f}s. Accuracy: {mean_accuracy:.4f}, Features: {num_features}, Fitness: {best_fitness:.4f}")
    return {
        "config_id": config_id,
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
        j_scores_per_class = jaccard_score(
            y_test, y_pred, average=None, labels=class_labels
        )
        j_score_macro = jaccard_score(y_test, y_pred, average="macro")

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
            f.write(f"Jaccard Score (Macro Avg): {j_score_macro:.4f}\n")
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.savefig(
            os.path.join(RESULTS_DIR, "confusion_matrix_best_config.png"), dpi=300
        )
        plt.close(fig)
        print("Salvo: Matriz de Confusão")

    #  Gráfico de Jaccard Score
    if len(selected_indices) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x=class_names, y=j_scores_per_class, palette="plasma", ax=ax)
        ax.axhline(
            y=j_score_macro,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Média Macro: {j_score_macro:.3f}",
        )
        ax.set_title(f"Jaccard Score por Classe (Melhor Configuração)", fontsize=16)
        ax.set_xlabel("Classe", fontsize=12)
        ax.set_ylabel("Jaccard Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend()
        for i, score in enumerate(j_scores_per_class):
            ax.text(
                i, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=11
            )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "jaccard_score_best_config.png"), dpi=300)
        plt.close(fig)
        print("Salvo: Gráfico de Jaccard Score")

    #  Heatmap de Similaridade Jaccard entre Soluções
    try:
        if not df_sorted_by_id["best_solution_vector"].empty:
            solutions = np.vstack(df_sorted_by_id["best_solution_vector"].values)
            num_solutions = len(solutions)
            jaccard_matrix = np.zeros((num_solutions, num_solutions))

            for i in range(num_solutions):
                for j in range(i, num_solutions):
                    score = jaccard_score(solutions[i], solutions[j])
                    jaccard_matrix[i, j] = score
                    jaccard_matrix[j, i] = score

            plt.figure(figsize=(16, 14))
            sns.heatmap(jaccard_matrix, cmap="coolwarm", vmin=0, vmax=1)
            plt.title(
                "Similaridade de Jaccard Entre as Melhores Soluções de Cada Execução",
                fontsize=16,
            )
            plt.xlabel("ID da Configuração", fontsize=12)
            plt.ylabel("ID da Configuração", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                os.path.join(RESULTS_DIR, "heatmap_jaccard_similarity_solutions.png"),
                dpi=300,
            )
            plt.close()
            print("Salvo: Heatmap de Similaridade Jaccard entre Soluções")
    except Exception as e:
        print(f"Não foi possível gerar o heatmap de similaridade Jaccard: {e}")

    #  Heatmap de Acurácia vs Hiperparâmetros
    try:
        df_for_heatmap = df.copy()
        df_for_heatmap["p_str"] = df_for_heatmap.get(
            "p", pd.Series(index=df_for_heatmap.index, dtype=str)
        ).fillna("")
        df_for_heatmap["config_label"] = (
            df_for_heatmap["metric"]
            + ", "
            + df_for_heatmap["weights"]
            + np.where(
                df_for_heatmap["metric"] == "minkowski",
                ", p="
                + df_for_heatmap["p_str"]
                .astype(str)
                .str.replace("\.0", "", regex=True),
                "",
            )
        )
        heatmap_data = df_for_heatmap.pivot_table(
            values="mean_accuracy_cv", index="config_label", columns="n_neighbors"
        )
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            linewidths=0.5,
            annot_kws={"size": 10},
        )
        plt.title("Acurácia Média na Validação Cruzada por Hiperparâmetro", fontsize=16)
        plt.xlabel("Número de Vizinhos (k)", fontsize=12)
        plt.ylabel("Configuração (Métrica, Peso, p)", fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, "heatmap_accuracy_vs_params.png"), dpi=300
        )
        plt.close()
        print("Salvo: Heatmap de Acurácia")
    except Exception as e:
        print(f"Não foi possível gerar o heatmap de acurácia: {e}")

    #  Pair Plot das Métricas de Resultado
    try:
        pair_df = df[
            [
                "mean_accuracy_cv",
                "best_fitness",
                "num_selected_features",
                "execution_time_sec",
                "metric",
            ]
        ].rename(
            columns={
                "mean_accuracy_cv": "Acurácia (CV)",
                "best_fitness": "Fitness",
                "num_selected_features": "Qtd. Features",
                "execution_time_sec": "Tempo (s)",
            }
        )
        g = sns.pairplot(pair_df, hue="metric", palette="plasma", corner=True)
        g.fig.suptitle(
            "Visão Geral das Relações Entre Métricas de Resultado", y=1.02, fontsize=16
        )
        plt.savefig(os.path.join(RESULTS_DIR, "pairplot_results_overview.png"), dpi=300)
        plt.close()
        print("Salvo: Pair Plot de Resultados")
    except Exception as e:
        print(f"Não foi possível gerar o pair plot: {e}")

    #  Gráfico de Dispersão Trade-off: Acurácia vs. N° de Features
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="num_selected_features",
        y="mean_accuracy_cv",
        hue="best_fitness",
        size="n_neighbors",
        style="metric",
        palette="viridis_r",
        sizes=(40, 200),
        alpha=0.8,
    )
    plt.title("Trade-off: Acurácia vs. Número de Features Selecionadas", fontsize=16)
    plt.xlabel("Número de Features Selecionadas", fontsize=12)
    plt.ylabel("Acurácia Média na Validação Cruzada", fontsize=12)
    plt.legend(
        title="Legenda", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "scatter_accuracy_vs_features_tradeoff.png"), dpi=300
    )
    plt.close()
    print("Salvo: Gráfico de Trade-off Acurácia vs Features")

    #  Gráfico de Fitness vs. Tempo de Execução
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
    top_n = min(5, len(df))
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

    # Relação entre Acurácia, Algoritmo e Leaf Size
    df_leaf = df.dropna(subset=['leaf_size']).copy()
    if not df_leaf.empty:
        df_leaf['leaf_size'] = df_leaf['leaf_size'].astype(int)
        g = sns.catplot(
            data=df_leaf, x='algorithm', y='mean_accuracy_cv', hue='leaf_size',
            kind='bar', palette='magma', height=6, aspect=1.5, legend_out=False
        )
        g.fig.suptitle('Acurácia Média (CV) vs. Algoritmo e Tamanho da Folha (Leaf Size)', y=1.03, fontsize=16)
        g.set_axis_labels('Algoritmo', 'Acurácia Média (CV)', fontsize=12)
        plt.legend(title='Leaf Size', loc='upper right')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(RESULTS_DIR, "barplot_accuracy_vs_algo_leafsize.png"), dpi=300)
        plt.close()
        print("Salvo: Gráfico de Acurácia vs. Algoritmo e Leaf Size")
    
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
