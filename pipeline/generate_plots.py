# -*- coding: utf-8 -*-
"""
Script para gerar gráficos baseados em resultados salvos dos pipelines.

Este script carrega os resultados salvos em JSON dos pipelines BDA-DNN e RHCB5
e gera gráficos que são possíveis sem re-executar os pipelines.

Gráficos gerados:
- Boxplots de comparação de métricas
- Matrizes de confusão agregadas
- Métricas por run
- Matrizes de confusão por run
- Delta de acurácia por run
- Heatmap de desvio padrão da matriz de confusão
- Curvas ROC
- Curvas Precision-Recall
- Testes de significância estatística
- Distribuições de performance
- Scatter de performance vs custo
- Análise de sensibilidade (para BDA-DNN)

Nota: Gráficos que requerem dados brutos (sinais, features) não podem ser gerados
sem salvar esses dados adicionalmente nos resultados.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
plt.ioff()
matplotlib.interactive(False)
plt.switch_backend('Agg')
matplotlib.use('Agg', force=True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("husl")

# Importar utilitários
from pipeline_utils import Plotting, NumpyEncoder, CLASS_NAMES

def load_results_from_dir(base_results_dir, pipeline_name):
    """
    Carrega todos os run_results.json de um diretório de pipeline.
    """
    results_list = []
    pipeline_dir = os.path.join(base_results_dir, pipeline_name)
    if not os.path.exists(pipeline_dir):
        print(f"Diretório {pipeline_dir} não encontrado. Pulando {pipeline_name}.")
        return results_list

    for run_dir in sorted(os.listdir(pipeline_dir)):
        run_path = os.path.join(pipeline_dir, run_dir)
        if os.path.isdir(run_path):
            results_file = os.path.join(run_path, "run_results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                        results_list.append(result)
                except Exception as e:
                    print(f"Erro ao carregar {results_file}: {e}")
            else:
                print(f"Arquivo run_results.json não encontrado em {run_path}")

    return results_list

def main():
    # Configurações
    RESULTS_BASE_DIR = "./results"
    
    # Encontrar o diretório de comparação mais recente
    if os.path.exists(RESULTS_BASE_DIR):
        comparison_dirs = [d for d in os.listdir(RESULTS_BASE_DIR) if d.startswith("comparison_run_")]
        if comparison_dirs:
            latest_comparison = max(comparison_dirs, key=lambda x: os.path.getmtime(os.path.join(RESULTS_BASE_DIR, x)))
            BASE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, latest_comparison)
            print(f"Usando diretório de resultados: {BASE_RESULTS_DIR}")
        else:
            print("Nenhum diretório de comparação encontrado em ../results.")
            return
    else:
        print("Diretório ../results não encontrado.")
        return
    
    PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, "generated_plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    SAVE_PLOTS = True  # Sempre salvar plots

    print("Carregando resultados salvos...")

    # Carregar resultados
    bda_results = load_results_from_dir(BASE_RESULTS_DIR, "BDA_DNN_runs")
    rhcb5_results = load_results_from_dir(BASE_RESULTS_DIR, "RHCB5_runs")

    if not bda_results and not rhcb5_results:
        print("Nenhum resultado encontrado. Certifique-se de que os pipelines foram executados e salvaram os resultados.")
        return

    results_dict = {}
    if bda_results:
        results_dict["BDA_DNN"] = bda_results
    if rhcb5_results:
        results_dict["RHCB5"] = rhcb5_results

    print(f"Resultados carregados: BDA_DNN ({len(bda_results)} runs), RHCB5 ({len(rhcb5_results)} runs)")

    # Gerar gráficos que são possíveis com os dados salvos

    # 1. Boxplots de comparação
    print("Gerando boxplots de comparação...")
    Plotting.plot_comparison_boxplots(results_dict, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 2. Matrizes de confusão agregadas
    for pipeline_name, results_list in results_dict.items():
        print(f"Gerando matriz de confusão agregada para {pipeline_name}...")
        Plotting.plot_aggregated_confusion_matrix(results_list, pipeline_name, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 3. Métricas por run
    for pipeline_name, results_list in results_dict.items():
        print(f"Gerando métricas por run para {pipeline_name}...")
        Plotting.plot_per_run_metrics(results_list, pipeline_name, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 4. Matrizes de confusão por run
    for pipeline_name, results_list in results_dict.items():
        print(f"Gerando matrizes de confusão por run para {pipeline_name}...")
        Plotting.plot_per_run_confusion_matrices(results_list, pipeline_name, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 5. Delta de acurácia por run
    print("Gerando delta de acurácia por run...")
    Plotting.plot_delta_accuracy_per_run(results_dict, PLOTS_DIR, SAVE_PLOTS)

    # 6. Heatmap de desvio padrão da matriz de confusão
    for pipeline_name, results_list in results_dict.items():
        print(f"Gerando heatmap de desvio padrão da matriz de confusão para {pipeline_name}...")
        Plotting.plot_confusion_matrix_std_heatmap(results_list, pipeline_name, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 7. Curvas ROC
    print("Gerando curvas ROC...")
    Plotting.plot_roc_curves(results_dict, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 8. Curvas Precision-Recall
    print("Gerando curvas Precision-Recall...")
    Plotting.plot_precision_recall_curves(results_dict, CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS)

    # 9. Testes de significância estatística
    print("Gerando testes de significância estatística...")
    Plotting.plot_statistical_significance(results_dict, PLOTS_DIR, SAVE_PLOTS)

    # 10. Plots de distribuição
    print("Gerando plots de distribuição...")
    Plotting.plot_distribution_plots(results_dict, PLOTS_DIR, SAVE_PLOTS)

    # 11. Scatter de performance vs custo
    print("Gerando scatter de performance vs custo...")
    Plotting.plot_performance_vs_cost_scatter(results_dict, PLOTS_DIR, SAVE_PLOTS)

    # 12. Análise de sensibilidade (apenas para BDA-DNN)
    if "BDA_DNN" in results_dict:
        print("Gerando análise de sensibilidade para BDA-DNN...")
        Plotting.plot_sensitivity_analysis(results_dict["BDA_DNN"], PLOTS_DIR, SAVE_PLOTS)

    # 13. Frequência de seleção de features (apenas para BDA-DNN, se feature_names estiver salvo)
    if "BDA_DNN" in results_dict and results_dict["BDA_DNN"] and "feature_names" in results_dict["BDA_DNN"][0]:
        feature_names = results_dict["BDA_DNN"][0]["feature_names"]
        print("Gerando frequência de seleção de features para BDA-DNN...")
        Plotting.plot_feature_selection_frequency(results_dict["BDA_DNN"], feature_names, PLOTS_DIR, SAVE_PLOTS)

    print(f"Gráficos gerados e salvos em: {PLOTS_DIR}")

if __name__ == "__main__":
    main()