# -*- coding: utf-8 -*-
"""
Orquestrador Principal de Comparação de Pipelines

Este script executa a tarefa principal:
1. Define o número de execuções (NUM_RUNS = 50).
2. Gera 50 seeds aleatórias únicas.
3. Executa o pipeline BDA-DNN 50 vezes, uma para cada seed.
4. Executa o pipeline RHCB5 50 vezes, usando AS MESMAS seeds da etapa 3.
5. Coleta os resultados de todas as 100 execuções.
6. Compila estatísticas robustas (média, mediana, std, etc.) para cada pipeline.
7. Gera gráficos comparativos (boxplots, matrizes de confusão agregadas).
8. Salva todos os resultados em um diretório de comparação.
"""

import os
import sys
import time
import datetime
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import gc
from sklearn.model_selection import train_test_split

# --- Importação dos Módulos de Pipeline ---
try:
    from pipeline_utils import Plotting, NumpyEncoder, CLASS_NAMES, DataHandler, FS, HIGHCUT_HZ, FILTER_ORDER, TEST_SIZE, VAL_SIZE
    from pipeline_bda_dnn import run_bda_dnn_pipeline, FeatureExtractor
    from pipeline_rhcb5 import run_rhcb5_pipeline
except ImportError as e:
    print(f"ERRO: Não foi possível importar os módulos dos pipelines: {e}")
    print("Certifique-se que 'pipeline_utils.py', 'pipeline_bda_dnn.py', e 'pipeline_rhcb5.py' estão no mesmo diretório.")
    sys.exit(1)

# --- Configurações da Comparação ---
NUM_RUNS = 50 # Número de vezes para executar cada pipeline
SAVE_PLOTS = True # Salvar plots agregados

# --- Configuração de Diretórios ---
# Assume que 'data' está no diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(current_dir, "..", "data")
BASE_RESULTS_DIR = os.path.join(current_dir, "results")

# Passa as constantes globais para as funções do pipeline
GLOBAL_CONSTANTS = {
    "BASE_DATA_DIR": BASE_DATA_DIR
}

# --- Funções Auxiliares de Análise ---

def compile_and_save_statistics(results_list, pipeline_name, output_dir):
    """
    Compila uma lista de dicionários de resultados em um DataFrame
    e calcula estatísticas robustas.

    Algumas colunas do DataFrame são específicas de cada pipeline e podem conter valores NaN para outros pipelines.
    Por exemplo, 'num_features', 'bda_fitness', e 'bda_time_sec' são relevantes apenas para o pipeline BDA-DNN e serão NaN para RHCB5.
    """
    print(f"\n--- Compilando Estatísticas para {pipeline_name} ---")
    
    # Extrai os dados relevantes
    data_for_df = []
    for run_res in results_list:
        if not isinstance(run_res, dict) or "final_metrics" not in run_res:
            run_id = run_res.get('run_id') if isinstance(run_res, dict) else None
            print(f"Aviso: Run {run_id} de {pipeline_name} falhou ou está incompleta. Descartando.")
            continue
            
        metrics = run_res["final_metrics"]
        report = metrics.get("classification_report", {})
        specificities = metrics.get("specificities", {})
        
        entry = {
            "run_id": run_res["run_id"],
            "seed": run_res["seed"],
            "accuracy": metrics.get("accuracy"),
            "f1_macro": report.get("macro avg", {}).get("f1-score"),
            "exec_time_sec": run_res.get("execution_time_sec"),
            "num_features": run_res.get("num_selected_features"), # Será NaN para RHCB5
            "bda_fitness": run_res.get("bda_best_fitness"), # Será NaN para RHCB5
            "bda_time_sec": run_res.get("bda_optimization_time_sec"), # NaN para RHCB5
            "dnn_time_sec": run_res.get("dnn_train_eval_time_sec"),
        }
        
        # Adiciona métricas por classe
        for i, name in enumerate(CLASS_NAMES):
            report_key = name
            spec_key = f"specificity_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
            entry[f"recall_{i}"] = report.get(report_key, {}).get("recall")
            entry[f"precision_{i}"] = report.get(report_key, {}).get("precision")
            entry[f"f1_score_{i}"] = report.get(report_key, {}).get("f1-score")
            entry[f"specificity_{i}"] = specificities.get(spec_key)

        data_for_df.append(entry)

    if not data_for_df:
        print(f"ERRO: Nenhum dado válido encontrado para {pipeline_name}.")
        return None, None
        
    df = pd.DataFrame(data_for_df)
    
    # Salva o DataFrame completo com todas as execuções
    csv_path_full = os.path.join(output_dir, f"stats_{pipeline_name}_full_runs.csv")
    df.to_csv(csv_path_full, index=False, float_format="%.6f")
    print(f"Resultados completos das {len(df)} execuções salvos em: {csv_path_full}")
    
    # Calcula estatísticas descritivas
    # Descarte de outliers (ex: usando IQR)
    # Para simplicidade acadêmica, 'describe' é robusto
    stats_df = df.describe(percentiles=[.25, .5, .75]).transpose()
    
    # Adiciona Mediana explicitamente
    stats_df['median'] = df.median(numeric_only=True)
    # Adiciona IQR
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
    # Adiciona Skewness (Assimetria) e Kurtosis (Curtose)
    stats_df['skew'] = df.skew(numeric_only=True)
    stats_df['kurtosis'] = df.kurtosis(numeric_only=True)
    
    stats_df = stats_df.sort_index()
    
    # Salva as estatísticas
    csv_path_stats = os.path.join(output_dir, f"stats_{pipeline_name}_summary.csv")
    stats_df.to_csv(csv_path_stats, float_format="%.6f")
    print(f"Resumo estatístico salvo em: {csv_path_stats}")
    
    print("\nResumo (Média e Mediana):")
    print(stats_df[['mean', 'median', 'std']])
    
    return df, stats_df

def run_pipeline_loop(pipeline_func, pipeline_name, run_seeds, base_results_dir):
    """Função auxiliar para executar o loop de 50 execuções."""
    all_results = []
    pipeline_run_dir = os.path.join(COMPARISON_RUN_DIR, f"{pipeline_name}_runs")
    os.makedirs(pipeline_run_dir, exist_ok=True)
    
    start_time_pipeline = time.time()
    
    for i in range(NUM_RUNS):
        run_id = i + 1
        seed = run_seeds[i]
        
        print("\n" + "="*80)
        print(f"--- Iniciando {pipeline_name} Run {run_id}/{NUM_RUNS} (Seed: {seed}) ---")
        print("="*80)
        
        try:
            result = pipeline_func(
                run_id=run_id,
                base_results_dir=pipeline_run_dir,
                global_constants=GLOBAL_CONSTANTS,
                random_seed_for_run=seed
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERRO CATASTRÓFICO na Run {run_id} de {pipeline_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"run_id": run_id, "seed": seed, "error": str(e)})
        
        # Limpa a memória da GPU (se aplicável) após cada execução
        tf.keras.backend.clear_session()
        gc.collect()

    total_time_min = (time.time() - start_time_pipeline) / 60.0
    print("\n" + "="*80)
    print(f"Loop de {NUM_RUNS} execuções de {pipeline_name} concluído.")
    print(f"Tempo total para {pipeline_name}: {total_time_min:.2f} minutos.")
    print("="*80 + "\n")
    
    return all_results

# --- Script Principal ---
def main():
    # Cria um diretório único para esta comparação
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    COMPARISON_RUN_DIR = os.path.join(BASE_RESULTS_DIR, f"comparison_run_{run_timestamp}")
    COMPARISON_PLOTS_DIR = os.path.join(COMPARISON_RUN_DIR, "plots")
    os.makedirs(COMPARISON_PLOTS_DIR, exist_ok=True)

    print(f"Iniciando Comparação de Pipelines (BDA-DNN vs RHCB5)")
    print(f"Número de Execuções por Pipeline: {NUM_RUNS}")
    print(f"Diretório de Resultados da Comparação: {COMPARISON_RUN_DIR}")
    print(f"Diretório de Dados: {BASE_DATA_DIR}")

    start_time_main = time.time()

    # 1. Gerar Seeds
    # Gera N seeds aleatórias únicas para garantir reprodutibilidade e
    # que cada pipeline use a mesma divisão de dados para a mesma "run_id".
    print(f"Gerando {NUM_RUNS} seeds aleatórias...")
    # Usa um gerador aleatório separado para as seeds
    seed_generator = np.random.RandomState(42) 
    run_seeds = [seed_generator.randint(0, 100000) for _ in range(NUM_RUNS)]

    # 2. Carregar dados uma vez (para estatísticas)
    print("Carregando dados uma vez...")
    try:
        raw_data, raw_labels = DataHandler.load_bonn_data(BASE_DATA_DIR)
    except Exception as e:
        print(f"ERRO ao carregar os dados: {e}")
        print("Certifique-se que o diretório de dados está correto e os arquivos necessários estão presentes.")
        sys.exit(1)

    # 3. Executar BDA-DNN
    bda_all_results = run_pipeline_loop(
        pipeline_func=run_bda_dnn_pipeline,
        pipeline_name="BDA_DNN",
        run_seeds=run_seeds,
        base_results_dir=COMPARISON_RUN_DIR
    )

    # 4. Executar RHCB5
    rhcb5_all_results = run_pipeline_loop(
        pipeline_func=run_rhcb5_pipeline,
        pipeline_name="RHCB5",
        run_seeds=run_seeds,
        base_results_dir=COMPARISON_RUN_DIR
    )

    # 4. Salvar resultados brutos
    print("\nSalvando resultados brutos de todas as execuções...")
    all_raw_results = {
        "bda_dnn_results": bda_all_results,
        "rhcb5_results": rhcb5_all_results
    }
    raw_results_path = os.path.join(COMPARISON_RUN_DIR, "all_raw_results.json")
    try:
        with open(raw_results_path, "w") as f:
            json.dump(all_raw_results, f, cls=NumpyEncoder, indent=4)
        print(f"Resultados brutos salvos em: {raw_results_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados brutos: {e}")

    # 5. Compilar Estatísticas
    bda_df, bda_stats = compile_and_save_statistics(
        bda_all_results, "BDA_DNN", COMPARISON_RUN_DIR
    )
    rhcb5_df, rhcb5_stats = compile_and_save_statistics(
        rhcb5_all_results, "RHCB5", COMPARISON_RUN_DIR
    )

    # 6. Análise Estatística de Comparação (Ex: Teste T)
    print("\n--- Análise Estatística Comparativa (Teste T) ---")
    if (bda_df is not None) and (rhcb5_df is not None):
        try:
            # Compara Acurácias
            acc_ttest = stats.ttest_ind(
                bda_df['accuracy'].dropna(), 
                rhcb5_df['accuracy'].dropna(), 
                equal_var=False # Welch's T-test
            )
            print(f"Teste T (Acurácia) | p-value: {acc_ttest.pvalue:.4g}")
            if acc_ttest.pvalue < 0.05:
                print("  > Diferença estatisticamente significativa na Acurácia.")
            else:
                print("  > Nenhuma diferença estatisticamente significativa na Acurácia.")

            # Compara F1-Macro
            f1_ttest = stats.ttest_ind(
                bda_df['f1_macro'].dropna(), 
                rhcb5_df['f1_macro'].dropna(), 
                equal_var=False
            )
            print(f"Teste T (F1-Macro) | p-value: {f1_ttest.pvalue:.4g}")
            if f1_ttest.pvalue < 0.05:
                print("  > Diferença estatisticamente significativa no F1-Macro.")
            else:
                print("  > Nenhuma diferença estatisticamente significativa no F1-Macro.")
        
        except Exception as e:
            print(f"Erro ao executar Teste T: {e}")

    # 7. Gerar Gráficos Comparativos Agregados
    print("\n--- Gerando Gráficos Comparativos Finais ---")
    results_dict_list_for_plotting = {
        "BDA_DNN": bda_all_results,
        "RHCB5": rhcb5_all_results
    }
    
    # Boxplots (função nova em pipeline_utils.py)
    Plotting.plot_comparison_boxplots(
        results_dict_list_for_plotting,
        CLASS_NAMES,
        COMPARISON_PLOTS_DIR,
        SAVE_PLOTS
    )
    
    # Matrizes de Confusão Agregadas
    Plotting.plot_aggregated_confusion_matrix(
        bda_all_results, "BDA_DNN", CLASS_NAMES,
        COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    Plotting.plot_aggregated_confusion_matrix(
        rhcb5_all_results, "RHCB5", CLASS_NAMES,
        COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # New aggregated plots
    Plotting.plot_statistical_significance(
        results_dict_list_for_plotting, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Per-run plots for each pipeline
    Plotting.plot_per_run_metrics(
        bda_all_results, "BDA_DNN", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    Plotting.plot_per_run_metrics(
        rhcb5_all_results, "RHCB5", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Per-run confusion matrices (will create many files)
    Plotting.plot_per_run_confusion_matrices(
        bda_all_results, "BDA_DNN", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    Plotting.plot_per_run_confusion_matrices(
        rhcb5_all_results, "RHCB5", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    total_main_time = time.time() - start_time_main
    print("\n" + "="*80)
    print("COMPARAÇÃO COMPLETA CONCLUÍDA")
    print(f"Tempo total da orquestração: {total_main_time/60:.2f} minutos.")
    print(f"Resultados, estatísticas e gráficos salvos em: {COMPARISON_RUN_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
