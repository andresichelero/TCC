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

# --- Configuração de Hardware ---
# Para alterar entre GPU e CPU, edite USE_GPU em pipeline_utils.py
# USE_GPU = True  # Usa GPU se disponível ou força uso de CPU
# Para habilitar/desabilitar XAI/SHAP, edite USE_XAI em pipeline_utils.py
# USE_XAI = True  # Executa análise XAI/SHAP no melhor modelo RHCB5

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

# --- Importação dos Módulos de Pipeline ---
try:
    # Imports from pipeline_utils:
    # - Plotting: used for all plotting functions at the end of main()
    # - NumpyEncoder: used for saving results as JSON
    # - CLASS_NAMES: used for metrics and plotting
    # - DataHandler: used for loading and preprocessing data
    # - FS, HIGHCUT_HZ, FILTER_ORDER: used for EEG preprocessing
    # - USE_XAI: controls whether XAI/SHAP analysis is performed
    from pipeline_utils import Plotting, NumpyEncoder, CLASS_NAMES, DataHandler, FS, HIGHCUT_HZ, FILTER_ORDER, USE_XAI, USE_GPU
    from pipeline_bda_dnn import run_bda_dnn_pipeline
    from pipeline_rhcb5 import run_rhcb5_pipeline
except ImportError as e:
    print(f"ERRO: Não foi possível importar os módulos dos pipelines: {e}")
    print("Certifique-se que 'pipeline_utils.py', 'pipeline_bda_dnn.py', e 'pipeline_rhcb5.py' estão no mesmo diretório.")
    sys.exit(1)

# --- Configurações da Comparação ---
NUM_RUNS = 30 # Número de vezes para executar cada pipeline
SAVE_PLOTS = True # Salvar plots agregados

# --- Configuração de Diretórios ---
# Assume que 'data' está no diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(current_dir, "..", "data", "Bonn")
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

def run_pipeline_loop(pipeline_func, pipeline_name, run_seeds, base_results_dir, **data_kwargs):
    """Função auxiliar para executar o loop de 50 execuções."""
    all_results = []
    pipeline_run_dir = os.path.join(base_results_dir, f"{pipeline_name}_runs")
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
                random_seed_for_run=seed,
                **data_kwargs
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"run_id": run_id, "seed": seed, "error": str(e)})
        
        # Limpa a memória da GPU (se aplicável) após cada execução
        if hasattr(tf, "config") and (getattr(tf.config, "list_physical_devices", None) is not None):
            # Só limpa sessão se TensorFlow está disponível e há GPU ou uso explícito de TF
            if USE_GPU:
                tf.keras.backend.clear_session() # pyright: ignore[reportAttributeAccessIssue]
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

    # 3. Pré-processar dados uma vez
    print("\n--- 2. Pré-processando Dados (Utils) ---")
    data_processed = DataHandler.preprocess_eeg(
        raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
    )

    # 4. Extrair características SWT uma vez (para BDA-DNN)
    print("\n--- 3. Extraindo Características SWT ---")
    from pipeline_bda_dnn import FeatureExtractor
    X_full_feat, feature_names = FeatureExtractor.extract_swt_features(data_processed, wavelet='db4', level=4)

    # 5. Executar BDA-DNN
    bda_all_results = run_pipeline_loop(
        pipeline_func=run_bda_dnn_pipeline,
        pipeline_name="BDA_DNN",
        run_seeds=run_seeds,
        base_results_dir=COMPARISON_RUN_DIR,
        X_full_feat=X_full_feat,
        feature_names=feature_names,
        raw_labels=raw_labels
    )

    # 6. Executar RHCB5
    rhcb5_all_results = run_pipeline_loop(
        pipeline_func=run_rhcb5_pipeline,
        pipeline_name="RHCB5",
        run_seeds=run_seeds,
        base_results_dir=COMPARISON_RUN_DIR,
        data_processed=data_processed,
        raw_labels=raw_labels,
        run_xai=False  # Always False for all runs, XAI only for best if enabled
    )

    # Find the best RHCB5 run and re-run with XAI
    valid_rhcb5_results = [
        r for r in rhcb5_all_results
        if 'final_metrics' in r
        and isinstance(r['final_metrics'], dict)
        and 'accuracy' in r['final_metrics']
        and 'execution_time_sec' in r
    ]
    
    if valid_rhcb5_results:
        # Select the best accuracy
        best_rhcb5_run = max(valid_rhcb5_results, key=lambda r: r['final_metrics']['accuracy'])
        best_accuracy = best_rhcb5_run['final_metrics']['accuracy']
        best_run_id = best_rhcb5_run['run_id']
        best_seed = best_rhcb5_run['seed']
        print(f"\nMelhor run RHCB5: Run {best_run_id}, Seed {best_seed}, Accuracy {best_accuracy:.4f}, Time {best_rhcb5_run['execution_time_sec']:.2f}s")
        
        # Re-run the best run with XAI enabled if USE_XAI is True
        if USE_XAI:
            print("Re-executando com XAI...")
            best_rhcb5_result_with_xai = run_rhcb5_pipeline(
                run_id=best_run_id,
                base_results_dir=os.path.join(COMPARISON_RUN_DIR, "RHCB5_runs"),
                global_constants=GLOBAL_CONSTANTS,
                random_seed_for_run=best_seed,
                data_processed=data_processed,
                raw_labels=raw_labels,
                run_xai=True
            )
            
            # Update the results with XAI
            for i, result in enumerate(rhcb5_all_results):
                if result.get('run_id') == best_run_id:
                    rhcb5_all_results[i] = best_rhcb5_result_with_xai
                    break
        else:
            print("XAI desabilitado. Pulando re-execução com XAI.")
    else:
        print("Aviso: Nenhum run RHCB5 válido encontrado.")
        best_rhcb5_run = None

    # Find the best BDA-DNN run and re-run with XAI
    valid_bda_results = [
        r for r in bda_all_results
        if 'final_metrics' in r
        and isinstance(r['final_metrics'], dict)
        and 'accuracy' in r['final_metrics']
        and 'execution_time_sec' in r
    ]
    
    if valid_bda_results:
        # Select the best accuracy
        best_bda_run = max(valid_bda_results, key=lambda r: r['final_metrics']['accuracy'])
        best_accuracy = best_bda_run['final_metrics']['accuracy']
        best_run_id = best_bda_run['run_id']
        best_seed = best_bda_run['seed']
        print(f"\nMelhor run BDA-DNN: Run {best_run_id}, Seed {best_seed}, Accuracy {best_accuracy:.4f}, Time {best_bda_run['execution_time_sec']:.2f}s")
        
        # Re-run the best run with XAI enabled if USE_XAI is True
        if USE_XAI:
            print("Re-executando BDA-DNN com XAI...")
            best_bda_result_with_xai = run_bda_dnn_pipeline(
                run_id=best_run_id,
                base_results_dir=os.path.join(COMPARISON_RUN_DIR, "BDA_DNN_runs"),
                global_constants=GLOBAL_CONSTANTS,
                random_seed_for_run=best_seed,
                X_full_feat=X_full_feat,
                feature_names=feature_names,
                raw_labels=raw_labels,
                run_xai=True
            )
            
            # Update the results with XAI
            for i, result in enumerate(bda_all_results):
                if result.get('run_id') == best_run_id:
                    bda_all_results[i] = best_bda_result_with_xai
                    break
        else:
            print("XAI desabilitado. Pulando re-execução com XAI para BDA-DNN.")
    else:
        print("Aviso: Nenhum run BDA-DNN válido encontrado.")
        best_bda_run = None

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

    # 6. Análise Estatística de Comparação (Teste T Pareado com Testes de Pressupostos)
    print("\n--- Análise Estatística Comparativa (Teste T Pareado) ---")
    statistical_comparison_results = {}
    if (bda_df is not None) and (rhcb5_df is not None):
        try:
            # Função auxiliar para testes de pressupostos e comparação
            def perform_statistical_tests(metric_name, bda_values, rhcb5_values):
                print(f"\n--- Testes para {metric_name} ---")
                
                results = {
                    "metric": metric_name,
                    "tests": {}
                }
                
                # Remove NaN values
                bda_clean = bda_values.dropna()
                rhcb5_clean = rhcb5_values.dropna()
                
                if len(bda_clean) != len(rhcb5_clean):
                    print(f"Aviso: Número diferente de valores válidos para {metric_name} (BDA: {len(bda_clean)}, RHCB5: {len(rhcb5_clean)})")
                    min_len = min(len(bda_clean), len(rhcb5_clean))
                    bda_clean = bda_clean[:min_len]
                    rhcb5_clean = rhcb5_clean[:min_len]
                
                if len(bda_clean) < 3:
                    print(f"Aviso: Poucos dados válidos para {metric_name}, pulando testes.")
                    return None
                
                # Calcula diferenças para teste pareado
                differences = bda_clean - rhcb5_clean
                
                # 1. Teste de Normalidade (Shapiro-Wilk) nas diferenças
                shapiro_stat, shapiro_p = stats.shapiro(differences)
                normality_assumption = shapiro_p > 0.05
                results["tests"]["shapiro_wilk"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "normal_distribution": normality_assumption
                }
                print(f"Teste de Shapiro-Wilk (normalidade das diferenças) | Estatística: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
                if normality_assumption:
                    print("  > Diferenças seguem distribuição normal (p > 0.05).")
                else:
                    print("  > Diferenças não seguem distribuição normal (p <= 0.05).")
                
                # 2. Teste de Wilcoxon (não-paramétrico pareado)
                wilcoxon_result = stats.wilcoxon(bda_clean, rhcb5_clean)
                wilcoxon_significant = wilcoxon_result.pvalue < 0.05  # type: ignore
                results["tests"]["wilcoxon"] = {
                    "statistic": float(wilcoxon_result.statistic),  # type: ignore
                    "p_value": float(wilcoxon_result.pvalue),  # type: ignore
                    "significant_difference": wilcoxon_significant
                }
                print(f"Teste de Wilcoxon | Estatística: {wilcoxon_result.statistic:.4f}, p-value: {wilcoxon_result.pvalue:.4f}")  # type: ignore
                if wilcoxon_significant:
                    print("  > Diferença estatisticamente significativa (Wilcoxon).")
                else:
                    print("  > Nenhuma diferença estatisticamente significativa (Wilcoxon).")
                
                # 3. Teste T Pareado (se pressuposto de normalidade for atendido)
                if normality_assumption:
                    ttest_stat, ttest_p = stats.ttest_rel(bda_clean, rhcb5_clean)
                    ttest_significant = ttest_p < 0.05
                    results["tests"]["paired_t_test"] = {
                        "statistic": float(ttest_stat),
                        "p_value": float(ttest_p),
                        "significant_difference": ttest_significant
                    }
                    print(f"Teste T Pareado | Estatística: {ttest_stat:.4f}, p-value: {ttest_p:.4f}")
                    if ttest_significant:
                        print("  > Diferença estatisticamente significativa (Teste T Pareado).")
                    else:
                        print("  > Nenhuma diferença estatisticamente significativa (Teste T Pareado).")
                else:
                    results["tests"]["paired_t_test"] = {
                        "not_performed": True,
                        "reason": "Pressuposto de normalidade não atendido"
                    }
                    print("  > Pressuposto de normalidade não atendido, considere usar Wilcoxon em vez do Teste T.")
                
                # 4. Tamanho do Efeito (Cohen's d)
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)  # ddof=1 para amostras pequenas
                cohens_d = mean_diff / std_diff if std_diff != 0 else 0.0
                results["effect_size"] = {
                    "cohens_d": float(cohens_d),
                    "mean_difference": float(mean_diff),
                    "std_difference": float(std_diff)
                }
                print(f"Tamanho do Efeito (Cohen's d): {cohens_d:.4f}")
                if abs(cohens_d) < 0.2:
                    interpretation = "Pequeno"
                elif abs(cohens_d) < 0.5:
                    interpretation = "Pequeno a Médio"
                elif abs(cohens_d) < 0.8:
                    interpretation = "Médio"
                else:
                    interpretation = "Grande"
                results["effect_size"]["interpretation"] = interpretation
                print(f"  Interpretação: {interpretation} (d = {abs(cohens_d):.1f})")
                
                return results
            
            # Compara Acurácias
            acc_results = perform_statistical_tests(
                "Acurácia", 
                bda_df['accuracy'], 
                rhcb5_df['accuracy']
            )
            if acc_results:
                statistical_comparison_results["accuracy"] = acc_results
            
            # Compara F1-Macro
            f1_results = perform_statistical_tests(
                "F1-Macro", 
                bda_df['f1_macro'], 
                rhcb5_df['f1_macro']
            )
            if f1_results:
                statistical_comparison_results["f1_macro"] = f1_results
            
            # Compara métricas por classe (foco na classe Ictal - 2)
            recall_2_results = perform_statistical_tests(
                "Recall Classe Ictal (2)", 
                bda_df['recall_2'], 
                rhcb5_df['recall_2']
            )
            if recall_2_results:
                statistical_comparison_results["recall_class_2"] = recall_2_results
            
            specificity_2_results = perform_statistical_tests(
                "Specificity Classe Ictal (2)", 
                bda_df['specificity_2'], 
                rhcb5_df['specificity_2']
            )
            if specificity_2_results:
                statistical_comparison_results["specificity_class_2"] = specificity_2_results
            
            f1_score_2_results = perform_statistical_tests(
                "F1-Score Classe Ictal (2)", 
                bda_df['f1_score_2'], 
                rhcb5_df['f1_score_2']
            )
            if f1_score_2_results:
                statistical_comparison_results["f1_score_class_2"] = f1_score_2_results
        
        except Exception as e:
            print(f"Erro ao executar testes estatísticos: {e}")
            import traceback
            traceback.print_exc()
    
    # Coletar dados adicionais para o JSON
    print("\n--- Coletando Dados Adicionais para Análise ---")
    
    # Delta accuracy per run
    bda_accs = [run['final_metrics']['accuracy'] for run in bda_all_results if 'final_metrics' in run]
    rhcb5_accs = [run['final_metrics']['accuracy'] for run in rhcb5_all_results if 'final_metrics' in run]
    if len(bda_accs) == len(rhcb5_accs):
        delta_accuracy = (np.array(rhcb5_accs) - np.array(bda_accs)).tolist()
        statistical_comparison_results["delta_accuracy_per_run"] = delta_accuracy
        print(f"Delta accuracy coletado para {len(delta_accuracy)} runs.")
    
    # Confusion matrix stds
    def collect_cm_stds(results_list, pipeline_name):
        cms = []
        for run_result in results_list:
            if 'final_metrics' in run_result and 'confusion_matrix' in run_result['final_metrics']:
                cm = np.array(run_result['final_metrics']['confusion_matrix'])
                cms.append(cm)
        if cms:
            cms_array = np.array(cms)
            std_cm = np.std(cms_array, axis=0).tolist()
            return std_cm
        return None
    
    bda_cm_std = collect_cm_stds(bda_all_results, "BDA_DNN")
    if bda_cm_std:
        statistical_comparison_results["confusion_matrix_std_BDA_DNN"] = bda_cm_std
    
    rhcb5_cm_std = collect_cm_stds(rhcb5_all_results, "RHCB5")
    if rhcb5_cm_std:
        statistical_comparison_results["confusion_matrix_std_RHCB5"] = rhcb5_cm_std
    
    # Feature selection frequencies for BDA
    selected_vectors = []
    for run_result in bda_all_results:
        if 'selected_features_vector' in run_result:
            vector = np.array(run_result['selected_features_vector'])
            selected_vectors.append(vector)
    if selected_vectors:
        selection_freq = np.sum(selected_vectors, axis=0).tolist()
        statistical_comparison_results["feature_selection_frequencies_BDA"] = selection_freq
        print(f"Frequências de seleção de features coletadas: {len(selection_freq)} features.")
    
    # Salvar resultados dos testes estatísticos
    if statistical_comparison_results:
        stats_comparison_path = os.path.join(COMPARISON_RUN_DIR, "statistical_comparison_results.json")
        try:
            with open(stats_comparison_path, "w") as f:
                json.dump(statistical_comparison_results, f, cls=NumpyEncoder, indent=4)
            print(f"Resultados dos testes estatísticos salvos em: {stats_comparison_path}")
        except Exception as e:
            print(f"Erro ao salvar resultados dos testes estatísticos: {e}")

    # 11. Testes de Estabilidade Estatística (Friedman, Nemenyi)
    print("\n--- Testes de Estabilidade Estatística ---")
    from scipy.stats import friedmanchisquare
    
    stability_results = {}
    
    # Friedman test for accuracies across runs
    if bda_df is not None and rhcb5_df is not None:
        bda_accs = bda_df['accuracy'].dropna().values
        rhcb5_accs = rhcb5_df['accuracy'].dropna().values
        
        min_len = min(len(bda_accs), len(rhcb5_accs))
        bda_accs = bda_accs[:min_len]
        rhcb5_accs = rhcb5_accs[:min_len]

    # Nemenyi test would require scikit-posthocs, but let's skip for now or implement manually
    # For now, just Friedman
    
    # Salvar resultados de estabilidade
    if stability_results:
        stability_path = os.path.join(COMPARISON_RUN_DIR, "stability_tests.json")
        try:
            with open(stability_path, "w") as f:
                json.dump(stability_results, f, indent=4)
            print(f"Testes de estabilidade salvos em: {stability_path}")
        except Exception as e:
            print(f"Erro ao salvar testes de estabilidade: {e}")
    from sklearn.inspection import permutation_importance
    
    feature_importance_results = {}
    
    # Para BDA-DNN - re-run the best run to get the model
    # NOTE: Currently disabled because run_bda_dnn_pipeline doesn't return models
    # and re-running for PI would be complex. Models are not saved in the pipeline.
    # if bda_df is not None and not bda_df['accuracy'].empty:
    #     best_bda_idx = bda_df['accuracy'].idxmax()
    #     best_bda_run = bda_all_results[best_bda_idx]
    #     best_bda_seed = best_bda_run['seed']
    #     best_bda_run_id = best_bda_run['run_id']
        
    #     print(f"Re-executando melhor run BDA (Run {best_bda_run_id}, Seed {best_bda_seed}) para análise de importância...")
        
    #     try:
            # Re-run BDA to get the trained model
    #         bda_result_for_pi = run_bda_dnn_pipeline(
    #             run_id=best_bda_run_id,
    #             base_results_dir=os.path.join(COMPARISON_RUN_DIR, "BDA_DNN_runs"),
    #             global_constants=GLOBAL_CONSTANTS,
    #             random_seed_for_run=best_bda_seed,
    #             X_full_feat=X_full_feat,
    #             feature_names=feature_names,
    #             raw_labels=raw_labels,
    #             return_model=True  # This parameter does not exist in the function signature
    #         )
            
    #         if 'model' in bda_result_for_pi and 'X_test' in bda_result_for_pi and 'y_test' in bda_result_for_pi:
    #             model = bda_result_for_pi['model']
    #             X_test = bda_result_for_pi['X_test']
    #             y_test = bda_result_for_pi['y_test']
                
    #             # Compute permutation importance
    #             pi_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy')
                
    #             bda_pi = {
    #                 "importances": pi_result.importances.tolist(),
    #                 "importances_mean": pi_result.importances_mean.tolist(),
    #                 "importances_std": pi_result.importances_std.tolist(),
    #                 "feature_names": bda_result_for_pi.get('selected_feature_names', feature_names[:len(pi_result.importances_mean)] if feature_names else None)
    #             }
    #             feature_importance_results["BDA_DNN"] = bda_pi
    #             print("Permutation Importance calculada para BDA-DNN.")
    #         else:
    #             print("Erro: Não foi possível obter modelo e dados de teste para BDA.")
    #     except Exception as e:
    #         print(f"Erro na análise de importância para BDA: {e}")
    
    # Para RHCB5 - já temos o modelo do best run with XAI
    if best_rhcb5_run and 'model' in best_rhcb5_run:
        print("Usando modelo RHCB5 já treinado para análise de importância...")
        try:
            model = best_rhcb5_run['model']
            # Need test data - re-run to get it or assume it's saved
            # For simplicity, since RHCB5 uses raw signals, and PI on 4096 features is heavy, perhaps skip or do on subset
            print("Permutation Importance para RHCB5 seria muito custoso (4096 features). Usando SHAP em vez disso.")
        except Exception as e:
            print(f"Erro na análise de importância para RHCB5: {e}")
    
    # Salvar resultados de importância
    if feature_importance_results:
        pi_path = os.path.join(COMPARISON_RUN_DIR, "feature_importance.json")
        try:
            with open(pi_path, "w") as f:
                json.dump(feature_importance_results, f, indent=4)
            print(f"Análise de importância de features salva em: {pi_path}")
        except Exception as e:
            print(f"Erro ao salvar importância de features: {e}")
    
    correlation_results = {}
    
    def calculate_correlations(df, pipeline_name, metrics_list):
        """Calcula correlações de Pearson entre métricas."""
        results = {}
        for i, metric1 in enumerate(metrics_list):
            for j, metric2 in enumerate(metrics_list):
                if i < j and metric1 in df.columns and metric2 in df.columns:
                    data1 = df[metric1].dropna()
                    data2 = df[metric2].dropna()
                    if len(data1) == len(data2) and len(data1) > 1:
                        corr, p_val = stats.pearsonr(data1, data2)
                        results[f"{metric1}_vs_{metric2}"] = {
                            "correlation": float(corr),
                            "p_value": float(p_val),
                            "significant": p_val < 0.05
                        }
        return results
    
    # Correlações para BDA-DNN
    if bda_df is not None:
        bda_metrics = ['accuracy', 'f1_macro', 'exec_time_sec', 'num_features', 'bda_fitness', 'dnn_time_sec']
        bda_corr = calculate_correlations(bda_df, "BDA_DNN", bda_metrics)
        correlation_results["BDA_DNN"] = bda_corr
        print("Correlações BDA-DNN:")
        for pair, res in bda_corr.items():
            print(f"  {pair}: r={res['correlation']:.3f}, p={res['p_value']:.3f} {'(sig)' if res['significant'] else '(ns)'}")
    
    # Correlações para RHCB5
    if rhcb5_df is not None:
        rhcb5_metrics = ['accuracy', 'f1_macro', 'exec_time_sec', 'dnn_time_sec']
        rhcb5_corr = calculate_correlations(rhcb5_df, "RHCB5", rhcb5_metrics)
        correlation_results["RHCB5"] = rhcb5_corr
        print("Correlações RHCB5:")
        for pair, res in rhcb5_corr.items():
            print(f"  {pair}: r={res['correlation']:.3f}, p={res['p_value']:.3f} {'(sig)' if res['significant'] else '(ns)'}")
    
    # Correlações entre pipelines (usando dados pareados)
    if bda_df is not None and rhcb5_df is not None:
        paired_corr = {}
        for metric in ['accuracy', 'f1_macro', 'exec_time_sec']:
            if metric in bda_df.columns and metric in rhcb5_df.columns:
                bda_data = bda_df[metric].dropna()
                rhcb5_data = rhcb5_df[metric].dropna()
                if len(bda_data) == len(rhcb5_data) and len(bda_data) > 1:
                    corr, p_val = stats.pearsonr(bda_data, rhcb5_data)
                    paired_corr[f"bda_{metric}_vs_rhcb5_{metric}"] = {
                        "correlation": float(corr),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05
                    }
        correlation_results["paired_pipelines"] = paired_corr
        print("Correlações entre pipelines:")
        for pair, res in paired_corr.items():
            print(f"  {pair}: r={res['correlation']:.3f}, p={res['p_value']:.3f} {'(sig)' if res['significant'] else '(ns)'}")
    
    # Salvar correlações
    if correlation_results:
        corr_path = os.path.join(COMPARISON_RUN_DIR, "correlation_analysis.json")
        try:
            with open(corr_path, "w") as f:
                json.dump(correlation_results, f, cls=NumpyEncoder, indent=4)
            print(f"Análise de correlação salva em: {corr_path}")
        except Exception as e:
            print(f"Erro ao salvar correlações: {e}")
    from scipy.stats import bootstrap
    
    confidence_intervals = {}
    
    def calculate_bootstrap_ci(data, n_resamples=10000, confidence_level=0.95):
        """Calcula intervalo de confiança via bootstrap."""
        if len(data) < 2:
            return None
        
        def statistic_func(sample):
            return np.mean(sample)
        
        try:
            res = bootstrap((data,), statistic_func, n_resamples=n_resamples, confidence_level=confidence_level, method='percentile')
            return {
                "mean": float(np.mean(data)),
                "ci_lower": float(res.confidence_interval.low),
                "ci_upper": float(res.confidence_interval.high),
                "ci_level": confidence_level
            }
        except Exception as e:
            print(f"Erro no cálculo de CI: {e}")
            return None
    
    # Para BDA-DNN
    if bda_df is not None:
        bda_ci = {}
        for metric in ['accuracy', 'f1_macro', 'exec_time_sec']:
            if metric in bda_df.columns:
                data = bda_df[metric].dropna()
                ci = calculate_bootstrap_ci(data.values)
                if ci:
                    bda_ci[metric] = ci
                    print(f"BDA-DNN {metric}: Mean={ci['mean']:.4f}, CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        confidence_intervals["BDA_DNN"] = bda_ci
    
    # Para RHCB5
    if rhcb5_df is not None:
        rhcb5_ci = {}
        for metric in ['accuracy', 'f1_macro', 'exec_time_sec']:
            if metric in rhcb5_df.columns:
                data = rhcb5_df[metric].dropna()
                ci = calculate_bootstrap_ci(data.values)
                if ci:
                    rhcb5_ci[metric] = ci
                    print(f"RHCB5 {metric}: Mean={ci['mean']:.4f}, CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        confidence_intervals["RHCB5"] = rhcb5_ci
    
    # Salvar CIs
    if confidence_intervals:
        ci_path = os.path.join(COMPARISON_RUN_DIR, "confidence_intervals.json")
        try:
            with open(ci_path, "w") as f:
                json.dump(confidence_intervals, f, indent=4)
            print(f"Intervalos de confiança salvos em: {ci_path}")
        except Exception as e:
            print(f"Erro ao salvar CIs: {e}")
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
    
    # Scatter plot: Performance vs Cost
    Plotting.plot_performance_vs_cost_scatter(
        results_dict_list_for_plotting,
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
    
    # Delta accuracy per run
    Plotting.plot_delta_accuracy_per_run(
        results_dict_list_for_plotting, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Confusion matrix std heatmaps
    Plotting.plot_confusion_matrix_std_heatmap(
        bda_all_results, "BDA_DNN", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    Plotting.plot_confusion_matrix_std_heatmap(
        rhcb5_all_results, "RHCB5", CLASS_NAMES, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Feature selection frequency for BDA
    # Get feature names from the first valid BDA run
    feature_names = None
    for run in bda_all_results:
        if 'feature_names' in run and run['feature_names'] is not None:
            feature_names = run['feature_names']
            break
    Plotting.plot_feature_selection_frequency(
        bda_all_results, feature_names, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Sensitivity Analysis for BDA-DNN
    Plotting.plot_sensitivity_analysis(
        bda_all_results, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    # Complementary Visualizations
    Plotting.plot_distribution_plots(
        results_dict_list_for_plotting, COMPARISON_PLOTS_DIR, SAVE_PLOTS
    )
    
    total_main_time = time.time() - start_time_main
    print("\n" + "="*80)
    print("COMPARAÇÃO COMPLETA CONCLUÍDA")
    print(f"Tempo total da orquestração: {total_main_time/60:.2f} minutos.")
    print(f"Resultados, estatísticas e gráficos salvos em: {COMPARISON_RUN_DIR}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro inesperado na execução principal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
