# -*- coding: utf-8 -*-
"""
Script de Análise de Agregação BDA-DNN

Este script realiza uma meta-análise de TODAS as execuções do BDA-DNN
dentro de um diretório de comparação.

Ele calcula a importância agregada de cada feature (das 143 originais)
baseando-se em:
1. A frequência com que a feature foi selecionada pelo BDA.
2. A acurácia do modelo que resultou dessa seleção.

Isso responde à pergunta: "Quais features, quando selecionadas, 
estão mais associadas a modelos de alta acurácia?"
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Importação dos Módulos do Projeto ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from pipeline_utils import Plotting
    from pipeline_bda_dnn import FeatureExtractor
except ImportError as e:
    print(f"ERRO: Não foi possível importar os módulos dos pipelines: {e}")
    sys.exit(1)

# --- Constantes ---
BASE_RESULTS_DIR = os.path.join(current_dir, "results")
BASE_DATA_DIR = os.path.join(current_dir, "..", "data") # Para carregar feature names


def find_latest_comparison_dir(base_results_dir):
    """Encontra o diretório 'comparison_run_*' mais recente."""
    list_of_dirs = glob.glob(os.path.join(base_results_dir, "comparison_run_*"))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

def get_feature_names():
    """Gera os nomes das 143 features para usar nos labels."""
    # Precisamos de 1 amostra de dados para gerar os nomes
    dummy_data = np.random.rand(1, 4096) 
    _, feature_names = FeatureExtractor.extract_swt_features(
        dummy_data, wavelet='db4', level=4
    )
    return feature_names # Lista de 143 nomes

def aggregate_bda_results(comparison_dir, feature_names):
    """
    Carrega todos os JSONs de resultado do BDA-DNN e calcula a 
    importância agregada de cada feature.
    """
    print("Iniciando análise de agregação BDA-DNN...")
    bda_runs_dir = os.path.join(comparison_dir, "BDA_DNN_runs")
    
    # 1. Encontrar todos os JSONs de resultado
    json_files = glob.glob(os.path.join(bda_runs_dir, "run_*", "run_results.json"))
    
    if not json_files:
        print(f"ERRO: Nenhum arquivo 'run_results.json' encontrado em {bda_runs_dir}")
        return

    all_run_data = []
    
    # 2. Ler cada JSON e extrair o vetor de features e a acurácia
    for f_path in json_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            
            # Pega o vetor (lista de 143 0s e 1s)
            vector = data["selected_features_vector"]
            # Pega a acurácia final
            accuracy = data["final_accuracy"]
            
            if len(vector) != len(feature_names):
                print(f"AVISO: {f_path} tem {len(vector)} features, esperado {len(feature_names)}. Pulando.")
                continue
                
            all_run_data.append({
                "accuracy": accuracy,
                "vector": vector
            })
        except Exception as e:
            print(f"Erro ao ler {f_path}: {e}")

    if not all_run_data:
        print("Nenhum dado de execução válido foi carregado.")
        return

    print(f"Dados de {len(all_run_data)} execuções do BDA-DNN carregados.")

    # 3. Criar DataFrames para análise
    # DataFrame com vetores (ex: 50 linhas, 143 colunas)
    df_features = pd.DataFrame([run['vector'] for run in all_run_data], 
                               columns=feature_names)
    
    # Série com acurácias (ex: 50 linhas)
    s_accuracy = pd.Series([run['accuracy'] for run in all_run_data])

    # 4. Calcular Métricas de Importância
    
    # Métrica 1: Contagem Simples de Seleção
    # (Quantas vezes cada feature foi selecionada, de 0 a 50)
    selection_count = df_features.sum(axis=0)

    # Métrica 2: Importância Ponderada pela Acurácia (a melhor)
    # (Feature_Vector_Transposto) x (Vetor_Acurácia)
    # (143, 50) x (50, 1) = (143, 1)
    weighted_importance = df_features.T.dot(s_accuracy)

    # 5. Criar DataFrame de resultados e ordenar
    df_results = pd.DataFrame({
        "feature_name": feature_names,
        "selection_count": selection_count,
        "weighted_importance": weighted_importance
    })
    
    df_results = df_results.sort_values(by="weighted_importance", ascending=False)
    
    return df_results

def plot_aggregated_importance(df_results, output_dir, top_n=30):
    """
    Plota um gráfico de barras horizontal com as features mais importantes.
    """
    print(f"Gerando gráfico de importância agregada (Top {top_n})...")
    
    # Pega as Top N
    df_plot = df_results.head(top_n).sort_values(by="weighted_importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4))
    
    # Plota a importância ponderada (barra principal)
    ax.barh(
        df_plot["feature_name"], 
        df_plot["weighted_importance"], 
        color='steelblue', 
        edgecolor='black',
        label='Importância Ponderada (Acurácia * Seleção)'
    )
    
    ax.set_title(f'BDA-DNN - Importância Agregada das Features (Top {top_n} de {len(df_results)})\n'
                 f'Ponderada pela Acurácia em {len(df_plot)} execuções')
    ax.set_xlabel('Pontuação de Importância Agregada (Soma da Acurácia Ponderada)')
    ax.set_ylabel('Feature')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adiciona a contagem de seleção (texto) em cada barra
    for i, (count, name) in enumerate(zip(df_plot["selection_count"], df_plot["feature_name"])):
        ax.text(
            0.01, # Posição X (logo no início da barra)
            i, # Posição Y (índice da barra)
            f'  (Selecionada {count}x)', 
            va='center', 
            ha='left', 
            fontsize=9, 
            color='white'
        )

    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Salvar o gráfico
    save_path = os.path.join(output_dir, "bda_dnn_aggregated_importance.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Gráfico de agregação salvo em: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Executa Análise de Agregação BDA-DNN."
    )
    parser.add_argument(
        "comparison_dir", 
        nargs='?', 
        default=None,
        help="Caminho para o diretório 'comparison_run_...'. "
             "Se omitido, usa o mais recente."
    )
    args = parser.parse_args()

    # 1. Encontrar Diretório de Resultados
    if args.comparison_dir:
        comparison_dir = args.comparison_dir
    else:
        print("Nenhum diretório especificado. Procurando o mais recente...")
        comparison_dir = find_latest_comparison_dir(BASE_RESULTS_DIR)
        
    if not comparison_dir or not os.path.isdir(comparison_dir):
        print(f"ERRO: Diretório de comparação não encontrado: {comparison_dir}")
        sys.exit(1)
            
    print(f"Usando diretório de comparação: {comparison_dir}")
    
    # Diretório de saída (mesma pasta do XAI)
    xai_dir = os.path.join(comparison_dir, "XAI_analysis")
    os.makedirs(xai_dir, exist_ok=True)

    try:
        # 2. Obter nomes das features
        feature_names = get_feature_names()
        
        # 3. Rodar a análise
        df_results = aggregate_bda_results(comparison_dir, feature_names)
        
        if df_results is not None:
            # 4. Salvar resultados em CSV
            csv_path = os.path.join(xai_dir, "bda_dnn_aggregated_importance.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"Resultados de agregação salvos em: {csv_path}")
            
            # 5. Plotar
            plot_aggregated_importance(df_results, xai_dir, top_n=30)
            
        print("\nAnálise de Agregação BDA-DNN concluída.")

    except Exception as e:
        print(f"ERRO fatal na análise de agregação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()