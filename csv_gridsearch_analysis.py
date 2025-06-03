import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

# --- Configurações ---
# Caminho para o arquivo JSON principal de resumo
JSON_MASTER_RESULTS_FILE = 'results/dnn_gridsearch_full_cv_output/gridsearch_ALL_SOURCES_summary.json' # Assumindo que está no mesmo dir do script ou ajuste

# Diretório base onde os CSVs individuais (cv_results_bda.csv, etc.) e o JSON estão.
# O script test_dnn_standalone.py salva os CSVs em subdiretórios por feature_source.
# Ex: results/dnn_gridsearch_full_cv_output/features_bda/cv_results_bda.csv
# E o JSON mestre em: results/dnn_gridsearch_full_cv_output/gridsearch_ALL_SOURCES_summary.json
# Vamos assumir que este script de análise está no diretório PAI de 'results'.
BASE_RESULTS_DIR_FROM_SCRIPT = 'results/dnn_gridsearch_full_cv_output' # Ajuste se o seu script de teste salvou em outro lugar

OUTPUT_PLOTS_DIR = 'results/analysis_plots_cv_full_data' # Novo diretório para esta análise
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# Hiperparâmetros a serem analisados
# Estes são os nomes das colunas como aparecem após o 'param_' prefixo no cv_results_
PARAMS_TO_ANALYZE = {
    'numeric': [
        'batch_size',
        'epochs',
        'model__learning_rate',
        'model__dropout_rate1', # Assumindo que as 3 taxas de dropout foram testadas com os mesmos valores
        'model__kernel_regularizer_strength'
    ],
    'categorical': [
        'model__optimizer_name',
        'model__kernel_regularizer_type',
        # 'model__momentum' # Adicionar se testado e relevante
    ]
}

def load_data_from_csv(feature_source_name):
    """Tenta carregar os resultados do CV de um arquivo CSV individual."""
    # Constrói o caminho esperado para o arquivo CSV
    # Ex: results/dnn_gridsearch_full_cv_output/features_bda/cv_results_bda.csv
    csv_path = os.path.join(BASE_RESULTS_DIR_FROM_SCRIPT, f"features_{feature_source_name}", f"cv_results_{feature_source_name}.csv")
    try:
        df = pd.read_csv(csv_path)
        print(f"Dados carregados com sucesso do CSV: {csv_path}")
        # Renomear colunas param_* para remover o prefixo para facilitar nos plots
        # O Pandas já carrega as colunas do cv_results_ com o prefixo 'param_'
        df.columns = [col.replace('param_model__', 'model__').replace('param_', '') if col.startswith('param_') else col for col in df.columns]
        return df
    except FileNotFoundError:
        print(f"AVISO: Arquivo CSV '{csv_path}' não encontrado para {feature_source_name}.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar CSV para {feature_source_name} de {csv_path}: {e}")
        return None

def load_data_from_json_summary(feature_source_name_filter):
    """Carrega os resultados do CV do sumário no arquivo JSON principal."""
    json_file_path = os.path.join(BASE_RESULTS_DIR_FROM_SCRIPT, JSON_MASTER_RESULTS_FILE)
    try:
        with open(json_file_path, 'r') as f:
            all_runs_data = json.load(f)
        
        for run_data in all_runs_data:
            if run_data.get("feature_source") == feature_source_name_filter:
                # A chave esperada é 'grid_cv_results_summary_sorted'
                cv_results_list_of_dicts = run_data.get("grid_cv_results_summary_sorted")
                if cv_results_list_of_dicts:
                    df = pd.DataFrame(cv_results_list_of_dicts)
                    # As colunas no JSON já devem estar como "param_batch_size", etc.
                    # Renomear para consistência com o carregamento do CSV
                    df.columns = [col.replace('param_model__', 'model__').replace('param_', '') if col.startswith('param_') else col for col in df.columns]
                    print(f"Dados carregados com sucesso do JSON para {feature_source_name_filter}.")
                    return df
                else:
                    print(f"Chave 'grid_cv_results_summary_sorted' não encontrada ou vazia para {feature_source_name_filter} no JSON.")
                    return pd.DataFrame()
        print(f"Fonte de features '{feature_source_name_filter}' não encontrada no arquivo JSON.")
        return pd.DataFrame()
    except FileNotFoundError:
        print(f"ERRO: Arquivo JSON de resultados '{json_file_path}' não encontrado.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERRO inesperado ao carregar dados do JSON para {feature_source_name_filter}: {e}")
        return pd.DataFrame()

# Funções de plotagem (plot_sensitivity, plot_boxplot_comparison, plot_heatmap_interaction, plot_score_vs_time)
# As funções de plotagem permanecem as mesmas da sua versão anterior (artefato analise_grafica_gridsearch_cv_log_melhorado)
# Vou incluí-las aqui para completude.

def plot_sensitivity(df, param_name, feature_source_name, output_dir, hue_param=None):
    col_param_name = param_name 
    
    if df.empty or col_param_name not in df.columns:
        print(f"DataFrame vazio ou parâmetro '{col_param_name}' não encontrado para {feature_source_name} em plot_sensitivity.")
        return

    plt.figure(figsize=(12, 7)) # Aumentado um pouco
    
    df_copy = df.copy()
    try:
        # Certificar que a coluna do parâmetro principal é numérica
        df_copy[col_param_name] = pd.to_numeric(df_copy[col_param_name], errors='coerce')
        # Certificar que mean_test_score é numérico
        df_copy['mean_test_score'] = pd.to_numeric(df_copy['mean_test_score'], errors='coerce')
        df_copy.dropna(subset=[col_param_name, 'mean_test_score'], inplace=True)
    except Exception as e:
        print(f"Erro ao converter colunas para numérico em plot_sensitivity ({param_name}): {e}")
        return

    if df_copy.empty:
        print(f"DataFrame vazio após tratamento de NaNs para {col_param_name} em {feature_source_name} (plot_sensitivity).")
        return
    
    col_hue_param = None
    if hue_param:
        col_hue_param = hue_param.replace('model__', '') if hue_param.startswith('model__') else hue_param
        if col_hue_param not in df_copy.columns:
            print(f"Aviso: Parâmetro de HUE '{col_hue_param}' (original: {hue_param}) não encontrado no DataFrame. Plotando sem HUE.")
            col_hue_param = None
        elif df_copy[col_hue_param].isnull().all(): # Checar se a coluna HUE é toda NaN
            print(f"Aviso: Parâmetro de HUE '{col_hue_param}' contém apenas NaNs. Plotando sem HUE.")
            col_hue_param = None


    sns.lineplot(data=df_copy, x=col_param_name, y='mean_test_score', marker='o', errorbar='sd', hue=col_hue_param, legend='auto')
    
    title = f'Sensibilidade: Score vs {col_param_name}'
    if col_hue_param:
        title += f' (Agrupado por {col_hue_param})'
    title += f'\nFonte: {feature_source_name}'
    
    plt.title(title, fontsize=14)
    plt.xlabel(col_param_name, fontsize=12)
    plt.ylabel('Acurácia Média de Teste (CV)', fontsize=12)
    if col_param_name == 'model__learning_rate' or col_param_name == 'learning_rate':
        plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.7)
    if col_hue_param:
        plt.legend(title=col_hue_param, bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize='10', fontsize='9')
        plt.tight_layout(rect=[0, 0, 0.82, 1]) # Ajustar para legenda
    else:
        plt.tight_layout()
        
    filename = os.path.join(output_dir, f"sensitivity_{col_param_name.replace('model__','')}{'_hue_' + col_hue_param.replace('model__','') if col_hue_param else ''}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de sensibilidade salvo em: {filename}")

def plot_boxplot_comparison(df, param_name, feature_source_name, output_dir):
    col_param_name = param_name
    if df.empty or col_param_name not in df.columns:
        print(f"DataFrame vazio ou parâmetro '{col_param_name}' não encontrado para {feature_source_name} em plot_boxplot_comparison.")
        return

    plt.figure(figsize=(10, 7)) # Aumentado um pouco
    
    # Tratar NaNs na coluna do parâmetro antes de plotar
    df_copy = df.copy()
    df_copy[col_param_name] = df_copy[col_param_name].fillna('Não Especificado')


    sns.stripplot(data=df_copy, x=col_param_name, y='mean_test_score', jitter=True, dodge=True, alpha=0.6, s=5)
    sns.boxplot(data=df_copy, x=col_param_name, y='mean_test_score', showfliers=False, color='lightgray', width=0.5,
                boxprops=dict(alpha=.7))
    
    plt.title(f'Comparação de Score por {col_param_name}\nFonte: {feature_source_name}', fontsize=14)
    plt.xlabel(col_param_name, fontsize=12)
    plt.ylabel('Acurácia Média de Teste (CV)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"boxplot_{col_param_name.replace('model__','')}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de boxplot salvo em: {filename}")

def plot_heatmap_interaction(df, param1_name, param2_name, feature_source_name, output_dir):
    col_param1_name = param1_name
    col_param2_name = param2_name
    
    if df.empty or col_param1_name not in df.columns or col_param2_name not in df.columns:
        print(f"DataFrame vazio ou parâmetros '{col_param1_name}' ou '{col_param2_name}' não encontrados para {feature_source_name} em plot_heatmap_interaction.")
        return

    df_subset = df.copy()
    df_subset.dropna(subset=[col_param1_name, col_param2_name, 'mean_test_score'], inplace=True)

    if df_subset.empty:
        print(f"DataFrame vazio após remover NaNs para heatmap {col_param1_name} vs {col_param2_name} em {feature_source_name}.")
        return

    # Caso especial para regularização
    is_reg_strength1 = 'kernel_regularizer_strength' in col_param1_name
    is_reg_type1 = 'kernel_regularizer_type' in col_param1_name
    is_reg_strength2 = 'kernel_regularizer_strength' in col_param2_name
    is_reg_type2 = 'kernel_regularizer_type' in col_param2_name

    if (is_reg_type1 and is_reg_strength2) or (is_reg_strength1 and is_reg_type2):
        type_col = col_param1_name if is_reg_type1 else col_param2_name
        strength_col = col_param2_name if is_reg_type1 else col_param1_name # A outra coluna é a de força
        
        # Filtra para apenas os tipos que usam força (l1, l2)
        df_subset = df_subset[df_subset[type_col].isin(['l1', 'l2'])]
        
        # Converte a coluna de força para numérica para garantir a ordem correta no heatmap
        df_subset[strength_col] = pd.to_numeric(df_subset[strength_col], errors='coerce')
        df_subset.dropna(subset=[strength_col], inplace=True)

        if df_subset.empty:
            print(f"Sem dados para heatmap de interação de regularizadores (l1/l2) para {feature_source_name}")
            return
    
    try:
        # Tenta converter para numérico se for o caso, para melhor ordenação no pivot
        for p in [col_param1_name, col_param2_name]:
            if df_subset[p].dtype == 'object':
                try:
                    df_subset[p] = pd.to_numeric(df_subset[p])
                except ValueError:
                    pass # Mantém como objeto se não puder converter

        pivot_df = df_subset.pivot_table(
            values='mean_test_score', 
            index=col_param1_name, 
            columns=col_param2_name, 
            aggfunc='max' # Usar max para ver o melhor resultado para cada combinação
        )
    except Exception as e:
        print(f"Erro ao criar pivot table (max) para {col_param1_name} vs {col_param2_name} em {feature_source_name}: {e}")
        try:
            pivot_df = df_subset.pivot_table(
                values='mean_test_score', 
                index=col_param1_name, 
                columns=col_param2_name, 
                aggfunc='mean'
            )
            print("Usando .mean() para pivot table como fallback.")
        except Exception as e2:
            print(f"Erro ao criar pivot table (fallback com .mean()) para {col_param1_name} vs {col_param2_name} em {feature_source_name}: {e2}")
            return

    if pivot_df.empty:
        print(f"Pivot table vazia para {col_param1_name} vs {col_param2_name} em {feature_source_name}.")
        return

    plt.figure(figsize=(12, 9)) # Aumentado um pouco
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis_r", linewidths=.5, cbar_kws={'label': 'Acurácia Média (CV)'})
    plt.title(f'Heatmap: Score para {col_param1_name} vs {col_param2_name}\nFonte: {feature_source_name} (Max/Mean Score Agrupado)', fontsize=14)
    plt.xlabel(col_param2_name, fontsize=12)
    plt.ylabel(col_param1_name, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"heatmap_{col_param1_name.replace('model__','')}_vs_{col_param2_name.replace('model__','')}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de heatmap salvo em: {filename}")


def plot_score_vs_time(df, feature_source_name, output_dir):
    if df.empty:
        print(f"DataFrame vazio para {feature_source_name}. Pulando plot score vs time.")
        return
    
    required_cols = ['mean_test_score', 'mean_fit_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colunas ausentes para o gráfico Score vs. Tempo para {feature_source_name}: {', '.join(missing_cols)}. Pulando este gráfico.")
        return

    # Converter para numérico e remover NaNs
    df_copy = df.copy()
    try:
        df_copy['mean_test_score'] = pd.to_numeric(df_copy['mean_test_score'], errors='coerce')
        df_copy['mean_fit_time'] = pd.to_numeric(df_copy['mean_fit_time'], errors='coerce')
        df_copy.dropna(subset=['mean_test_score', 'mean_fit_time'], inplace=True)
    except Exception as e:
        print(f"Erro ao converter colunas para numérico em plot_score_vs_time: {e}")
        return
        
    if df_copy.empty:
        print(f"DataFrame vazio após tratamento de NaNs para plot_score_vs_time ({feature_source_name}).")
        return

    plt.figure(figsize=(12, 7)) # Aumentado
    hue_col_processed = 'model__optimizer_name' 
    if hue_col_processed not in df_copy.columns:
        hue_col_processed = None 
    
    scatter_plot = sns.scatterplot(data=df_copy, x='mean_fit_time', y='mean_test_score', hue=hue_col_processed, alpha=0.7, s=60, legend='auto')
    plt.title(f'Score de Teste (CV) vs. Tempo Médio de Treino\nFonte: {feature_source_name}', fontsize=14)
    plt.xlabel('Tempo Médio de Treino (s)', fontsize=12)
    plt.ylabel('Acurácia Média de Teste (CV)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    if hue_col_processed:
        plt.legend(title='Otimizador', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize='10', fontsize='9')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        plt.tight_layout()
        
    filename = os.path.join(output_dir, f"score_vs_fit_time_{feature_source_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico Score vs. Tempo salvo em: {filename}")


# --- Loop Principal de Análise ---
FEATURE_SOURCES_TO_PROCESS = ['bda', 'bpso'] 

# Nomes de colunas como esperamos que estejam no DataFrame após o processamento em load_data_from_csv/json
# (ou seja, 'param_' removido, 'model__' mantido para params do modelo)
PARAMS_TO_ANALYZE_PROCESSED = {
    'numeric': [
        'batch_size',
        'epochs',
        'model__learning_rate', 
        'model__dropout_rate1', # Assumindo que dropout_rate1 é representativo ou você pode adicionar _rate2, _rate3
        'model__kernel_regularizer_strength'
    ],
    'categorical': [
        'model__optimizer_name',
        'model__kernel_regularizer_type',
    ]
}

for fs_name in FEATURE_SOURCES_TO_PROCESS:
    print(f"\n\n--- Processando Fonte de Features: {fs_name.upper()} ---")
    
    # Tentar carregar do CSV primeiro
    df_cv_results = load_data_from_csv(fs_name)
    
    if df_cv_results is None or df_cv_results.empty:
        print(f"Recorrendo ao JSON para {fs_name} pois o CSV não foi carregado ou está vazio.")
        df_cv_results = load_data_from_json_summary(fs_name)

    if df_cv_results.empty:
        print(f"Nenhum dado de resultado de CV encontrado para {fs_name} (nem CSV, nem JSON). Pulando.")
        continue

    fs_output_dir = os.path.join(OUTPUT_PLOTS_DIR, fs_name)
    os.makedirs(fs_output_dir, exist_ok=True)

    print(f"Colunas disponíveis para {fs_name} após carregamento e processamento: {df_cv_results.columns.tolist()}")

    # Garante que as colunas de score e tempo são numéricas
    if 'mean_test_score' in df_cv_results.columns:
        df_cv_results['mean_test_score'] = pd.to_numeric(df_cv_results['mean_test_score'], errors='coerce')
    if 'mean_fit_time' in df_cv_results.columns:
        df_cv_results['mean_fit_time'] = pd.to_numeric(df_cv_results['mean_fit_time'], errors='coerce')


    # 1. Gráficos de Sensibilidade
    for param_key in PARAMS_TO_ANALYZE_PROCESSED['numeric']:
        if param_key in df_cv_results.columns:
            hue_opt_key = 'model__optimizer_name' if 'model__optimizer_name' in df_cv_results.columns and param_key != 'model__optimizer_name' else None
            plot_sensitivity(df_cv_results.copy(), param_key, fs_name, fs_output_dir, hue_param=hue_opt_key)
        else:
            print(f"Parâmetro numérico '{param_key}' não encontrado no DataFrame para {fs_name} para plot de sensibilidade.")

    # 2. Boxplots
    for param_key in PARAMS_TO_ANALYZE_PROCESSED['categorical']:
        if param_key in df_cv_results.columns:
            plot_boxplot_comparison(df_cv_results.copy(), param_key, fs_name, fs_output_dir)
        else:
            print(f"Parâmetro categórico '{param_key}' não encontrado no DataFrame para {fs_name} para boxplot.")

    # 3. Heatmaps
    param_pairs_for_heatmap = [
        ('model__learning_rate', 'model__dropout_rate1'),
        ('epochs', 'batch_size'),
        ('model__optimizer_name', 'model__learning_rate') 
    ]
    for p1, p2 in param_pairs_for_heatmap:
        if p1 in df_cv_results.columns and p2 in df_cv_results.columns:
            plot_heatmap_interaction(df_cv_results.copy(), p1, p2, fs_name, fs_output_dir)
        else:
            print(f"Skipping heatmap for {p1} vs {p2} for {fs_name}: um ou ambos os parâmetros não estão no DataFrame.")
            
    if 'model__kernel_regularizer_type' in df_cv_results.columns and \
       'model__kernel_regularizer_strength' in df_cv_results.columns:
        plot_heatmap_interaction(df_cv_results.copy(), 
                                 'model__kernel_regularizer_type', 
                                 'model__kernel_regularizer_strength', 
                                 f"{fs_name}", 
                                 fs_output_dir) # Nome do arquivo será heatmap_type_vs_strength
    else:
        print(f"Não foi possível gerar heatmap de regularização para {fs_name}: colunas ausentes.")


    # 4. Gráfico Score vs. Tempo de Treino
    plot_score_vs_time(df_cv_results.copy(), fs_name, fs_output_dir)

print(f"\nAnálise gráfica concluída. Verifique o diretório: {os.path.abspath(OUTPUT_PLOTS_DIR)}")

