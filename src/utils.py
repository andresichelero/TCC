# src/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from scipy.signal import welch # Para o espectro de frequência
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns 
import json

# Flag global para controlar salvamento de plots (para não mostrar interativamente em execuções longas)
# E diretório para salvar
SAVE_PLOTS = True # Mudar para False se quiser ver interativamente (pode pausar o script)
PLOTS_DIR = "results/plots"
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def visualize_knn_decision_boundary(
    X_train_all_features,
    y_train,
    selected_features_vector,
    class_names=None,
    title="Fronteira de Decisão KNN",
    filename="knn_decision_boundary.png",
):

    selected_indices = np.where(selected_features_vector == 1)[0]
    if len(selected_indices) < 2:
        print("Visualização da fronteira de decisão requer pelo menos 2 features.")
        return

    # 1. Seleciona e Padroniza as features
    X_selected = X_train_all_features[:, selected_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 2. Reduz para 2D com PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # 3. Treina um novo KNN nos dados 2D
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_2d, y_train)

    # 4. Cria um meshgrid para plotar a fronteira
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 5. Plota
    fig = plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_train, s=20, edgecolor="k", cmap=plt.cm.RdYlBu
    )

    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.title(title)

    if class_names:
        legend1 = plt.legend(
            handles=scatter.legend_elements()[0], labels=class_names, title="Classes"
        )
        plt.gca().add_artist(legend1)

    _handle_plot(fig, filename, title)


def plot_optimization_diagnostics(
    curves_dict, title="Diagnóstico da Otimização", filename="opt_diagnostics.png"
):
    num_plots = len(curves_dict)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]

    iterations = range(len(next(iter(curves_dict.values()))))

    for ax, (metric_name, curve_data) in zip(axs, curves_dict.items()):
        ax.plot(iterations, curve_data, marker=".")
        ax.set_title(f"Evolução de '{metric_name}' do Melhor Agente")
        ax.set_ylabel(metric_name)
        ax.grid(True)

    axs[-1].set_xlabel("Iteração")
    plt.tight_layout()
    _handle_plot(fig, filename, title)


def calculate_specificity(y_true, y_pred, class_label, num_classes):
    """
    Calcula a especificidade para uma classe específica.
    Especificidade = TN / (TN + FP)
    Args:
        y_true (np.ndarray): Rótulos verdadeiros.
        y_pred (np.ndarray): Rótulos previstos.
        class_label (int): O rótulo da classe para a qual calcular a especificidade (0, 1, 2,...).
        num_classes (int): Número total de classes.
    Returns:
        float: Valor da especificidade.
    """
    # Cria uma matriz de confusão para a classe específica vs. todas as outras
    # Convertendo para um problema binário para a classe de interesse
    y_true_binary = (y_true == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)

    # Matriz de confusão para o problema binário da classe_label
    # [[TN, FP],
    # TN aqui significa que uma amostra que NÃO é da classe_label foi corretamente classificada como NÃO sendo da classe_label.
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    if cm_binary.shape == (2,2): # Garante que há ambas as classes (positiva e negativa para class_label)
        tn = cm_binary[0, 0]  # Verdadeiros Negativos para a classe_label
        fp = cm_binary[0, 1]  # Falsos Positivos para a classe_label
    elif cm_binary.shape == (1,1) and np.all(y_true_binary == 0) and np.all(y_pred_binary == 0): # Todas são negativas para a classe e previstas como negativas
        tn = cm_binary[0,0]
        fp = 0
    elif cm_binary.shape == (1,1) and np.all(y_true_binary == 1) and np.all(y_pred_binary == 1): # Todas são positivas para a classe e previstas como positivas (TN e FP são 0)
        tn = 0
        fp = 0
    else: # Caso inesperado ou dados insuficientes
        print(f"Warning: Could not calculate specificity reliably for class {class_label} due to CM shape: {cm_binary.shape}")
        return np.nan

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return specificity

def calculate_all_metrics(y_true, y_pred, class_names=None):
    """
    Calcula e imprime acurácia, relatório de classificação e especificidade por classe.
    Args:
        y_true (np.ndarray): Rótulos verdadeiros.
        y_pred (np.ndarray): Rótulos previstos.
        class_names (list, optional): Nomes das classes para o relatório.
    Returns:
        dict: Dicionário contendo as métricas.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist() # Convertendo para lista para fácil serialização
    }

    print(f"\nMatriz de Confusão:\n{cm}")
    print(f"\nAcurácia Geral: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


    # Determina o número de classes a partir dos rótulos (ou usa len(class_names) se fornecido)
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_unique_classes = len(unique_labels)
    if class_names:
        num_classes_from_names = len(class_names)
        # Adicionar lógica para usar a mais confiável ou alertar sobre inconsistência

    specificities = {}
    print("Especificidade por classe:", flush=True)
    
    # Itera sobre as classes que realmente existem nos dados y_true ou nos nomes fornecidos
    # Se class_names é ["N","I","P"], classes são 0,1,2
    if class_names:
        target_classes_for_specificity = range(len(class_names))
    else: # Se não há nomes, usa as classes únicas nos dados
        target_classes_for_specificity = sorted(unique_labels)


    for class_val in target_classes_for_specificity:
        # class_val será 0, 1, 2...
        spec = calculate_specificity(y_true, y_pred, class_label=class_val, num_classes=num_unique_classes) # num_classes é o total para contexto
        class_name_str = class_names[class_val] if class_names and class_val < len(class_names) else f"Classe {class_val}"
        print(f"  - {class_name_str}: {spec:.4f}", flush=True)
        specificities[f"specificity_{class_name_str.replace(' ', '_').replace('(', '').replace(')', '')}"] = spec # Nome de chave mais limpo
    
    metrics["specificities"] = specificities
    return metrics

# Função auxiliar para salvar ou mostrar plot
def _handle_plot(fig, filename, title=""):
    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, filename)
        fig.savefig(filepath)
        print(f"Plot salvo em: {filepath}", flush=True)
        plt.close(fig) # Fecha a figura para liberar memória
    else:
        if title: fig.suptitle(title, fontsize=16) # Adiciona título se for mostrar interativamente
        plt.show()

def plot_eeg_segments(segments_dict, fs=173.61, n_segments_to_plot=1, base_filename="eeg_segment"):
    """
    Plota exemplos de segmentos EEG e seus espectros de frequência.
    Args:
        segments_dict (dict): Dicionário como {'Raw': raw_data, 'Processed': processed_data}
                              onde cada data é (n_samples, n_features_time_domain).
        fs (float): Frequência de amostragem.
        n_segments_to_plot (int): Número de segmentos de exemplo para plotar.
        base_filename (str): Nome base para salvar os arquivos de plot.
    """
    if not segments_dict:
        return

    num_types = len(segments_dict)
    keys = list(segments_dict.keys())
    
    # Garante que há dados suficientes para plotar
    min_available_segments = min(data.shape[0] for data in segments_dict.values() if data is not None and data.ndim > 1)
    if min_available_segments == 0:
        print("Aviso: Nenhum segmento disponível para plotar em plot_eeg_segments.")
        return

    segments_to_plot_actual = min(n_segments_to_plot, min_available_segments)

    for i in range(segments_to_plot_actual):
        fig, axs = plt.subplots(num_types, 2, figsize=(15, 5 * num_types))
        if num_types == 1: # Ajustar axs se for apenas um tipo de dado
            axs = np.array([axs]) 

        for row_idx, key in enumerate(keys):
            data_array = segments_dict[key]
            if data_array is None or data_array.shape[0] <= i:
                if axs[row_idx, 0]: axs[row_idx, 0].set_title(f"{key} - Segmento {i} (Dados Indisponíveis)")
                if axs[row_idx, 1]: axs[row_idx, 1].set_title(f"Espectro {key} - Segmento {i} (Dados Indisponíveis)")
                continue

            segment = data_array[i, :]
            time_vector = np.arange(segment.size) / fs

            # Plot do sinal no domínio do tempo
            axs[row_idx, 0].plot(time_vector, segment)
            axs[row_idx, 0].set_title(f"{key} - Segmento {i} (Domínio do Tempo)")
            axs[row_idx, 0].set_xlabel("Tempo (s)")
            axs[row_idx, 0].set_ylabel("Amplitude")
            axs[row_idx, 0].grid(True)

            # Plot do espectro de frequência (usando Welch para PSD)
            frequencies, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
            axs[row_idx, 1].semilogy(frequencies, psd) # Escala log para o PSD
            axs[row_idx, 1].set_title(f"Espectro {key} - Segmento {i} (PSD via Welch)")
            axs[row_idx, 1].set_xlabel("Frequência (Hz)")
            axs[row_idx, 1].set_ylabel("PSD (V^2/Hz)")
            axs[row_idx, 1].grid(True)
            axs[row_idx, 1].set_xlim(0, fs / 2) # Mostrar até Nyquist

        plt.tight_layout()
        _handle_plot(fig, f"{base_filename}_example_{i}.png", f"Exemplo de Segmento EEG {i}")

def plot_swt_coefficients(coeffs_map, segment_idx=0, base_filename="swt_coeffs"):
    """
    Plota os coeficientes SWT (Aproximação e Detalhes) para um segmento.
    Args:
        coeffs_map (dict): Dicionário como {'A4': cA4_array, 'D4': cD4_array, ...}
        segment_idx (int): Índice do segmento original (para nome do arquivo/título).
        base_filename (str): Nome base para salvar os arquivos de plot.
    """
    if not coeffs_map:
        return
    
    band_names = list(coeffs_map.keys())
    num_bands_to_plot = len(band_names)
    if num_bands_to_plot == 0:
        return

    fig, axs = plt.subplots(num_bands_to_plot, 1, figsize=(12, 3 * num_bands_to_plot), sharex=True)
    if num_bands_to_plot == 1: # Ajustar axs se for apenas uma banda
            axs = np.array([axs])

    for idx, band_name in enumerate(band_names):
        coeffs = coeffs_map[band_name]
        if coeffs is not None and isinstance(coeffs, np.ndarray) and coeffs.ndim > 0 and coeffs.size > 0:
            axs[idx].plot(coeffs)
            axs[idx].set_title(f"Coeficientes {band_name}")
            axs[idx].set_ylabel("Amplitude")
        else:
            axs[idx].set_title(f"Coeficientes {band_name} (Indisponível ou Escalar)")
    
    axs[-1].set_xlabel("Amostra do Coeficiente")
    plt.tight_layout()
    _handle_plot(fig, f"{base_filename}_segment_{segment_idx}.png", f"Coeficientes SWT - Segmento {segment_idx}")

def plot_dnn_training_history(history, title="Histórico de Treinamento DNN", filename="dnn_training_history.png"):
    """Plota o histórico de treinamento de um modelo Keras (loss e accuracy)."""
    if not history:
        print("Nenhum histórico de treinamento para plotar.", flush=True)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    if 'loss' in history and 'val_loss' in history:
        ax1.plot(history['loss'], label='Loss Treino')
        ax1.plot(history['val_loss'], label='Loss Validação')
        ax1.set_title('Loss do Modelo')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Época')
        ax1.legend(loc='upper right')
        ax1.grid(True)
    else:
        ax1.set_title('Dados de Loss Indisponíveis')

    # Plot Accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        ax2.plot(history['accuracy'], label='Acurácia Treino')
        ax2.plot(history['val_accuracy'], label='Acurácia Validação')
        ax2.set_title('Acurácia do Modelo')
        ax2.set_ylabel('Acurácia')
        ax2.set_xlabel('Época')
        ax2.legend(loc='lower right')
        ax2.grid(True)
    else:
        ax2.set_title('Dados de Acurácia Indisponíveis')

    plt.tight_layout()
    _handle_plot(fig, filename, title)

def plot_final_metrics_comparison_bars(results_dict, base_filename="final_metrics"):
    """
    Cria gráficos de barras comparando as métricas finais (Acurácia, Precisão, Recall, F1, Especificidade)
    para diferentes pipelines (e.g., BDA+DNN, BPSO+DNN).
    Args:
        results_dict (dict): Dicionário como {'BDA+DNN': metrics_bda, 'BPSO+DNN': metrics_bpso}
                             onde metrics_* é o dicionário retornado por calculate_all_metrics.
    """
    if not results_dict:
        return

    pipelines = list(results_dict.keys())
    if not pipelines: return

    # Métricas gerais
    overall_metrics = ['accuracy'] # F1-macro será adicionado
    metric_values_overall = {m: [] for m in overall_metrics + ['f1_macro']}

    # Métricas por classe (assumindo 3 classes: Normal, Interictal, Ictal)
    class_labels = ["Normal (0)", "Interictal (1)", "Ictal (2)"] # Deve corresponder ao classification_report
    per_class_metrics = ['precision', 'recall', 'f1-score', 'specificity']
    metric_values_per_class = {f"{cls_label}_{m}": [] for cls_label in class_labels for m in per_class_metrics}
    
    num_selected_features_list = []

    for pipe_name in pipelines:
        res = results_dict.get(pipe_name)
        if not res: continue # Pula se não houver resultado para este pipeline

        num_selected_features_list.append(res.get('num_selected_features', 0))

        # Métricas gerais
        metric_values_overall['accuracy'].append(res.get('accuracy', 0) * 100)
        if 'classification_report' in res and isinstance(res['classification_report'], dict):
            metric_values_overall['f1_macro'].append(res['classification_report'].get('macro avg', {}).get('f1-score', 0) * 100)
        else:
            metric_values_overall['f1_macro'].append(0)


        # Métricas por classe
        for cls_idx, cls_label_key in enumerate(class_labels):
            if 'classification_report' in res and isinstance(res['classification_report'], dict):
                class_report_data = res['classification_report'].get(cls_label_key, {})
                metric_values_per_class[f"{cls_label_key}_precision"].append(class_report_data.get('precision', 0) * 100)
                metric_values_per_class[f"{cls_label_key}_recall"].append(class_report_data.get('recall', 0) * 100)
                metric_values_per_class[f"{cls_label_key}_f1-score"].append(class_report_data.get('f1-score', 0) * 100)
            else: # Preenche com 0 se não houver dados
                metric_values_per_class[f"{cls_label_key}_precision"].append(0)
                metric_values_per_class[f"{cls_label_key}_recall"].append(0)
                metric_values_per_class[f"{cls_label_key}_f1-score"].append(0)

            # Especificidade
            if 'specificities' in res and isinstance(res['specificities'], dict):
                 metric_values_per_class[f"{cls_label_key}_specificity"].append(res['specificities'].get(f"specificity_class_{cls_idx}", 0) * 100)
            else:
                 metric_values_per_class[f"{cls_label_key}_specificity"].append(0)


    x = np.arange(len(pipelines))  # Posições dos labels

    # Plot Acurácia e F1-Macro
    fig_overall, ax_overall = plt.subplots(figsize=(10, 6))
    width = 0.35
    rects1 = ax_overall.bar(x - width/2, metric_values_overall['accuracy'], width, label='Acurácia (%)')
    rects2 = ax_overall.bar(x + width/2, metric_values_overall['f1_macro'], width, label='F1-Score Macro (%)')
    ax_overall.set_ylabel('Pontuação (%)')
    ax_overall.set_title('Comparação de Acurácia Geral e F1-Score Macro')
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(pipelines)
    ax_overall.legend()
    ax_overall.bar_label(rects1, padding=3, fmt='%.2f')
    ax_overall.bar_label(rects2, padding=3, fmt='%.2f')
    ax_overall.set_ylim(0, 105)
    plt.tight_layout()
    _handle_plot(fig_overall, f"{base_filename}_overall_accuracy_f1.png", "Acurácia e F1-Macro")

    # Plot Métricas por Classe (Recall/Sensibilidade como exemplo, pode fazer para outras)
    # Recall (Sensibilidade)
    fig_recall, ax_recall = plt.subplots(figsize=(12, 7))
    width_recall = 0.25
    offsets = np.array([-width_recall, 0, width_recall]) # Para 3 classes
    
    for i, cls_label_key in enumerate(class_labels):
        recalls = metric_values_per_class[f"{cls_label_key}_recall"]
        rects = ax_recall.bar(x + offsets[i], recalls, width_recall, label=f'Recall {cls_label_key}')
        ax_recall.bar_label(rects, padding=3, fmt='%.2f')

    ax_recall.set_ylabel('Recall / Sensibilidade (%)')
    ax_recall.set_title('Comparação de Recall (Sensibilidade) por Classe')
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(pipelines)
    ax_recall.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax_recall.set_ylim(0, 105)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta para a legenda não cortar
    _handle_plot(fig_recall, f"{base_filename}_recall_per_class.png", "Recall por Classe")

    # Plot Número de Features Selecionadas
    fig_nfeat, ax_nfeat = plt.subplots(figsize=(8, 5))
    rects_nfeat = ax_nfeat.bar(pipelines, num_selected_features_list, color=['skyblue', 'lightcoral'])
    ax_nfeat.set_ylabel('Número de Features Selecionadas')
    ax_nfeat.set_title('Número de Features Selecionadas por Algoritmo')
    ax_nfeat.bar_label(rects_nfeat, padding=3)
    plt.tight_layout()
    _handle_plot(fig_nfeat, f"{base_filename}_num_features.png", "Número de Features")

def plot_convergence_curves(curves, labels, title="Curvas de Convergência", filename="optimizers_convergence.png"):
    """
    Plota múltiplas curvas de convergência.
    Args:
        curves (list of np.ndarray): Lista de arrays, cada um sendo uma curva de convergência.
        labels (list of str): Lista de rótulos para cada curva.
        title (str): Título do gráfico.
    """
    fig = plt.figure(figsize=(10, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.legend()
    plt.grid(True)
    #plt.show()
    _handle_plot(fig, filename, title) # Usa a função auxiliar

def plot_data_distribution_pca(X_dict, y_dict, title="Distribuição dos Dados (PCA)", filename="data_distribution_pca.png", class_names=None):
    """
    Plota a distribuição dos diferentes conjuntos de dados (treino, val, teste)
    após aplicar PCA para redução a 2D.
    Args:
        X_dict (dict): Dicionário de matrizes de features. Ex: {'Treino': X_train_feat, 'Validação': X_val_feat, 'Teste': X_test_feat}
        y_dict (dict): Dicionário de rótulos correspondentes. Ex: {'Treino': y_train, 'Validação': y_val, 'Teste': y_test}
        title (str): Título do gráfico.
        filename (str): Nome do arquivo para salvar o plot.
        class_names (list, optional): Nomes das classes para a legenda.
    """
    if not X_dict or not y_dict:
        print("Dicionários X ou y estão vazios. Não é possível plotar a distribuição.")
        return

    combined_X = []
    combined_y = []
    set_labels = [] # Para diferenciar treino, val, teste no plot

    print(f"Plotando distribuição para os conjuntos: {list(X_dict.keys())}")

    for set_name, X_data in X_dict.items():
        if X_data is None or X_data.shape[0] == 0:
            print(f"Aviso: Conjunto '{set_name}' está vazio ou é None. Pulando.")
            continue
        if y_dict.get(set_name) is None or len(y_dict[set_name]) != X_data.shape[0]:
            print(f"Aviso: Rótulos para o conjunto '{set_name}' estão ausentes ou com tamanho incorreto. Pulando.")
            continue
            
        combined_X.append(X_data)
        combined_y.append(y_dict[set_name])
        set_labels.extend([set_name] * X_data.shape[0])

    if not combined_X:
        print("Nenhum dado válido para combinar e plotar.")
        return

    X_all = np.vstack(combined_X)
    y_all = np.concatenate(combined_y)
    
    # 1. Padronizar os dados antes do PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 2. Aplicar PCA
    if X_scaled.shape[1] < 2:
        print("Número de features menor que 2, não é possível aplicar PCA para 2D. Plotando 1D se possível.")
        if X_scaled.shape[1] == 1:
            pca_result = X_scaled
            df_pca = pd.DataFrame(data={'PC1': pca_result.flatten(), 'label': y_all, 'set': set_labels})
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.stripplot(x='PC1', y='set', hue='label', data=df_pca, ax=ax, jitter=True, dodge=True, palette='viridis')
            ax.set_title(f'{title} (1D)')
            if class_names:
                handles, labels = ax.get_legend_handles_labels()
                # Mapear rótulos numéricos para nomes de classes
                new_labels = [class_names[int(label)] if label.isdigit() and int(label) < len(class_names) else label for label in labels]
                ax.legend(handles, new_labels, title='Classe')
            else:
                 ax.legend(title='Classe')
            plt.tight_layout()
            _handle_plot(fig, filename, title)
        return

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df_pca['label'] = y_all
    df_pca['set'] = set_labels # Adiciona a identificação do conjunto (Treino, Val, Teste)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"PCA: Variância explicada pelos 2 componentes principais: {explained_variance_ratio.sum()*100:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 9))
    
    scatter_plot = sns.scatterplot(
        x="PC1", y="PC2",
        hue="label",
        style="set",# Marcador por conjunto (Treino, Val, Teste)
        data=df_pca,
        palette="plasma", # esquema de cores (ex: viridis, plasma, coolwarm)
        s=50,# Tamanho dos marcadores
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel(f'Componente Principal 1 ({explained_variance_ratio[0]*100:.2f}% variância)')
    ax.set_ylabel(f'Componente Principal 2 ({explained_variance_ratio[1]*100:.2f}% variância)')
    ax.grid(True)

    handles, labels = scatter_plot.get_legend_handles_labels()
    num_classes = len(np.unique(y_all))
    num_sets = len(np.unique(set_labels))
    hue_handles = handles[1:num_classes+1]
    hue_labels_orig = labels[1:num_classes+1]
    style_handles = handles[num_classes+2 : num_classes+2+num_sets]
    style_labels = labels[num_classes+2 : num_classes+2+num_sets]

    if class_names:
        final_hue_labels = [class_names[int(lbl)] if lbl.isdigit() and int(lbl) < len(class_names) else lbl for lbl in hue_labels_orig]
    else:
        final_hue_labels = hue_labels_orig
        
    leg1 = ax.legend(hue_handles, final_hue_labels, title='Classe', loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg1)
    
    if style_handles and style_labels:
        ax.legend(style_handles, style_labels, title='Conjunto', loc='lower left', bbox_to_anchor=(1.01, 0))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    _handle_plot(fig, filename, title)

if __name__ == '__main__':
    y_true_ex = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2, 2])
    y_pred_ex = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2, 2, 2])
    class_names_ex = ["Normal (0)", "Interictal (1)", "Ictal (2)"]
    print("--- Testando calculate_all_metrics ---")
    metrics_results = calculate_all_metrics(y_true_ex, y_pred_ex, class_names=class_names_ex)
    print("\nMétricas retornadas (dict):")
    import json
    print(json.dumps(metrics_results, indent=2))
    curve1 = np.array([10, 8, 6, 5, 4.5, 4, 3.9])
    curve2 = np.array([12, 10, 7, 6, 5.5, 5, 4.8])
    print("\n--- Testando plot_convergence_curves ---")
    plot_convergence_curves([curve1, curve2], ["Algoritmo A", "Algoritmo B"], title="Teste de Convergência")
