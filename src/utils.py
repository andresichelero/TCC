# src/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from scipy.signal import welch # Para o espectro de frequência

# Flag global para controlar salvamento de plots (para não mostrar interativamente em execuções longas)
# E diretório para salvar
SAVE_PLOTS = True # Mude para False se quiser ver interativamente (pode pausar o script)
PLOTS_DIR = "results/plots"

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