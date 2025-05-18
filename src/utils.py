# src/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

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
    #  [FN, TP]] (onde TP é para a classe_label ser corretamente prevista)
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


    num_classes = len(np.unique(np.concatenate((y_true,y_pred)))) # Determina o número de classes a partir dos dados
    specificities = {}
    print("Especificidade por classe:")
    for i in range(num_classes):
        spec = calculate_specificity(y_true, y_pred, class_label=i, num_classes=num_classes)
        class_name_str = class_names[i] if class_names and i < len(class_names) else f"Classe {i}"
        print(f"  - {class_name_str}: {spec:.4f}")
        specificities[f"specificity_class_{i}"] = spec
    
    metrics["specificities"] = specificities
    return metrics


def plot_convergence_curves(curves, labels, title="Curvas de Convergência"):
    """
    Plota múltiplas curvas de convergência.
    Args:
        curves (list of np.ndarray): Lista de arrays, cada um sendo uma curva de convergência.
        labels (list of str): Lista de rótulos para cada curva.
        title (str): Título do gráfico.
    """
    plt.figure(figsize=(10, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


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