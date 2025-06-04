# src/fitness_function.py
import numpy as np
import gc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate_fitness(binary_feature_vector,
                     X_train_all_features,
                     y_train,
                     alpha, # Peso para a taxa de erro
                     beta,  # Peso para o número de características
                     verbose_level=0 # Para logs internos da função de fitness, se necessário
                     ):
    """
    Avalia a aptidão de um subconjunto de características binário usando KNN e validação cruzada.
    Menor valor de fitness é melhor.
    Args:
        binary_feature_vector (np.ndarray): Vetor binário de seleção (0s e 1s).
        X_train_all_features (np.ndarray): Matriz de características de treino completa.
        y_train (np.ndarray): Rótulos de treino.
        alpha (float): Peso para a taxa de erro na fórmula de fitness.
        beta (float): Peso para a razão de características na fórmula de fitness.
        verbose_level (int): Nível de verbosidade para logs internos.
    Returns:
        float: Valor de fitness (menor é melhor).
    """
    selected_indices = np.where(binary_feature_vector == 1)[0]
    num_selected = len(selected_indices)
    total_num_features_available = len(binary_feature_vector)

    # Penalidade se nenhuma característica for selecionada
    if num_selected == 0:
        # Retorna o pior fitness possível: erro máximo (1.0) e razão de features máxima (1.0)
        return alpha * 1.0 + beta * 1.0

    X_train_selected = X_train_all_features[:, selected_indices]

    # Verifica se, após a seleção, ainda temos alguma feature.
    # Isso é um pouco redundante com num_selected == 0, mas é uma boa verificação.
    if X_train_selected.shape[1] == 0:
        return alpha * 1.0 + beta * 1.0

    # Configuração do KNN e Validação Cruzada
    # O artigo menciona KNN com 10-fold cross-validation.
    # k=5 é um valor comum para n_neighbors, o artigo não especifica.
    knn = KNeighborsClassifier(n_neighbors=5)
    
    n_folds = 10
    # É crucial usar StratifiedKFold para problemas de classificação para manter a proporção das classes.
    # O dataset Bonn tem 100 amostras por classe (A, D, E).
    # Após o split inicial (e.g., 80/20), o X_train terá ~80 amostras de cada classe (se for balanceado).
    # O y_train para os otimizadores é derivado do X_train_p, que é 80% - VAL_SIZE do total.
    # Ex: Total 300 -> X_train_p (antes da extração de features) é ~ (300 * (1 - 0.20 - 0.15)) = 300 * 0.65 = 195.
    # Se balanceado, ~65 amostras por classe. Isso é suficiente para 10 folds.
    
    min_samples_per_class_in_ytrain = np.min(np.bincount(y_train))
    
    if min_samples_per_class_in_ytrain < n_folds:
        if verbose_level > 0:
            print(f"Fitness Warning: Smallest class in y_train has {min_samples_per_class_in_ytrain} samples, "
                  f"which is less than n_folds={n_folds}. Adjusting n_folds to {min_samples_per_class_in_ytrain}.")
        # Ajusta n_folds se for maior que o número de amostras na menor classe
        # A CV ainda requer pelo menos 2 amostras por classe para n_folds >= 2.
        if min_samples_per_class_in_ytrain < 2 : # Não é possível fazer CV
            if verbose_level > 0:
                print("Fitness Error: Smallest class has < 2 samples. Cannot perform CV. Returning max fitness.")
            return alpha * 1.0 + beta * 1.0
        n_folds = min_samples_per_class_in_ytrain


    # Usar random_state para reprodutibilidade da divisão da validação cruzada
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    try:
        # cross_val_score retorna um array de scores para cada fold.
        # O scoring padrão para classificadores é 'accuracy'.
        accuracies = cross_val_score(knn, X_train_selected, y_train, cv=cv_splitter, scoring='accuracy')
        accuracy = np.mean(accuracies)
    except ValueError as e:
        # Isso pode acontecer se, por exemplo, uma dobra da CV acabar com apenas uma classe.
        if verbose_level > 0:
            print(f"Fitness Error: ValueError during cross_val_score for KNN: {e}. "
                  f"Num selected features: {num_selected}. Returning max fitness.")
        return alpha * 1.0 + beta * 1.0 # Pior fitness

    error_rate = 1.0 - accuracy

    # Calcula a razão de características selecionadas
    feature_ratio = num_selected / total_num_features_available

    # Calcula o fitness final
    fitness = alpha * error_rate + beta * feature_ratio

    # Limpeza de memória (pode ser menos crítico com KNN do que com TensorFlow/Keras)
    del knn
    del X_train_selected
    gc.collect()

    return fitness

if __name__ == '__main__':
    # Exemplo de uso com dados dummy
    N_TRAIN_SAMPLES_DUMMY = 150 # Suficiente para 10-fold CV
    N_TOTAL_FEATURES_DUMMY = 45
    N_CLASSES_DUMMY = 3

    # Gerar dados dummy com distribuição de classes que permita 10-fold CV
    y_train_dummy = np.concatenate([
        np.zeros(N_TRAIN_SAMPLES_DUMMY // N_CLASSES_DUMMY, dtype=int),
        np.ones(N_TRAIN_SAMPLES_DUMMY // N_CLASSES_DUMMY, dtype=int),
        np.full(N_TRAIN_SAMPLES_DUMMY - 2 * (N_TRAIN_SAMPLES_DUMMY // N_CLASSES_DUMMY), 2, dtype=int)
    ])
    np.random.shuffle(y_train_dummy) # Embaralhar para garantir
    X_train_dummy = np.random.rand(N_TRAIN_SAMPLES_DUMMY, N_TOTAL_FEATURES_DUMMY)


    print(f"Dados dummy: X_train_dummy shape: {X_train_dummy.shape}, y_train_dummy shape: {y_train_dummy.shape}")
    print(f"Contagem de classes em y_train_dummy: {np.bincount(y_train_dummy)}")


    # Vetor de características binário de exemplo (selecionando as primeiras 10 features)
    example_binary_vector_knn = np.zeros(N_TOTAL_FEATURES_DUMMY, dtype=int)
    example_binary_vector_knn[:10] = 1
    print(f"\nExemplo de vetor binário (primeiras 10 selecionadas):\n{example_binary_vector_knn}")

    alpha_test = 0.99
    beta_test = 0.01

    print("\nAvaliando fitness com KNN (verbose_level=1)...")
    fitness_value_knn = evaluate_fitness(
        example_binary_vector_knn,
        X_train_dummy, y_train_dummy,
        alpha=alpha_test, beta=beta_test,
        verbose_level=1
    )
    print(f"\nFitness calculado com KNN: {fitness_value_knn:.4f}")

    print("\nTestando com nenhuma feature selecionada (KNN)...")
    no_feature_vector_knn = np.zeros(N_TOTAL_FEATURES_DUMMY, dtype=int)
    fitness_no_features_knn = evaluate_fitness(
        no_feature_vector_knn,
        X_train_dummy, y_train_dummy,
        alpha=alpha_test, beta=beta_test,
        verbose_level=1
    )
    # Esperado: alpha * 1.0 (erro max) + beta * 1.0 (ratio max, se num_selected/total for 1, mas aqui é 0)
    # Na verdade, se num_selected == 0, o ratio é 0, mas a penalidade é alpha*1 + beta*1
    print(f"Fitness com 0 features (KNN): {fitness_no_features_knn:.4f} (esperado: {alpha_test*1.0 + beta_test*1.0})")

    print("\nTestando com todas as features selecionadas (KNN)...")
    all_features_vector_knn = np.ones(N_TOTAL_FEATURES_DUMMY, dtype=int)
    fitness_all_features_knn = evaluate_fitness(
        all_features_vector_knn,
        X_train_dummy, y_train_dummy,
        alpha=alpha_test, beta=beta_test,
        verbose_level=1
    )
    print(f"Fitness com todas as features (KNN): {fitness_all_features_knn:.4f}")