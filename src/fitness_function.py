# Função de aptidão unificada
# src/fitness_function.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
try:
    from .dnn_model import build_dnn_model
except ImportError:
    from dnn_model import build_dnn_model


def evaluate_fitness(binary_feature_vector,
                     X_train_all_features, y_train,
                     X_val_all_features, y_val,
                     dnn_training_params={'epochs': 100, 'batch_size': 32, 'patience': 10},
                     alpha=0.99, beta=0.01, verbose=0):
    """
    Avalia a aptidão de um subconjunto de características binário.
    Menor valor de fitness é melhor.
    Args:
        binary_feature_vector (np.ndarray): Vetor binário de seleção (0s e 1s).
        X_train_all_features (np.ndarray): Matriz de características de treino completa.
        y_train (np.ndarray): Rótulos de treino.
        X_val_all_features (np.ndarray): Matriz de características de validação completa.
        y_val (np.ndarray): Rótulos de validação.
        dnn_training_params (dict): Parâmetros de treino da DNN (epochs, batch_size, patience).
        alpha (float): Peso para a taxa de erro.
        beta (float): Peso para o número de características.
        verbose (int): Nível de verbosidade para o treinamento do Keras (0, 1 ou 2).
    Returns:
        float: Valor de fitness (menor é melhor).
    """
    selected_indices = np.where(binary_feature_vector == 1)[0] # Obter índices
    num_selected = len(selected_indices)
    total_num_features_available = len(binary_feature_vector)

    # Penalidade se nenhuma característica for selecionada
    if num_selected == 0:
        return 1.0 * alpha + 1.0 * beta # Fitness máximo (pior) -> erro max + num_features max

    # Seleciona as características
    X_train_selected = X_train_all_features[:, selected_indices]
    X_val_selected = X_val_all_features[:, selected_indices]

    # Limpa a sessão Keras para evitar acúmulo de modelos/gráficos na memória
    tf.keras.backend.clear_session()

    # Constrói e compila a DNN
    model = build_dnn_model(num_selected_features=num_selected, num_classes=len(np.unique(y_train)),
                            jit_compile_dnn=False)

    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=dnn_training_params.get('patience', 10),
        restore_best_weights=True,
        verbose=verbose
    )

    # Treina a DNN
    history = model.fit(
        X_train_selected, y_train,
        epochs=dnn_training_params.get('epochs', 100),
        batch_size=dnn_training_params.get('batch_size', 32),
        validation_data=(X_val_selected, y_val),
        callbacks=[early_stopping],
        verbose=verbose # 0 para menos output durante treino da fitness
    )

    # Avalia no conjunto de validação (usando os melhores pesos restaurados pelo EarlyStopping)
    loss, accuracy = model.evaluate(X_val_selected, y_val, verbose=verbose)
    error_rate = 1.0 - accuracy

    # Calcula o fitness
    # Fitness = alpha * error_rate + beta * (num_selected / total_num_features)
    feature_ratio = num_selected / total_num_features_available
    fitness = alpha * error_rate + beta * feature_ratio

    # Libera memória do modelo explicitamente
    del model
    del history

    return fitness

if __name__ == '__main__':
    # Exemplo de uso com dados dummy
    N_TRAIN_SAMPLES = 100
    N_VAL_SAMPLES = 20
    N_TOTAL_FEATURES = 45
    N_CLASSES = 3

    # Gerar dados dummy
    X_train_dummy = np.random.rand(N_TRAIN_SAMPLES, N_TOTAL_FEATURES)
    y_train_dummy = np.random.randint(0, N_CLASSES, N_TRAIN_SAMPLES)
    X_val_dummy = np.random.rand(N_VAL_SAMPLES, N_TOTAL_FEATURES)
    y_val_dummy = np.random.randint(0, N_CLASSES, N_VAL_SAMPLES)

    # Vetor de características binário de exemplo (selecionando as primeiras 10 features)
    example_binary_vector = np.zeros(N_TOTAL_FEATURES, dtype=int)
    example_binary_vector[:10] = 1
    print(f"Exemplo de vetor binário (primeiras 10 selecionadas):\n{example_binary_vector}")

    dnn_params_test = {'epochs': 5, 'batch_size': 16, 'patience': 2} # Menos épocas para teste rápido

    print("\nAvaliando fitness (verbose=1 para output do Keras)...")
    fitness_value = evaluate_fitness(
        example_binary_vector,
        X_train_dummy, y_train_dummy,
        X_val_dummy, y_val_dummy,
        dnn_training_params=dnn_params_test,
        alpha=0.99, beta=0.01,
        verbose=1 # 0 para silenciar o output do Keras
    )
    print(f"\nFitness calculado: {fitness_value:.4f}")

    print("\nTestando com nenhuma feature selecionada...")
    no_feature_vector = np.zeros(N_TOTAL_FEATURES, dtype=int)
    fitness_no_features = evaluate_fitness(
        no_feature_vector,
        X_train_dummy, y_train_dummy,
        X_val_dummy, y_val_dummy,
        dnn_training_params=dnn_params_test
    )
    print(f"Fitness com 0 features: {fitness_no_features:.4f} (esperado: 0.99*1 + 0.01*1 = 1.0)")

    print("\nTestando com todas as features selecionadas...")
    all_features_vector = np.ones(N_TOTAL_FEATURES, dtype=int)
    fitness_all_features = evaluate_fitness(
        all_features_vector,
        X_train_dummy, y_train_dummy,
        X_val_dummy, y_val_dummy,
        dnn_training_params=dnn_params_test,
        verbose=1
    )
    print(f"Fitness com todas as features: {fitness_all_features:.4f}")