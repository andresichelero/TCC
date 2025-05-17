# Definição do modelo DNN
# src/dnn_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_dnn_model(num_selected_features, num_classes=3, jit_compile_dnn=False):
    """
    Constrói e compila o modelo DNN (MLP).
    Args:
        num_selected_features (int): Número de características de entrada.
        num_classes (int): Número de classes de saída (default é 3).
    Returns:
        keras.Model: Modelo Keras compilado.
    """
    if num_selected_features <= 0:
        raise ValueError("O número de características selecionadas deve ser maior que zero.")

    model = keras.Sequential(name="MLP_Classifier")
    model.add(layers.Input(shape=(num_selected_features,), name="Input_Layer"))
    model.add(layers.Dense(10, activation='sigmoid', name="Hidden_Layer_1"))
    model.add(layers.Dense(10, activation='sigmoid', name="Hidden_Layer_2"))
    model.add(layers.Dense(10, activation='sigmoid', name="Hidden_Layer_3"))
    model.add(layers.Dense(num_classes, activation='softmax', name="Output_Layer"))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # Apropriado para rótulos inteiros
                  metrics=['accuracy'], jit_compile=jit_compile_dnn)

    ##model.summary()
    return model

if __name__ == '__main__':
    # Exemplo de uso
    num_features_exemplo = 25 # Suponha que 25 features foram selecionadas
    try:
        dnn_model_exemplo = build_dnn_model(num_features_exemplo)
        print(f"\nModelo DNN criado com {num_features_exemplo} features de entrada:")
        #dnn_model_exemplo.summary()

        # Teste com 0 features
        # build_dnn_model(0) # Deve levantar ValueError
    except ValueError as ve:
        print(f"Erro: {ve}")

    # Verificar se TensorFlow está usando GPU
    print("\nGPUs Disponíveis para TensorFlow:", tf.config.list_physical_devices('GPU'))