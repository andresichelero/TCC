# Definição do modelo DNN
# src/dnn_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

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
    use_relu = False # Usar activation relu ou sigmoid (tendo problemas com sigmoid)
    if use_relu:
        model.add(layers.Dense(32, activation='relu', name="Hidden_Layer_1"))
        model.add(layers.Dropout(0.3)) # Adicionar Dropout para regularização
        model.add(layers.Dense(32, activation='relu', name="Hidden_Layer_2"))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(16, activation='relu', name="Hidden_Layer_3"))
    else:
        # Arquitetura do artigo com Sigmoid - e adição de BatchNormalization para tentar melhorar as curvas de aprendizado
        # 3 camadas ocultas, 10 neurônios cada, ativação sigmoide
        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense")) # use_bias=False quando BN é usada logo após
        model.add(layers.BatchNormalization(name="Hidden_Layer_1_BN"))
        model.add(layers.Activation('sigmoid', name="Hidden_Layer_1_Sigmoid"))
        model.add(layers.Dropout(0.1, name="Hidden_Layer_1_Dropout"))

        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_2_Dense"))
        model.add(layers.BatchNormalization(name="Hidden_Layer_2_BN"))
        model.add(layers.Activation('sigmoid', name="Hidden_Layer_2_Sigmoid"))
        model.add(layers.Dropout(0.1, name="Hidden_Layer_2_Dropout"))

        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_3_Dense"))
        model.add(layers.BatchNormalization(name="Hidden_Layer_3_BN"))
        model.add(layers.Activation('sigmoid', name="Hidden_Layer_3_Sigmoid"))
        model.add(layers.Dropout(0.1, name="Hidden_Layer_3_Dropout"))

    model.add(layers.Dense(num_classes, activation='softmax', name="Output_Layer"))
    opt = Adam(learning_rate=0.0009)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile_dnn,
    )

    ##model.summary()
    return model

if __name__ == '__main__':
    # Exemplo de uso
    num_features_exemplo = 25 # Suponha que 25 features foram selecionadas
    try:
        dnn_model_exemplo = build_dnn_model(num_features_exemplo)
        print(f"\nModelo DNN criado com {num_features_exemplo} features de entrada:")
        dnn_model_exemplo.summary()

        # Teste com 0 features
        # build_dnn_model(0) # Deve levantar ValueError
    except ValueError as ve:
        print(f"Erro: {ve}")

    # Verificar se TensorFlow está usando GPU
    print("\nGPUs Disponíveis para TensorFlow:", tf.config.list_physical_devices('GPU'))
