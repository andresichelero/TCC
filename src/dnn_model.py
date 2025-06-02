# src/dnn_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def build_dnn_model(num_selected_features, num_classes=3, jit_compile_dnn=False, dnn_config_id="original"):
    """
    Constrói e compila o modelo DNN (MLP).
    Args:
        num_selected_features (int): Número de características de entrada.
        num_classes (int): Número de classes de saída (default é 3).
        jit_compile_dnn (bool): Se deve usar compilação JIT.
        dnn_config_id (str): Identificador para diferentes configurações da DNN.
    Returns:
        keras.Model: Modelo Keras compilado.
    """
    if num_selected_features <= 0:
        raise ValueError("O número de características selecionadas deve ser maior que zero.")

    model = keras.Sequential(name=f"MLP_Classifier_{dnn_config_id}")
    model.add(layers.Input(shape=(num_selected_features,), name="Input_Layer"))

    print(f"Construindo DNN com config_id: {dnn_config_id}", flush=True)
    opt = None # Inicializa o otimizador

    if dnn_config_id == "original":
        # Arquitetura com Sigmoid e BatchNormalization
        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
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
        opt = Adam()
    elif dnn_config_id == "config_A":
        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
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
        opt = Adam(learning_rate=0.0005)
    elif dnn_config_id == "config_B":
        model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
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
        opt = Adam(learning_rate=0.005)
    else:
        raise ValueError(f"ID de configuração da DNN desconhecido: {dnn_config_id}")

    model.add(layers.Dense(num_classes, activation='softmax', name="Output_Layer"))

    if opt is None:
        print("AVISO: Otimizador não definido explicitamente para a config, usando Adam padrão.", flush=True)
        opt = Adam()

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile_dnn,
    )
    return model

if __name__ == '__main__':
    num_features_exemplo = 25
    try:
        print("\nTestando 'original' config:")
        dnn_model_orig = build_dnn_model(num_features_exemplo, dnn_config_id="original")
        dnn_model_orig.summary(print_fn=lambda x: print(x, flush=True))

        print("\nTestando 'config_A':")
        dnn_model_A = build_dnn_model(num_features_exemplo, dnn_config_id="config_A")
        dnn_model_A.summary(print_fn=lambda x: print(x, flush=True))

        print("\nTestando 'config_B':")
        dnn_model_B = build_dnn_model(num_features_exemplo, dnn_config_id="config_B")
        dnn_model_B.summary(print_fn=lambda x: print(x, flush=True))

    except ValueError as ve:
        print(f"Erro: {ve}", flush=True)

    print("\nGPUs Disponíveis para TensorFlow:", tf.config.list_physical_devices('GPU'), flush=True)