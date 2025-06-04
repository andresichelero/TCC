# src/dnn_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def build_dnn_model(num_selected_features, num_classes=3, jit_compile_dnn=False):
    """
    Constrói e compila o modelo DNN (MLP) final.
    Conforme o artigo: 3 camadas ocultas, cada uma com 10 neurônios, ativação sigmoidal.
    Args:
        num_selected_features (int): Número de características de entrada.
        num_classes (int): Número de classes de saída (default é 3).
        jit_compile_dnn (bool): Se deve compilar com XLA.
    Returns:
        keras.Model: Modelo Keras compilado.
    """
    if num_selected_features <= 0:
        raise ValueError(
            "O número de características selecionadas deve ser maior que zero."
        )

    model = keras.Sequential(name="MLP_Classifier_Final")
    model.add(layers.Input(shape=(num_selected_features,), name="Input_Layer"))

    # Arquitetura conforme o artigo: 3 camadas ocultas, 10 neurônios cada, sigmoidal
    # O uso de BatchNormalization antes da ativação é uma prática comum e pode ajudar.
    # O artigo não detalha BN, mas é uma adição razoável para estabilidade.
    # Dropout também é uma adição para regularização. O artigo não especifica dropout.
    # Para ser estritamente fiel, poderia remover BN e Dropout se causar problemas ou
    # se os resultados não baterem, mas geralmente são benéficos.

    # Camada Oculta 1
    model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_1_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_1_BN"))
    model.add(layers.Activation("sigmoid", name="Hidden_Layer_1_Sigmoid"))
    model.add(
        layers.Dropout(0.1, name="Hidden_Layer_1_Dropout")
    )  # Dropout pode ser ajustado ou removido

    # Camada Oculta 2
    model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_2_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_2_BN"))
    model.add(layers.Activation("sigmoid", name="Hidden_Layer_2_Sigmoid"))
    model.add(layers.Dropout(0.1, name="Hidden_Layer_2_Dropout"))

    # Camada Oculta 3
    model.add(layers.Dense(10, use_bias=False, name="Hidden_Layer_3_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_3_BN"))
    model.add(layers.Activation("sigmoid", name="Hidden_Layer_3_Sigmoid"))
    model.add(layers.Dropout(0.1, name="Hidden_Layer_3_Dropout"))

    # Camada de Saída
    model.add(layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

    # O otimizador Adam é uma escolha comum. O artigo não especifica o otimizador para a DNN.
    opt = Adam()  # Learning rate padrão
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",  # Adequado para classificação multiclasse com rótulos inteiros
        metrics=["accuracy"],
        jit_compile=jit_compile_dnn,
    )

    return model


if __name__ == "__main__":
    # Exemplo de uso
    num_features_exemplo = 20  # Suponha que 20 features foram selecionadas
    try:
        print(
            f"\nConstruindo modelo DNN final com {num_features_exemplo} features de entrada:"
        )
        dnn_model_exemplo = build_dnn_model(num_features_exemplo, num_classes=3)
        dnn_model_exemplo.summary()

        print(
            "\nVerificando se TensorFlow está usando GPU:",
            tf.config.list_physical_devices("GPU"),
        )

    except ValueError as ve:
        print(f"Erro: {ve}")
