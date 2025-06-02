# src/dnn_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l1, l2

def build_dnn_model(num_selected_features,
                      num_classes=3,
                      jit_compile_dnn=False,
                      learning_rate=0.001,
                      optimizer_name='adam',
                      momentum=None, # Usado apenas se optimizer_name for 'sgd'
                      kernel_regularizer_type=None, # 'l1', 'l2', ou None
                      kernel_regularizer_strength=0.01, # Força do regularizador
                      dropout_rate1=0.1,
                      dropout_rate2=0.1,
                      dropout_rate3=0.1
                      ):
    """
    Constrói e compila o modelo DNN (MLP) com hiperparâmetros configuráveis.

    Args:
        num_selected_features (int): Número de características de entrada.
        num_classes (int): Número de classes de saída.
        jit_compile_dnn (bool): Se deve usar compilação JIT.
        learning_rate (float): Taxa de aprendizado para o otimizador.
        optimizer_name (str): Nome do otimizador ('adam', 'sgd', 'rmsprop').
        momentum (float, optional): Momento para o otimizador SGD.
        kernel_regularizer_type (str, optional): Tipo de regularizador de kernel ('l1', 'l2', None).
        kernel_regularizer_strength (float): Força do regularizador de kernel.
        dropout_rate1 (float): Taxa de dropout após a primeira camada densa.
        dropout_rate2 (float): Taxa de dropout após a segunda camada densa.
        dropout_rate3 (float): Taxa de dropout após a terceira camada densa.

    Returns:
        keras.Model: Modelo Keras compilado.
    """
    if num_selected_features <= 0:
        raise ValueError("O número de características selecionadas deve ser maior que zero.")

    # Configuração do regularizador de kernel
    regularizer = None
    if kernel_regularizer_type == 'l1':
        regularizer = l1(kernel_regularizer_strength)
    elif kernel_regularizer_type == 'l2':
        regularizer = l2(kernel_regularizer_strength)
    elif kernel_regularizer_type is not None:
        print(f"Aviso: Tipo de regularizador de kernel '{kernel_regularizer_type}' desconhecido. Nenhum regularizador será aplicado.", flush=True)


    model = keras.Sequential(name=f"MLP_Classifier_Opt-{optimizer_name}_LR-{learning_rate:.0e}_Reg-{kernel_regularizer_type}_Drop-{dropout_rate1:.1f}-{dropout_rate2:.1f}-{dropout_rate3:.1f}")
    model.add(layers.Input(shape=(num_selected_features,), name="Input_Layer"))

    # Camada Oculta 1
    model.add(layers.Dense(10, use_bias=False, kernel_regularizer=regularizer, name="Hidden_Layer_1_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_1_BN"))
    model.add(layers.Activation('sigmoid', name="Hidden_Layer_1_Sigmoid"))
    model.add(layers.Dropout(dropout_rate1, name="Hidden_Layer_1_Dropout"))

    # Camada Oculta 2
    model.add(layers.Dense(10, use_bias=False, kernel_regularizer=regularizer, name="Hidden_Layer_2_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_2_BN"))
    model.add(layers.Activation('sigmoid', name="Hidden_Layer_2_Sigmoid"))
    model.add(layers.Dropout(dropout_rate2, name="Hidden_Layer_2_Dropout"))

    # Camada Oculta 3
    model.add(layers.Dense(10, use_bias=False, kernel_regularizer=regularizer, name="Hidden_Layer_3_Dense"))
    model.add(layers.BatchNormalization(name="Hidden_Layer_3_BN"))
    model.add(layers.Activation('sigmoid', name="Hidden_Layer_3_Sigmoid"))
    model.add(layers.Dropout(dropout_rate3, name="Hidden_Layer_3_Dropout"))

    # Camada de Saída
    model.add(layers.Dense(num_classes, activation='softmax', name="Output_Layer"))

    # Configuração do otimizador
    if optimizer_name.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=momentum if momentum is not None else 0.0)
    elif optimizer_name.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Otimizador desconhecido: {optimizer_name}. Use 'adam', 'sgd', ou 'rmsprop'.")

    # Compilação do modelo
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile_dnn,
    )
    return model

if __name__ == '__main__':
    num_features_exemplo = 25
    print(f"\n--- Testando build_dnn_model com {num_features_exemplo} features ---")

    print("\n1. Testando com Adam (default), L2 reg, dropouts default (0.1):")
    model1 = build_dnn_model(
        num_selected_features=num_features_exemplo,
        kernel_regularizer_type='l2',
        kernel_regularizer_strength=0.005
    )
    model1.summary(print_fn=lambda x: print(x, flush=True))
    print(f"   Otimizador de model1: {model1.optimizer.get_config()}", flush=True)


    print("\n2. Testando com SGD, L1 reg, dropouts customizados:")
    model2 = build_dnn_model(
        num_selected_features=num_features_exemplo,
        optimizer_name='sgd',
        learning_rate=0.01,
        momentum=0.9,
        kernel_regularizer_type='l1',
        kernel_regularizer_strength=0.02,
        dropout_rate1=0.15,
        dropout_rate2=0.25,
        dropout_rate3=0.35
    )
    model2.summary(print_fn=lambda x: print(x, flush=True))
    print(f"   Otimizador de model2: {model2.optimizer.get_config()}", flush=True)


    print("\n3. Testando com RMSprop, sem regularizador, sem dropout (taxas = 0.0):")
    model3 = build_dnn_model(
        num_selected_features=num_features_exemplo,
        optimizer_name='rmsprop',
        learning_rate=0.0005,
        kernel_regularizer_type=None,
        dropout_rate1=0.0,
        dropout_rate2=0.0,
        dropout_rate3=0.0
    )
    model3.summary(print_fn=lambda x: print(x, flush=True))
    print(f"   Otimizador de model3: {model3.optimizer.get_config()}", flush=True)

    try:
        print("\n4. Testando com otimizador inválido:")
        build_dnn_model(num_features_exemplo, optimizer_name='invalid_optimizer_name_test')
    except ValueError as ve:
        print(f"   Erro esperado capturado: {ve}", flush=True)
    
    print("\n5. Testando com tipo de regularizador inválido (deve gerar aviso, mas não erro):")
    model5 = build_dnn_model(
        num_selected_features=num_features_exemplo,
        kernel_regularizer_type='l3_invalid',
        kernel_regularizer_strength=0.01
    )
    model5.summary(print_fn=lambda x: print(x, flush=True))


    print("\nGPUs Disponíveis para TensorFlow:", tf.config.list_physical_devices('GPU'), flush=True)
