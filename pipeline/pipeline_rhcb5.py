# -*- coding: utf-8 -*-
"""
Pipeline RHCB5 (Baseado em rhcb5.py)

Este script foi refatorado para ser uma 'biblioteca' callable.
A lógica principal foi encapsulada na função run_rhcb5_pipeline.

Atualizações:
- Adicionada Data Augmentation (Time Shifting) para robustez.
- Arquitetura modificada para incluir uma Camada de Atenção (Attention Layer)
  após a BiLSTM para melhor interpretabilidade e performance.
- Adicionada plotagem XAI para os pesos da Camada de Atenção.

Ele depende de 'pipeline_utils.py' para:
- DataHandler (load, preprocess, split)
- Metrics (calculate_all_metrics)
- Plotting (plot_dnn_training_history, etc.)
- Constantes Globais (FS, HIGHCUT_HZ, etc.)
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, 
    Bidirectional, LSTM, Dense, Activation,
    Lambda, Softmax
)
# AdditiveAttention é a camada Bahdanau-style
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
import matplotlib.pyplot as plt

# Importa os utilitários compartilhados
try:
    import pipeline_utils
    from pipeline_utils import (
        DataHandler, Metrics, Plotting, NumpyEncoder,
        CLASS_NAMES, NUM_CLASSES, FS, HIGHCUT_HZ, FILTER_ORDER,
        TEST_SIZE, VAL_SIZE, TARGET_INPUT_LENGTH
    )
except ImportError:
    print("ERRO: Não foi possível importar 'pipeline_utils.py'. "
          "Certifique-se que o arquivo está no mesmo diretório.")
    sys.exit(1)

# --- Constantes Específicas do Pipeline RHCB5 ---
NUM_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE_RHCB5 = 15 # Paciência do EarlyStopping
AUGMENT_MAX_SHIFT_PERCENT = 0.2 # % máximo do sinal para "rolar" (Time Shift)

# Nível de verbosidade
VERBOSE_LEVEL = 1 # 0 = silencioso, 1 = progresso
SAVE_PLOTS_PER_RUN = True # Enable plots for individual runs

# --- Função de Aumentação de Dados ---

def _time_shift_augmentation(signal, max_shift_percent=0.2):
    """
    Aplica um "roll" (time shift) aleatório em um único sinal.
    O que sai de um lado, entra no outro.
    """
    signal = signal.numpy() # Converte de Tensor para Numpy para 'roll'
    length = signal.shape[0]
    
    max_shift = int(length * max_shift_percent)
    if max_shift < 1:
        return signal.astype(np.float32)
        
    shift = np.random.randint(-max_shift, max_shift)
    
    signal_aug = np.roll(signal, shift, axis=0)
    
    return signal_aug.astype(np.float32)

# --- Classes de Lógica Específicas do RHCB5 ---

def build_rhcb5_model(input_shape, num_classes):
    """
    Constrói a arquitetura do modelo RHCB5, agora com Camada de Atenção.
    """
    inputs = Input(shape=input_shape)
    
    # Bloco CNN (Extrator de Features)
    x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x) # 4096 -> 2048
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x) # 2048 -> 1024
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x) # 1024 -> 512
    x = Dropout(0.2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x) # 512 -> 256
    x = Dropout(0.2)(x)
    
    # Bloco Recorrente (Processador de Sequência)
    # BiLSTM agora retorna a sequência inteira para a camada de Atenção
    x_lstm = Bidirectional(LSTM(256, return_sequences=True, name="bilstm"))(x)
    
    # --- Bloco de Atenção (Interpretabilidade) ---
    # Implementação padrão de Atenção (Bahdanau-style) para classificação
    
    # 1. 'u' é a representação interna da LSTM
    # (batch, 256, 512) -> (batch, 256, 256)
    u = Dense(256, activation='tanh', name='attention_tanh')(x_lstm)
    
    # 2. 'scores' calcula a importância de cada timestep
    # (batch, 256, 256) -> (batch, 256, 1)
    scores = Dense(1, activation=None, name='attention_scores_raw')(u)
    
    # 3. 'attention_weights' normaliza os scores (0 a 1)
    # Esta é a saída que usaremos para XAI
    attention_weights = Softmax(axis=1, name='attention_weights')(scores) # (batch, 256, 1)    
    # 4. 'context_vector' pondera a saída da LSTM pelos pesos da atenção
    # (batch, 256, 512) * (batch, 256, 1) = (batch, 256, 512)
    context_vector = x_lstm * attention_weights
    
    # 5. Resumo do vetor de contexto
    # (batch, 256, 512) -> (batch, 512)
    context_vector_sum = Lambda(lambda x: tf.reduce_sum(x, axis=1), 
                                name='attention_context_sum')(context_vector)

    # Bloco de Classificação (MLP)
    x = Dense(256, activation='relu')(context_vector_sum)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax', name="main_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="RHCB5_Attention_Model")
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- Funções de Interpretabilidade (XAI) ---

def make_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Gera heatmap Grad-CAM para o modelo RHCB5.
    (Analisa as camadas Conv1D)
    """
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv1D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No Conv1D layer found in the model for Grad-CAM")
    
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)
    
    return heatmap.numpy()

def apply_gradcam_to_samples(model, X_data, y_data, class_names, plots_dir, save_plots, num_samples_per_class=3):
    """
    Aplica Grad-CAM a amostras de sinais de cada classe.
    """
    print("Gerando Grad-CAM heatmaps (Foco da CNN)...")
    
    fig, axes = plt.subplots(len(class_names), num_samples_per_class, 
                            figsize=(5*num_samples_per_class, 4*len(class_names)))
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(y_data == class_idx)[0]
        if len(class_indices) == 0:
            print(f"Aviso: Nenhuma amostra de teste para a classe {class_name} no XAI.")
            continue
        
        if len(class_indices) < num_samples_per_class:
            selected_indices = class_indices
        else:
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
        
        for sample_idx, data_idx in enumerate(selected_indices):
            signal = X_data[data_idx]
            signal_reshaped = signal.reshape(1, -1, 1)
            
            heatmap = make_gradcam_heatmap(model, signal_reshaped.astype(np.float32))
            
            ax = axes[class_idx, sample_idx]
            ax.plot(signal.flatten(), color='blue', alpha=0.7, label='Signal')
            
            # Upsample do heatmap (de 256 para 4096)
            upsample_factor = signal.shape[0] // heatmap.shape[0]
            heatmap_upsampled = np.repeat(heatmap, upsample_factor)
            # Garante o mesmo tamanho caso haja arredondamento
            if len(heatmap_upsampled) < signal.shape[0]:
                heatmap_upsampled = np.pad(heatmap_upsampled, (0, signal.shape[0] - len(heatmap_upsampled)), 'edge')

            signal_range = np.max(signal) - np.min(signal)
            heatmap_scaled = heatmap_upsampled * signal_range * 0.5
            
            # Corrigido para plotar heatmap no local correto
            ax.fill_between(range(len(heatmap_scaled)), 
                          np.min(signal) - heatmap_scaled, # Começa abaixo do sinal
                          np.min(signal), # Termina no min do sinal
                          color='red', alpha=0.3, label='Grad-CAM')
            
            ax.set_title(f'{class_name} - Sample {sample_idx+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Plotting._handle_plot(fig, "gradcam_heatmaps.png", plots_dir, save_plots, 
                         "Grad-CAM Heatmaps for RHCB5 (CNN Focus)")

def plot_attention_heatmap(attention_model, X_data, y_data, class_names, plots_dir, save_plots, upsample_factor, num_samples_per_class=3):
    """
    Extrai e plota os pesos da camada de Atenção para amostras de cada classe.
    """
    print("Gerando Attention heatmaps (Foco da LSTM/Atenção)...")
    
    # Pega pesos de atenção para TODOS os dados de teste de uma vez
    attention_weights = attention_model.predict(X_data) # Shape (n_samples, 256, 1)
    attention_weights = np.squeeze(attention_weights, axis=-1) # (n_samples, 256)

    fig, axes = plt.subplots(len(class_names), num_samples_per_class, 
                            figsize=(5*num_samples_per_class, 4*len(class_names)))
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(y_data == class_idx)[0]
        if len(class_indices) == 0:
            print(f"Aviso: Nenhuma amostra de teste para a classe {class_name} no XAI.")
            continue
            
        if len(class_indices) < num_samples_per_class:
            selected_indices = class_indices
        else:
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
        
        for sample_idx, data_idx in enumerate(selected_indices):
            signal = X_data[data_idx]
            heatmap = attention_weights[data_idx] # Pega o peso (256,)
            
            # Upsample do heatmap (de 256 para 4096)
            heatmap_upsampled = np.repeat(heatmap, upsample_factor)
            if len(heatmap_upsampled) < signal.shape[0]:
                heatmap_upsampled = np.pad(heatmap_upsampled, (0, signal.shape[0] - len(heatmap_upsampled)), 'edge')

            ax = axes[class_idx, sample_idx]
            ax.plot(signal.flatten(), color='blue', alpha=0.7, label='Signal')
            
            # Normaliza e escala o heatmap de atenção
            heatmap_norm = (heatmap_upsampled - np.min(heatmap_upsampled)) / (np.max(heatmap_upsampled) - np.min(heatmap_upsampled) + 1e-10)
            signal_range = np.max(signal) - np.min(signal)
            heatmap_scaled = heatmap_norm * signal_range * 0.5 
            
            ax.fill_between(range(len(heatmap_scaled)), 
                          np.min(signal) - heatmap_scaled,
                          np.min(signal),
                          color='green', alpha=0.3, label='Attention')
            
            ax.set_title(f'{class_name} - Sample {sample_idx+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Plotting._handle_plot(fig, "attention_heatmaps.png", plots_dir, save_plots, 
                         "Attention Weights Heatmaps for RHCB5 (LSTM/Attention Focus)")


def run_rhcb5_pipeline(run_id, base_results_dir, global_constants, random_seed_for_run, data_processed=None, raw_labels=None, run_xai=False):
    """
    Encapsula a execução completa do pipeline RHCB5 para uma única execução.
    """
    start_time_total = time.time()
    
    # 1. Configurar diretórios e seeds
    RUN_RESULTS_DIR = os.path.join(base_results_dir, f"run_{run_id:02d}_seed_{random_seed_for_run}")
    PLOTS_DIR = os.path.join(RUN_RESULTS_DIR, "plots")
    MODEL_SAVE_PATH = os.path.join(RUN_RESULTS_DIR, 'best_rhcb5_model.keras')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    np.random.seed(random_seed_for_run)
    tf.random.set_seed(random_seed_for_run)
    
    print(f"Iniciando RHCB5 Run {run_id} (Seed: {random_seed_for_run})")
    print(f"Resultados individuais em: {RUN_RESULTS_DIR}")

    # Configuração GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth habilitado para {len(gpus)} GPUs.")
        except RuntimeError as e:
            print(f"Erro ao habilitar memory growth: {e}")

    run_results = {
        "run_id": run_id,
        "seed": random_seed_for_run,
        "pipeline_name": "RHCB5"
    }

    try:
        # 2/3. Carregar e Pré-processar Dados (se não fornecidos)
        if data_processed is None or raw_labels is None:
            print("\n--- 1. Carregando Dados (Utils) ---")
            BASE_DATA_DDIR = global_constants.get("BASE_DATA_DIR", "../data")
            raw_data, raw_labels = DataHandler.load_bonn_data(BASE_DATA_DIR)
            
            print("\n--- 2. Pré-processando Dados (Utils) ---")
            data_processed = DataHandler.preprocess_eeg(
                raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
            )
            # Plots de pré-processamento (opcional, mantido)
            Plotting.plot_original_vs_filtered_signals(raw_data, data_processed, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                                                     title="Original vs Filtered Signals", filename="preprocessing_comparison.png")
        else:
            pass # Usar dados fornecidos
        
        # RHCB5 espera entrada (N, 4096, 1)
        X = np.expand_dims(data_processed, axis=-1).astype(np.float32)
        y = raw_labels.astype(np.int32)
        
        del data_processed # Libera memória
        gc.collect()

        # 4. Dividir Dados
        X_train, X_val, X_test, y_train, y_val, y_test = (
            DataHandler.split_data(
                X, y,
                test_size=TEST_SIZE, val_size=VAL_SIZE,
                random_state=random_seed_for_run,
            )
        )
        del X, y # Libera memória
        
        # --- 5. Preparar tf.data.Dataset com Aumentação ---
        print("\n--- 5. Preparando tf.data.Dataset com Aumentação ---")
        
        def _augment_fn(signal, label):
            # Função wrapper para time shifting
            # [length, 1]
            signal_aug = tf.py_function(
                _time_shift_augmentation, 
                [signal, AUGMENT_MAX_SHIFT_PERCENT], 
                tf.float32
            )
            signal_aug.set_shape([TARGET_INPUT_LENGTH, 1]) # Garante que o shape seja conhecido
            label.set_shape([])
            return signal_aug, label

        # Criar o dataset de treino
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=len(X_train))
        train_ds = train_ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Criar datasets de validação e teste (sem aumentação)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(BATCH_SIZE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(BATCH_SIZE)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Limpar memória (X_test/y_test mantidos para XAI)
        del X_train, y_train, X_val, y_val
        gc.collect()

        # 6. Construir e Treinar o Modelo
        print("\n--- 6. Construindo e Treinando o Modelo RHCB5 (com Atenção) ---")
        start_time_dnn_train = time.time()
        
        tf.keras.backend.clear_session()
        input_shape = (TARGET_INPUT_LENGTH, 1)
        model = build_rhcb5_model(input_shape, NUM_CLASSES)
        
        if VERBOSE_LEVEL > 0:
            model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE_RHCB5, restore_best_weights=True, verbose=VERBOSE_LEVEL),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=VERBOSE_LEVEL)
        ]
        
        history = model.fit(
            train_ds, # <-- Dataset de treino aumentado
            epochs=NUM_EPOCHS,
            validation_data=val_ds, # <-- Dataset de validação
            callbacks=callbacks,
            verbose=VERBOSE_LEVEL
        )
        
        history_data = history.history
        run_results["dnn_train_eval_time_sec"] = time.time() - start_time_dnn_train
        
        if SAVE_PLOTS_PER_RUN:
            Plotting.plot_dnn_training_history(
                history_data, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                title=f"Histórico de Treino RHCB5 (com Atenção) - Run {run_id}",
                filename="rhcb5_training_history.png",
            )
        
        # 7. Avaliação Final no Conjunto de Teste
        print("\n--- 7. Avaliação Final no Conjunto de Teste ---")
        # Carrega o melhor modelo salvo
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("Melhor modelo (best_rhcb5_model.keras) carregado para avaliação.")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o modelo salvo. Usando o modelo final em memória. Erro: {e}")

        test_loss, test_accuracy = model.evaluate(test_ds, verbose=VERBOSE_LEVEL) # <-- Dataset de teste
        print(f"Acurácia no teste (Run {run_id}): {test_accuracy:.4f}")
        
        y_pred_probs = model.predict(test_ds) # <-- Dataset de teste
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # y_test (numpy) foi mantido para esta métrica
        final_metrics = Metrics.calculate_all_metrics(y_test, y_pred, class_names=CLASS_NAMES)
        
        if final_metrics:
            run_results["final_metrics"] = final_metrics
            run_results["final_accuracy"] = final_metrics.get("accuracy", 0.0)
            run_results["num_selected_features"] = np.nan 

        # 8. Análise de Interpretabilidade (XAI)
        if run_xai:
            print("\n--- 8. Análise de Interpretabilidade (XAI) ---")
            
            # --- XAI 1: Grad-CAM (Foco da CNN) ---
            try:
                apply_gradcam_to_samples(
                    model, X_test, y_test, CLASS_NAMES, 
                    PLOTS_DIR, SAVE_PLOTS_PER_RUN
                )
                run_results["gradcam_completed"] = True
            except Exception as e:
                print(f"Erro na análise Grad-CAM: {e}")
                run_results["gradcam_error"] = str(e)
                
            # --- XAI 2: Attention Weights (Foco da LSTM/Atenção) ---
            print("Criando modelo XAI de Atenção...")
            try:
                attention_weights_output = model.get_layer('attention_weights').output
                attention_model = Model(inputs=model.input, outputs=attention_weights_output)
                
                # Fator de upsample = 4096 / 256 = 16
                upsample_factor = (TARGET_INPUT_LENGTH // attention_weights_output.shape[1])
                
                plot_attention_heatmap(
                    attention_model, X_test, y_test, CLASS_NAMES,
                    PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                    upsample_factor=upsample_factor
                )
                run_results["attention_xai_completed"] = True
            except Exception as e:
                print(f"Erro na análise de Atenção: {e}")
                run_results["attention_xai_error"] = str(e)
        else:
            print("\n--- 8. Análise de Interpretabilidade (XAI) - Pulada ---")

        del model, X_test, y_test, y_pred_probs, y_pred # Libera memória
        gc.collect()

    except Exception as e:
        print(f"ERRO na execução {run_id} (RHCB5): {e}")
        import traceback
        traceback.print_exc()
        run_results["error"] = str(e)

    # 9. Finalização
    total_execution_time = time.time() - start_time_total
    run_results["execution_time_sec"] = total_execution_time
    print(f"RHCB5 Run {run_id} concluída. Tempo total: {total_execution_time/60:.2f} minutos.")
    
    # Salva os resultados individuais
    results_file_path = os.path.join(RUN_RESULTS_DIR, "run_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(run_results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Erro ao salvar resultados individuais da Run {run_id}: {e}")

    return run_results

# Este arquivo não deve ser executado diretamente
if __name__ == "__main__":
    print("Este é um script de biblioteca de pipeline (RHCB5) e deve ser chamado por 'main.py'.")