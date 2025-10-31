# -*- coding: utf-8 -*-
"""
Pipeline RHCB5 (Baseado em rhcb5.py)

Este script foi refatorado para ser uma 'biblioteca' callable.
A lógica principal foi encapsulada na função run_rhcb5_pipeline.

Ele depende de 'pipeline_utils.py' para:
- DataHandler (load, preprocess, split)
- Metrics (calculate_all_metrics)
- Plotting (plot_dnn_training_history, etc.)
- Constantes Globais (FS, HIGHCUT_HZ, etc.)
"""

import os
import sys
import time
import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, 
    Bidirectional, LSTM, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc

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

# Nível de verbosidade
VERBOSE_LEVEL = 1 # 0 = silencioso, 1 = progresso
SAVE_PLOTS_PER_RUN = True # Enable plots for individual runs

# --- Classes de Lógica Específicas do RHCB5 ---

def build_rhcb5_model(input_shape, num_classes):
    """
    Constrói a arquitetura do modelo RHCB5.
    (Mantido de rhcb5.py)
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=False))(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs, name="RHCB5_Epilepsy_Model")
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- Função Principal do Pipeline ---

def run_rhcb5_pipeline(run_id, base_results_dir, global_constants, random_seed_for_run, data_processed=None, raw_labels=None):
    """
    Encapsula a execução completa do pipeline RHCB5 para uma única execução.
    
    Args:
        run_id (int): Identificador desta execução (ex: 1, 2, ..., 50).
        base_results_dir (str): Diretório base para salvar os resultados desta execução.
        global_constants (dict): Dicionário de constantes globais (ex: BASE_DATA_DIR).
        random_seed_for_run (int): A seed a ser usada para esta execução específica.
        data_processed (np.ndarray): Dados pré-processados (se fornecida, pula carregamento e pré-processamento).
        raw_labels (np.ndarray): Labels dos dados (se fornecida).

    Returns:
        dict: Um dicionário contendo as métricas finais e resultados desta execução.
    """
    start_time_total = time.time()
    
    # 1. Configurar diretórios e seeds para esta execução
    RUN_RESULTS_DIR = os.path.join(base_results_dir, f"run_{run_id:02d}_seed_{random_seed_for_run}")
    PLOTS_DIR = os.path.join(RUN_RESULTS_DIR, "plots")
    MODEL_SAVE_PATH = os.path.join(RUN_RESULTS_DIR, 'best_rhcb5_model.h5')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Define as seeds para esta execução
    np.random.seed(random_seed_for_run)
    tf.random.set_seed(random_seed_for_run)
    
    print(f"Iniciando RHCB5 Run {run_id} (Seed: {random_seed_for_run})")
    print(f"Resultados individuais em: {RUN_RESULTS_DIR}")

    # Configure GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs disponíveis: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth habilitado para GPUs.")
        except RuntimeError as e:
            print(f"Erro ao habilitar memory growth: {e}")
    else:
        print("Nenhuma GPU encontrada. TensorFlow usará CPU.")

    # Dicionário para armazenar todos os resultados desta execução
    run_results = {
        "run_id": run_id,
        "seed": random_seed_for_run,
        "pipeline_name": "RHCB5"
    }

    try:
        if data_processed is None or raw_labels is None:
            # Fallback: carregar e processar dados
            print("\n--- 1. Carregando Dados (Utils) ---")
            BASE_DATA_DIR = global_constants["BASE_DATA_DIR"]
            raw_data, raw_labels = DataHandler.load_bonn_data(BASE_DATA_DIR)
            
            print("\n--- 2. Pré-processando Dados (Utils) ---")
            data_processed = DataHandler.preprocess_eeg(
                raw_data, fs=FS, highcut_hz=HIGHCUT_HZ, order=FILTER_ORDER
            )
            
            # Generate preprocessing plots
            Plotting.plot_original_vs_filtered_signals(raw_data, data_processed, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                                                     title="Original vs Filtered Signals", filename="preprocessing_comparison.png")
            Plotting.plot_power_spectral_density(raw_data, FS, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                                               title="Power Spectral Density - Raw Signals", filename="psd_raw.png")
            Plotting.plot_power_spectral_density(data_processed, FS, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                                               title="Power Spectral Density - Filtered Signals", filename="psd_filtered.png")
        else:
            # Usar dados pré-processados
            pass
        
        # RHCB5 espera entrada (N, 4096, 1)
        X = np.expand_dims(data_processed, axis=-1).astype(np.float32)
        y = raw_labels.astype(np.int32)
        
        del data_processed # Libera memória
        gc.collect()

        # 4. Dividir Dados (usando DataHandler do pipeline_utils)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            DataHandler.split_data(
                X, y,
                test_size=TEST_SIZE, val_size=VAL_SIZE,
                random_state=random_seed_for_run, # Usa a seed da execução
            )
        )
        del X, y # Libera memória
        gc.collect()

        # 5. Construir e Treinar o Modelo
        print("\n--- 4. Construindo e Treinando o Modelo RHCB5 ---")
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
            X_train, y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=VERBOSE_LEVEL
        )
        
        history_data = history.history
        run_results["dnn_train_eval_time_sec"] = time.time() - start_time_dnn_train
        
        if SAVE_PLOTS_PER_RUN:
            Plotting.plot_dnn_training_history(
                history_data, PLOTS_DIR, SAVE_PLOTS_PER_RUN,
                title=f"Histórico de Treino RHCB5 - Run {run_id}",
                filename="rhcb5_training_history.png",
            )
        
        del X_train, y_train, X_val, y_val # Libera memória
        gc.collect()

        # 6. Avaliação Final
        print("\n--- 5. Avaliação Final no Conjunto de Teste ---")
        # Carrega o melhor modelo salvo
        try:
            model.load_weights(MODEL_SAVE_PATH)
            print("Melhor modelo (best_rhcb5_model.h5) carregado para avaliação.")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o modelo salvo. Usando o modelo final em memória. Erro: {e}")

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=VERBOSE_LEVEL)
        print(f"Acurácia no teste (Run {run_id}): {test_accuracy:.4f}")
        
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        final_metrics = Metrics.calculate_all_metrics(y_test, y_pred, class_names=CLASS_NAMES)
        
        if final_metrics:
            run_results["final_metrics"] = final_metrics
            run_results["final_accuracy"] = final_metrics.get("accuracy", 0.0)
            # RHCB5 não seleciona features
            run_results["num_selected_features"] = np.nan 
        
        del model, X_test, y_test, y_pred_probs, y_pred # Libera memória
        gc.collect()

    except Exception as e:
        print(f"ERRO na execução {run_id} (RHCB5): {e}")
        import traceback
        traceback.print_exc()
        run_results["error"] = str(e)

    # 7. Finalização
    total_execution_time = time.time() - start_time_total
    run_results["execution_time_sec"] = total_execution_time
    print(f"RHCB5 Run {run_id} concluída. Tempo total: {total_execution_time/60:.2f} minutos.")
    
    # Salva os resultados individuais desta execução
    results_file_path = os.path.join(RUN_RESULTS_DIR, "run_results.json")
    try:
        with open(results_file_path, "w") as f:
            json.dump(run_results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Erro ao salvar resultados individuais da Run {run_id}: {e}")

    return run_results

# Este arquivo não deve ser executado diretamente, mas sim importado por main.py
if __name__ == "__main__":
    print("Este é um script de biblioteca de pipeline (RHCB5) e deve ser chamado por 'main.py'.")
