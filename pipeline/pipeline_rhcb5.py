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
import json
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, 
    Bidirectional, LSTM, Dense
)
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

# --- Funções de Interpretabilidade (XAI) ---

def make_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for RHCB5 model.
    
    Args:
        model: Trained Keras model
        img_array: Input signal array (shape: (1, seq_length, 1))
        pred_index: Index of the class to explain (if None, uses predicted class)
    
    Returns:
        heatmap: Grad-CAM heatmap
    """
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv1D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No Conv1D layer found in the model for Grad-CAM")
    
    # Create a model that maps the input to the activations of the last conv layer
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    # Weight the convolutional outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)
    
    return heatmap.numpy()

def apply_gradcam_to_samples(model, X_data, y_data, class_names, plots_dir, save_plots, num_samples_per_class=3):
    """
    Apply Grad-CAM to sample signals from each class and generate visualizations.
    
    Args:
        model: Trained Keras model
        X_data: Input data (shape: (n_samples, seq_length, 1))
        y_data: True labels
        class_names: List of class names
        plots_dir: Directory to save plots
        save_plots: Whether to save plots
        num_samples_per_class: Number of samples to analyze per class
    """
    print("Generating Grad-CAM heatmaps...")
    
    fig, axes = plt.subplots(len(class_names), num_samples_per_class, 
                            figsize=(5*num_samples_per_class, 4*len(class_names)))
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)
    
    samples_analyzed = 0
    
    for class_idx, class_name in enumerate(class_names):
        # Get samples from this class
        class_indices = np.where(y_data == class_idx)[0]
        if len(class_indices) < num_samples_per_class:
            selected_indices = class_indices
        else:
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
        
        for sample_idx, data_idx in enumerate(selected_indices):
            # Get the signal
            signal = X_data[data_idx]
            signal_reshaped = signal.reshape(1, -1, 1)
            
            # Generate heatmap
            heatmap = make_gradcam_heatmap(model, signal_reshaped.astype(np.float32))
            
            # Plot
            ax = axes[class_idx, sample_idx]
            
            # Plot original signal
            ax.plot(signal.flatten(), color='blue', alpha=0.7, label='Signal')
            
            # Overlay heatmap (scaled to signal range)
            signal_range = np.max(signal) - np.min(signal)
            heatmap_scaled = heatmap * signal_range * 0.5  # Scale for visibility
            ax.fill_between(range(len(heatmap)), 
                          np.min(signal) - heatmap_scaled,
                          np.max(signal) + heatmap_scaled,
                          color='red', alpha=0.3, label='Grad-CAM')
            
            ax.set_title(f'{class_name} - Sample {sample_idx+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            samples_analyzed += 1
    
    plt.tight_layout()
    Plotting._handle_plot(fig, "gradcam_heatmaps.png", plots_dir, save_plots, 
                         "Grad-CAM Heatmaps for RHCB5")

def perform_shap_analysis(model, X_background, X_test, y_test, class_names, plots_dir, save_plots, nsamples=500):
    """
    Perform SHAP analysis on the RHCB5 model.
    
    Args:
        model: Trained Keras model
        X_background: Background dataset for SHAP (smaller subset)
        X_test: Test dataset
        y_test: Test labels
        class_names: List of class names
        plots_dir: Directory to save plots
        save_plots: Whether to save plots
        nsamples: Number of samples for SHAP approximation
    """
    try:
        import shap
        print("Performing SHAP analysis...")
        
        # Create SHAP explainer
        # For sequential models, we need to flatten the input for KernelExplainer
        def model_wrapper(x):
            # x is flattened (batch, 4096), reshape to (batch, 4096, 1)
            x_reshaped = x.reshape(x.shape[0], -1, 1)
            return model.predict(x_reshaped)
        
        background_size = min(20, len(X_background))  # Smaller background for KernelExplainer speed
        background_flat = X_background[:background_size].reshape(background_size, -1)  # Flatten to (20, 4096)
        
        # Use KernelExplainer with flattened input
        explainer = shap.KernelExplainer(model_wrapper, background_flat)
        
        # Select fewer test samples for analysis (KernelExplainer is slower)
        test_size = min(5, len(X_test))
        test_indices = np.random.choice(len(X_test), test_size, replace=False)
        X_test_sample = X_test[test_indices]
        X_test_flat = X_test_sample.reshape(test_size, -1)  # Flatten to (5, 4096)
        y_test_sample = y_test[test_indices]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_flat)
        
        # Get predictions for the test sample
        predictions = model.predict(X_test_sample)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Handle SHAP values shape
        if isinstance(shap_values, list):
            # Multi-class case: list of 2D arrays (n_samples, n_features)
            shap_for_pred = np.array([shap_values[pred_class][i] for i, pred_class in enumerate(pred_classes)])
        else:
            # Single output case
            shap_for_pred = shap_values
        
        # shap_for_pred should be 2D (n_samples, n_features)
        
        # Save SHAP values BEFORE attempting to plot (so we don't lose data if plotting fails)
        shap_results = {
            "shap_values_sample": shap_for_pred.tolist(),
            "test_predictions": predictions.tolist(),
            "test_true_labels": y_test_sample.tolist(),
            "pred_classes": pred_classes.tolist(),
            "background_size": background_size,
            "test_sample_size": test_size
        }
        
        # Save to JSON file
        shap_json_path = os.path.join(plots_dir, "shap_values.json")
        try:
            with open(shap_json_path, 'w') as f:
                json.dump(shap_results, f, indent=4)
            print(f"SHAP values saved to: {shap_json_path}")
        except Exception as e:
            print(f"Warning: Could not save SHAP values to JSON: {e}")
        
        # Now attempt to create plots
        try:
            # Plot summary plot
            fig_summary = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_for_pred, 
                             feature_names=[f't_{i}' for i in range(shap_for_pred.shape[1])],
                             show=False)
            Plotting._handle_plot(fig_summary, "shap_summary_plot.png", plots_dir, save_plots,
                                 "SHAP Summary Plot for RHCB5")
        except Exception as e:
            print(f"Warning: Could not create SHAP summary plot: {e}")
        
        try:
            # Plot waterfall plot for first sample
            fig_waterfall = plt.figure(figsize=(12, 6))
            # Get expected value for the predicted class of the first sample
            if isinstance(explainer.expected_value, list):
                expected_val = explainer.expected_value[pred_classes[0]]
            else:
                expected_val = explainer.expected_value
            
            # Create an Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_for_pred[0],
                base_values=expected_val,
                data=X_test_flat[0],
                feature_names=[f't_{i}' for i in range(shap_for_pred.shape[1])]
            )
            
            shap.plots.waterfall(explanation, show=False)
            Plotting._handle_plot(fig_waterfall, "shap_waterfall_sample.png", plots_dir, save_plots,
                                 "SHAP Waterfall Plot - Sample 1")
        except Exception as e:
            print(f"Warning: Could not create SHAP waterfall plot: {e}")
        
        return shap_results
        
    except ImportError:
        print("SHAP library not available. Skipping SHAP analysis.")
        return None
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None

def run_rhcb5_pipeline(run_id, base_results_dir, global_constants, random_seed_for_run, data_processed=None, raw_labels=None, run_xai=False):
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
    MODEL_SAVE_PATH = os.path.join(RUN_RESULTS_DIR, 'best_rhcb5_model.keras')
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
        
        # 6. Análise de Interpretabilidade (XAI)
        if run_xai:
            print("\n--- 6. Análise de Interpretabilidade (XAI) ---")
            
            # Grad-CAM analysis
            try:
                apply_gradcam_to_samples(
                    model, X_test, y_test, CLASS_NAMES, 
                    PLOTS_DIR, SAVE_PLOTS_PER_RUN
                )
                run_results["gradcam_completed"] = True
            except Exception as e:
                print(f"Erro na análise Grad-CAM: {e}")
                run_results["gradcam_error"] = str(e)
            
            # SHAP analysis
            try:
                # Use a subset of training data as background
                background_size = min(100, len(X_train))
                background_indices = np.random.choice(len(X_train), background_size, replace=False)
                X_background = X_train[background_indices]
                
                shap_results = perform_shap_analysis(
                    model, X_background, X_test[:5], y_test[:5],  # Limit test size for SHAP
                    CLASS_NAMES, PLOTS_DIR, SAVE_PLOTS_PER_RUN
                )
                if shap_results:
                    run_results["shap_results"] = shap_results
                    run_results["shap_completed"] = True
                else:
                    run_results["shap_completed"] = False
            except Exception as e:
                print(f"Erro na análise SHAP: {e}")
                run_results["shap_error"] = str(e)
        else:
            print("\n--- 6. Análise de Interpretabilidade (XAI) - Pulada ---")

        del X_train, y_train, X_val, y_val # Libera memória
        gc.collect()

        # 7. Avaliação Final no Conjunto de Teste
        print("\n--- 7. Avaliação Final no Conjunto de Teste ---")
        # Carrega o melhor modelo salvo
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("Melhor modelo (best_rhcb5_model.keras) carregado para avaliação.")
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

    # 8. Finalização
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
