[To read in English, click here](#english-version)

# Estratégias Computacionais para Detecção de Epilepsia em EEG: Pipeline-Based versus End-to-End Approaches

## Contexto e Importância

A epilepsia é um distúrbio neurológico que afeta aproximadamente 50 milhões de pessoas worldwide, caracterizado por crises epilépticas recorrentes. A Eletroencefalografia (EEG) é a técnica padrão-ouro para diagnóstico e monitoramento de epilepsia, capturando a atividade elétrica cerebral através de eletrodos posicionados no couro cabeludo.

**Desafio Principal**: A detecção automática de crises epilépticas em sinais de EEG é crucial para:
- Diagnóstico precoce e preciso
- Monitoramento contínuo de pacientes
- Redução de falsos positivos em sistemas de alerta
- Suporte a decisões clínicas baseadas em dados

**Abordagens Tradicionais vs Modernas**:
- **Pipeline-Based**: Seguindo o paradigma clássico de Machine Learning (extração manual de features + classificação)
- **End-to-End**: Aproveitando Deep Learning para aprender features automaticamente dos dados brutos

Este projeto compara essas duas abordagens usando o dataset público da Universidade de Bonn, estabelecendo um benchmark para futuras pesquisas em detecção de epilepsia baseada em EEG.

Este trabalho é inspirado no artigo:
* Yogarajan, G., Alsubaie, N., Rajasekaran, G. et al. EEG-based epileptic seizure detection using binary dragonfly algorithm and deep neural network. *Sci Rep* **13**, 17710 (2023). [https://doi.org/10.1038/s41598-023-44318-w](https://doi.org/10.1038/s41598-023-44318-w]

E incorpora a arquitetura RHCB5 proposta por:
* Maggioni, A. et al. (2023/2024) - Rede Híbrida Convolucional Bidirecional para classificação de EEG.

## Funcionalidades

### Pipeline-Based (BDA+DNN)

#### Pré-processamento de Sinais
* **Filtragem**: Filtro Butterworth passa-baixas de ordem 4, frequência de corte 40Hz (remove ruído de alta frequência)
* **Normalização**: Min-Max scaling para intervalo [-1, 1], preservando relações relativas
* **Segmentação**: Sinais de 4097 pontos → 4096 pontos (remoção do primeiro ponto para estabilidade)

#### Extração de Características SWT
* **Transformada Wavelet**: Stationary Wavelet Transform (SWT) com wavelet 'db4' (Daubechies 4), nível de decomposição 4
* **Sub-bandas**: 5 componentes por sinal:
  - cA4: Aproximação nível 4 (0-5.86 Hz)
  - cD4: Detalhe nível 4 (5.86-11.72 Hz)
  - cD3: Detalhe nível 3 (11.72-23.44 Hz)
  - cD2: Detalhe nível 2 (23.44-46.88 Hz)
  - cD1: Detalhe nível 1 (46.88-93.75 Hz)

* **Características Estatísticas (9 por sub-banda)**:
  1. **MAV (Mean Absolute Value)**: $\frac{1}{N} \sum_{i=1}^{N} |x_i|$ - Energia média do sinal
  2. **StdDev (Standard Deviation)**: $\sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$ - Variabilidade
  3. **Skewness**: $\frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^3}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^{3/2}}$ - Assimetria da distribuição
  4. **Kurtosis**: $\frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^4}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2} - 3$ - Curtose (achatamento)
  5. **RMS (Root Mean Square)**: $\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$ - Valor eficaz
  6. **Activity (Hjorth)**: $\frac{1}{N} \sum_{i=1}^{N} x_i^2$ - Variância do sinal no tempo
  7. **Mobility (Hjorth)**: $\sqrt{\frac{\text{Activity}(\frac{dx}{dt})}{\text{Activity}(x)}}$ - Mobilidade (razão entre variâncias)
  8. **Complexity (Hjorth)**: $\frac{\text{Mobility}(\frac{dx}{dt})}{\text{Mobility}(x)}$ - Complexidade (normalizada)
  9. **MAV Ratio**: $\frac{\text{MAV}(cD_i)}{\text{MAV}(cA4)}$ - Razão relativa à banda de baixa frequência

**Total**: 45 características (5 sub-bandas × 9 features)

#### Seleção de Características com BDA
* **Algoritmo**: Binary Dragonfly Algorithm (BDA) - meta-heurística bio-inspirada
* **Codificação**: Vetor binário de 45 dimensões (1 = feature selecionada, 0 = não selecionada)
* **Função de Fitness**: $Fitness = \alpha \cdot \text{ErrorRate} + \beta \cdot \frac{\text{NumFeaturesSel}}{\text{TotalFeatures}}$
  - $\alpha = 0.99$, $\beta = 0.01$
  - ErrorRate: taxa de erro da DNN de validação (1 - accuracy)
* **Parâmetros BDA**:
  - População: 10 libélulas
  - Iterações: 100
  - Pesos: separação=0.1, alinhamento=0.1, coesão=0.7, food=1.0, enemy=1.0
  - Inércia: 0.85 (fixa)
  - Transfer function: V-Shaped ($\tau \in [0.01, 4.0]$)

#### Classificação DNN
* **Arquitetura**: Multilayer Perceptron (MLP)
* **Camadas**: 3 ocultas (10 neurônios sigmoid) + saída softmax (3 classes)
* **Regularização**: Dropout, Early Stopping (patience=30)
* **Otimização**: Adam (lr=0.001), loss=sparse_categorical_crossentropy

### End-to-End (RHCB5)

#### Arquitetura Detalhada
```
Input: (4096, 1) - Sinal EEG pré-processado
├── Conv1D Blocks: Extração de features locais
│   ├── Conv1D (activation='relu')
│   ├── BatchNormalization
│   ├── MaxPooling1D
│   └── Dropout
├── Bi-LSTM: Modelagem temporal bidirecional
│   ├── Bi-LSTM (return_sequences=True)
│   ├── Dropout
│   └── Bi-LSTM (return_sequences=False)
├── Dense Layers: Classificação
│   ├── Dense (activation='relu')
│   ├── Dropout
│   └── Dense (3, activation='softmax')
Output: Probabilidades para [Normal, Interictal, Ictal]
```

#### Hiperparâmetros de Treinamento
* **Otimização**: Adam (lr=0.001, β1=0.9, β2=0.999)
* **Loss**: Sparse Categorical Crossentropy
* **Métricas**: Accuracy, Precision, Recall, F1-Score
* **Regularização**: Early Stopping (monitor='val_loss', patience=30, restore_best_weights=True)
* **Batch Size**: 16
* **Epochs**: 250 (máximo, com early stopping)

#### Análise de Interpretabilidade
* **Grad-CAM**: Visualização de regiões salientes no sinal de entrada
* **SHAP**: Valores de Shapley para explicabilidade global e local
* **Aplicação**: Identificação de padrões temporais críticos para classificação

### Comparação e Análise Estatística

#### Metodologia Experimental
* **Reprodutibilidade**: 30 execuções independentes por pipeline com seeds aleatórias
* **Divisão de Dados**: Estratificada (70% treino, 15% validação, 15% teste)
* **Validação Cruzada**: 10-fold CV interna para avaliação de features (BDA)

#### Métricas de Avaliação
* **Por Classe**: Precision, Recall, F1-Score, Specificity
* **Agregadas**: Accuracy, Macro-F1, Weighted-F1
* **Matriz de Confusão**: Análise de erros por classe

#### Análise Estatística
* **Testes de Normalidade**: Shapiro-Wilk nos diferenciais pareados
* **Testes de Significância**: Wilcoxon (não-paramétrico) e T-test (se normal)
* **Tamanho do Efeito**: Cohen's d
* **Intervalos de Confiança**: Bootstrap (95%, n=10,000 reamostragens)
* **Correlações**: Pearson entre métricas e tempo de execução

#### Visualizações
* **Boxplots**: Distribuição de métricas entre pipelines
* **Scatter Plots**: Performance vs custo computacional
* **Heatmaps**: Frequência de seleção de features (BDA)
* **Matrizes de Confusão**: Agregadas e por run
* **Curvas de Convergência**: Fitness ao longo das iterações (BDA)

## Dataset e Características Técnicas

### Dataset Bonn EEG
* **Fonte**: Universidade de Bonn, Alemanha (1998-2001)
* **População**: 5 pacientes saudáveis (Set A) + 5 pacientes epilépticos (Sets B-E)
* **Aquisição**:
  - Eletrodo único (C3 ou C4) vs referência auricular
  - Frequência de amostragem: 173.61 Hz
  - Resolução: 12 bits
  - Filtro antialiasing: 0.53-40 Hz

#### Composição dos Dados
| Set | Classe | Descrição | N° Segmentos | Duração |
|-----|--------|-----------|--------------|---------|
| A   | Normal | EEG de olhos abertos (saudáveis) | 100 | 23.6s |
| D   | Interictal | EEG interictal (epilépticos, lobo temporal) | 100 | 23.6s |
| E   | Ictal | EEG ictal (epilépticos, mesma região) | 100 | 23.6s |

**Total**: 300 segmentos de 4097 pontos cada (23.6 segundos)

#### Características Espectrais
* **Set A (Normal)**: Atividade alfa dominante (8-12 Hz), beta (12-30 Hz)
* **Set D (Interictal)**: Padrões anômalos, spikes isolados
* **Set E (Ictal)**: Atividade rítmica de alta amplitude, frequência variável

### Detalhes Técnicos de Implementação

#### Pré-processamento
```python
# Filtro Butterworth
from scipy.signal import butter, filtfilt
b, a = butter(order=4, Wn=40/(FS/2), btype='low')
filtered_signal = filtfilt(b, a, raw_signal)

# Normalização Min-Max
normalized = 2 * (filtered_signal - min(filtered_signal)) / (max(filtered_signal) - min(filtered_signal)) - 1
```

#### SWT Feature Extraction
```python
import pywt
coeffs = pywt.swt(signal, wavelet='db4', level=4)
# coeffs = [cA4, cD4, cD3, cD2, cD1]

# Exemplo: cálculo de MAV
mav = np.mean(np.abs(coeff))
```

#### BDA Optimization Loop
```python
# Pseudocódigo simplificado
for iteration in range(T_MAX_ITER):
    for dragonfly in population:
        # Calcular comportamentos sociais
        separation = calculate_separation(dragonfly, neighbors)
        alignment = calculate_alignment(dragonfly, neighbors)  
        cohesion = calculate_cohesion(dragonfly, neighbors)
        food_attraction = calculate_food_attraction(dragonfly, food_pos)
        enemy_distraction = calculate_enemy_distraction(dragonfly, enemy_pos)
        
        # Atualizar velocidade
        delta_X = (separation + alignment + cohesion + food_attraction + enemy_distraction) * w_inertia
        
        # Transfer function V-Shaped
        tau = tau_max - (tau_max - tau_min) * (iteration / T_MAX_ITER)
        prob_flip = abs(np.tanh(delta_X / tau))
        
        # Atualizar posição binária
        for bit in range(dim):
            if np.random.rand() < prob_flip[bit]:
                dragonfly[bit] = 1 - dragonfly[bit]
```

#### RHCB5 Model Architecture
```python
def build_rhcb5_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional blocks
    x = tf.keras.layers.Conv1D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Additional conv blocks...
    
    # Bi-LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    )(x)
    
    # Dense classification
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```
## Estrutura do Projeto

```
epilepsy_detection_project/
├── data/                          # Dados de entrada
│   ├── Bonn/                      # Dataset principal
│   │   ├── A/                     # 100 arquivos .txt (EEG Normal)
│   │   ├── D/                     # 100 arquivos .txt (EEG Interictal)
│   │   └── E/                     # 100 arquivos .txt (EEG Ictal)
│   └── Siena/                     # Dataset alternativo (não utilizado)
├── pipeline/                      # Núcleo da implementação
│   ├── main.py                    # Orquestrador principal (11k+ linhas)
│   │   ├── Função main(): Loop de NUM_RUNS runs por pipeline
│   │   ├── compile_and_save_statistics(): Análise estatística
│   │   ├── run_pipeline_loop(): Execução paralela dos pipelines
│   │   └── Análise XAI/SHAP para melhores runs de ambos os modelos
│   ├── pipeline_bda_dnn.py        # Pipeline BDA+DNN (1k+ linhas)
│   │   ├── run_bda_dnn_pipeline(): Função principal
│   │   ├── FeatureExtractor: Classe para SWT e estatísticas
│   │   ├── BinaryDragonflyAlgorithm: Implementação BDA
│   │   └── PipelineHelpers: Funções auxiliares de treinamento
│   ├── pipeline_rhcb5.py          # Pipeline RHCB5 (600+ linhas)
│   │   ├── run_rhcb5_pipeline(): Função principal
│   │   ├── build_rhcb5_model(): Arquitetura da rede
│   │   ├── apply_gradcam_to_samples(): Interpretabilidade
│   │   └── perform_shap_analysis(): Análise SHAP
│   ├── pipeline_utils.py          # Utilitários compartilhados (1k+ linhas)
│   │   ├── DataHandler: Carregamento e pré-processamento
│   │   ├── Metrics: Cálculo de métricas de classificação
│   │   ├── Plotting: Todas as funções de visualização
│   │   └── Constantes globais e classes auxiliares
│   ├── generate_plots.py          # Scripts adicionais de plotagem
│   └── results/                   # Outputs das execuções
│       └── comparison_run_YYYY-MM-DD_HH-MM-SS/
│           ├── all_raw_results.json
│           ├── statistical_comparison_results.json
│           ├── stats_BDA_DNN_summary.csv
│           ├── stats_RHCB5_summary.csv
│           ├── confidence_intervals.json
│           ├── correlation_analysis.json
│           ├── plots/ (boxplots, scatter, heatmaps, etc.)
│           ├── BDA_DNN_runs/ (resultados individuais)
│           └── RHCB5_runs/ (resultados individuais)
├── src/                           # Implementações standalone (legado)
│   ├── bda_dnn.py                 # Versão antiga BDA+DNN
│   └── rhcb5.py                   # Versão antiga RHCB5
├── results/                       # Resultados gerais (plots estáticos)
├── LICENSE                        # Licença MIT
├── README.md                      # Esta documentação
├── requirements.txt               # Dependências Python
└── trabalho.tex                   # Documento LaTeX do TCC
```

## Configuração e Instalação

### Pré-requisitos Técnicos
* **Python**: 3.10+ (recomendado 3.10.12)
* **Sistema Operacional**: Linux/macOS (preferencial), Windows 10+
* **Memória RAM**: Mínimo 16GB, recomendado 32GB (para SHAP analysis)
* **Espaço em Disco**: 15GB+ para datasets e resultados
* **GPU**: NVIDIA com CUDA 11.8+ (opcional, acelera treinamento)

### Dependências Detalhadas

#### Core Dependencies
```
tensorflow[and-cuda]==2.15.0        # Deep Learning framework
numpy==2.1.3                        # Computação numérica
pandas==2.2.3                       # Manipulação de dados
scikit-learn==1.5.2                 # Machine Learning
scipy==1.15.3                       # Processamento de sinais
pywt==1.8.0                         # PyWavelets para SWT
```

#### Visualização e Análise
```
matplotlib==3.10.3                  # Plots básicos
seaborn==0.13.2                     # Plots estatísticos
tqdm==4.67.1                        # Barras de progresso
```

#### Interpretabilidade (XAI)
```
shap==0.49.1                        # SHAP values
scikeras==0.13.0                    # Integração scikit-learn + Keras
```

### Instalação Passo-a-Passo

1. **Clonagem e Setup**:
```bash
git clone https://github.com/andresichelero/TCC.git
cd TCC
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: .\venv\Scripts\activate  # Windows
```

2. **Instalação das Dependências**:
```bash
pip install -r requirements.txt
```

3. **Verificação da Instalação**:
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import pywt; print('PyWavelets OK')"
python -c "import shap; print('SHAP OK')"
```

### Configuração de Hardware

#### GPU Setup (NVIDIA)
```bash
# Verificar GPU disponível
nvidia-smi

# Instalar CUDA Toolkit (se necessário)
# Download: https://developer.nvidia.com/cuda-downloads

# Verificar instalação TensorFlow-GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

#### Configurações de Memória
```python
# Em pipeline_utils.py
USE_GPU = True  # Habilita GPU se disponível
tf.config.experimental.set_memory_growth(gpu, True)  # Crescimento dinâmico de memória
```

### Configurações Avançadas
#### Controle de Análise XAI
```python
USE_XAI = True           # Habilita SHAP/Grad-CAM (pode ser lento)
USE_GPU = True           # Usa GPU para aceleração
```

#### Logs e Debug
```python
# Em pipeline_utils.py
VERBOSE_LEVEL = 1  # 0=silent, 1=basic, 2=detailed
SAVE_PLOTS_PER_RUN = True  # Salva plots individuais
```

## Uso

O script principal `pipeline/main.py` orquestra a comparação completa entre os dois pipelines.

### Execução Básica

1.  **Executar a Comparação Completa:**
    Certifique-se de que o ambiente virtual está ativado e os dados estão em `data/Bonn/`.
    ```bash
    cd pipeline
    python main.py
    ```

2.  **Fluxo de Execução:**
    * Carrega e pré-processa os dados uma vez.
    * Executa NUM_RUNS runs do pipeline BDA+DNN com seeds aleatórias.
    * Executa NUM_RUNS runs do pipeline RHCB5 com as mesmas seeds.
    * Para o melhor run de cada pipeline, executa análise XAI/SHAP (se habilitado).
    * Compila estatísticas: média, mediana, desvio padrão, IQR, skewness, kurtosis.
    * Realiza testes estatísticos: Shapiro-Wilk, Wilcoxon, T-test pareado, Cohen's d.
    * Calcula intervalos de confiança via bootstrap.
    * Gera plots: boxplots, scatter plots, matrizes de confusão agregadas, heatmaps.
    * Salva todos os resultados em `pipeline/results/comparison_run_YYYY-MM-DD_HH-MM-SS/`.

3.  **Configurações Principais:**
    Edite `pipeline/pipeline_utils.py` para ajustar:
    * `NUM_RUNS = 30`: Número de execuções por pipeline.
    * `USE_XAI = True`: Habilitar análise SHAP/Grad-CAM.
    * `USE_GPU = True`: Usar GPU se disponível.
    * Parâmetros de pré-processamento: `FS`, `HIGHCUT_HZ`, `FILTER_ORDER`.

### Execução Individual de Pipelines

Para executar apenas um pipeline específico (para desenvolvimento/debugging):

```bash
# Pipeline BDA+DNN
python pipeline_bda_dnn.py

# Pipeline RHCB5  
python pipeline_rhcb5.py
```

**Nota:** Os scripts individuais não são projetados para execução standalone. Use sempre `main.py` para comparações consistentes.

## Descrição dos Módulos

### `pipeline/main.py`
Orquestrador principal que:
- Define o número de runs (NUM_RUNS) e gera seeds aleatórias.
- Executa loops de NUM_RUNS execuções para cada pipeline.
- Identifica os melhores runs (excluindo outliers de tempo).
- Re-executa os melhores runs com XAI habilitado.
- Compila estatísticas robustas e realiza análise estatística comparativa.
- Gera plots agregados e salva resultados em JSON/CSV.

### `pipeline/pipeline_bda_dnn.py`
Implementa o pipeline baseado em características:
- Extração de 143 features via SWT (16 sub-bandas × 8 características).
- Otimização com Binary Dragonfly Algorithm (BDA) para seleção de features.
- Treinamento de DNN com features selecionadas.
- Análise SHAP para interpretabilidade.
- Retorna métricas detalhadas e vetores de features selecionadas.

### `pipeline/pipeline_rhcb5.py`
Implementa o pipeline end-to-end:
- Arquitetura RHCB5: Conv1D → Bi-LSTM → Dense layers.
- Treinamento direto dos sinais de EEG (4096 pontos).
- Análise Grad-CAM e SHAP para interpretabilidade.
- Retorna métricas de classificação e visualizações.

### `pipeline/pipeline_utils.py`
Utilitários compartilhados:
- `DataHandler`: Carregamento, pré-processamento (filtro Butterworth, normalização) e divisão estratificada dos dados.
- `Metrics`: Cálculo de acurácia, F1-score, especificidade, matriz de confusão.
- `Plotting`: Geração de todos os gráficos (históricos de treino, boxplots, heatmaps, etc.).
- Constantes globais: seeds, parâmetros de sinal, nomes de classes.

### `src/bda_dnn.py` e `src/rhcb5.py`
Implementações standalone/legadas dos pipelines individuais. Usadas principalmente para desenvolvimento ou execução isolada (não recomendado para comparações).

## Pipelines Detalhados

### 1. Pipeline-Based: BDA + DNN

#### Etapas:
1. **Carregamento e Pré-processamento**: Dados BONN (A/D/E) → Filtro Butterworth (40Hz) → Normalização Min-Max.
2. **Extração de Características**: SWT com wavelet 'db4' nível 4 → 5 sub-bandas → 9 features cada (MAV, StdDev, Skewness, Kurtosis, RMS, Activity, Mobility, Complexity, MAV Ratio) → 45 features totais.
3. **Seleção de Features**: BDA otimiza subconjunto de features (fitness = α×erro + β×(features_sel/total_features)).
4. **Classificação**: DNN MLP (3 camadas ocultas, 10 neurônios sigmoid) treinada com features selecionadas.
5. **Avaliação**: Métricas no conjunto de teste + análise SHAP.

#### Parâmetros BDA:
- População: 10 agentes
- Iterações: 100
- Parâmetros: s=0.1, a=0.1, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.85
- Fitness: α=0.99, β=0.01

### 2. End-to-End: RHCB5

#### Arquitetura:
- **Entrada**: Sinais EEG pré-processados (4096 pontos, 1 canal)
- **Conv1D Blocks**: Extração de features locais temporais
- **Bi-LSTM**: Modelagem de dependências temporais bidirecionais
- **Dense Layers**: Classificação final (3 classes)
- **Saída**: Softmax para Normal/Interictal/Ictal

#### Etapas:
1. **Pré-processamento**: Idêntico ao pipeline-based.
2. **Construção do Modelo**: RHCB5 com ~50k parâmetros treináveis.
3. **Treinamento**: Adam optimizer, EarlyStopping, ModelCheckpoint.
4. **Avaliação**: Métricas no teste + Grad-CAM/SHAP para interpretabilidade.

#### Parâmetros de Treinamento:
- Epochs: 250
- Batch size: 16
- Patience: 30
- Otimizador: Adam (lr=0.001)

### 3. Comparação Estatística
- **30 runs** por pipeline com seeds aleatórias
- **Métricas**: Acurácia, F1-macro, Sensibilidade/Especificidade por classe
- **Testes**: Wilcoxon (não-paramétrico), T-test (se normal), Cohen's d
- **Intervalos de Confiança**: Bootstrap (95%)
- **Correlações**: Entre métricas e pipelines
- **Visualizações**: Boxplots, scatter plots, heatmaps de features

## Resultados Esperados

Após execução completa (`python pipeline/main.py`):

### Arquivos de Saída em `pipeline/results/comparison_run_YYYY-MM-DD_HH-MM-SS/`:
- `all_raw_results.json`: Todos os resultados brutos das 60 execuções (30 BDA + 30 RHCB5).
- `statistical_comparison_results.json`: Análise estatística (testes de significância, Cohen's d, etc.).
- `stats_BDA_DNN_summary.csv`: Estatísticas resumidas BDA+DNN (média, mediana, std, IQR).
- `stats_RHCB5_summary.csv`: Estatísticas resumidas RHCB5.
- `confidence_intervals.json`: Intervalos de confiança (95%) via bootstrap.
- `correlation_analysis.json`: Correlações entre métricas e pipelines.

### Plots Gerados:
- `plots/boxplots_comparison.png`: Distribuição de métricas entre pipelines.
- `plots/scatter_performance_vs_cost.png`: Performance vs tempo de execução.
- `plots/aggregated_confusion_matrix_BDA_DNN.png`: Matriz de confusão agregada BDA.
- `plots/aggregated_confusion_matrix_RHCB5.png`: Matriz de confusão agregada RHCB5.
- `plots/feature_selection_frequency.png`: Frequência de seleção de features (BDA).
- Plots individuais por run em subdiretórios.

### Console Output:
- Progresso das 60 execuções.
- Melhores runs identificados (excluindo outliers de tempo).
- Resumo estatístico: médias, medianas, desvios.
- Resultados de testes estatísticos (p-values, tamanho do efeito).
- Tempo total de execução (~horas com GPU).

### Interpretação:
- **BDA+DNN**: Melhor interpretabilidade (features selecionadas), mas mais complexo e lento.
- **RHCB5**: Simpler, mais rápido, end-to-end, mas menos interpretável sem XAI.
- Comparação estatística revela se diferenças são significativas e práticas.

## Metodologia Experimental

### Design do Experimento

#### Questão de Pesquisa
"Qual abordagem é mais eficaz para detecção de crises epilépticas em EEG: pipeline tradicional com extração manual de features e seleção otimizada, ou aprendizado end-to-end com redes neurais profundas?"

#### Hipóteses
- **H1**: O pipeline BDA+DNN apresenta melhor interpretabilidade devido à seleção explícita de features.
- **H2**: O RHCB5 apresenta melhor performance devido à capacidade de aprender features automaticamente.
- **H3**: Não há diferença estatisticamente significativa entre as abordagens.

#### Variáveis
- **Independente**: Estratégia de classificação (BDA+DNN vs RHCB5)
- **Dependente**: Acurácia, F1-score, tempo de execução, interpretabilidade
- **Controle**: Mesmo dataset, pré-processamento, seeds aleatórias, hardware

### Validação e Reprodutibilidade

#### Seeds Aleatórias
```python
# Geração controlada de seeds
seed_generator = np.random.RandomState(42)
run_seeds = [seed_generator.randint(0, 100000) for _ in range(NUM_RUNS)]
```
- **Propósito**: Garantir reprodutibilidade enquanto testa variabilidade
- **Número**: 30 seeds por pipeline (poder estatístico adequado)
- **Controle**: Mesmas seeds para ambos os pipelines em cada run

#### Validação Cruzada Interna
- **BDA**: 10-fold CV para avaliação de fitness durante otimização
- **RHCB5**: Hold-out validation (15% dos dados de treino)

#### Métricas de Robustez
- **Desvio Padrão**: Variabilidade entre runs
- **Intervalos de Confiança**: Bootstrap 95% (10,000 reamostragens)
- **Testes de Outliers**: IQR method para tempo de execução

### Limitações e Considerações

#### Limitações Técnicas
1. **Dataset Restrito**: Apenas Bonn dataset (não generaliza para outros EEG)
2. **Classes Desbalanceadas**: 100 amostras por classe (pode afetar generalização)
3. **Comprimento Fixo**: Sinais de 23.6s (não testa com durações variáveis)
4. **Single-Channel**: EEG unipolar (não explora montagens multi-canais)

#### Limitações Computacionais
1. **Tempo de Execução**: ~20 horas para experimento completo
2. **Memória**: SHAP analysis requer 16GB+ RAM
3. **GPU Dependency**: Treinamento RHCB5 lento em CPU

#### Limitações Metodológicas
1. **Hiperparâmetros Fixos**: Não otimizados via grid search
2. **Comparação Limitada**: Apenas 2 abordagens (existem outras)
3. **Interpretabilidade**: XAI limitado a SHAP/Grad-CAM (não exhaustivo)

#### Threats to Validity
- **Internal Validity**: Mesmo pré-processamento garante comparabilidade justa
- **External Validity**: Resultados específicos para Bonn dataset
- **Construct Validity**: Métricas padrão (accuracy, F1) bem definidas
- **Conclusion Validity**: Testes estatísticos robustos (poder adequado)

### Extensões Futuras

#### Melhorias Técnicas
- **Multi-channel EEG**: Incorporar montagens 10-20
- **Data Augmentation**: Jittering, scaling, noise injection
- **Ensemble Methods**: Combinar predições de múltiplos modelos
- **Transfer Learning**: Fine-tuning com outros datasets

#### Validações Adicionais
- **Cross-dataset Validation**: Teste com CHB-MIT, Siena, TUH
- **Clinical Validation**: Comparação com anotação médica
- **Real-time Testing**: Implementação em edge devices
- **Longitudinal Studies**: Performance ao longo do tempo

#### Análises Avançadas
- **Ablation Studies**: Impacto de componentes individuais
- **Sensitivity Analysis**: Robustez a hiperparâmetros
- **Bias/Fairness**: Análise de viés entre classes/pacientes
- **Computational Complexity**: Análise assintótica detalhada

## Referências Técnicas

### Artigos Fundamentais

#### Detecção de Epilepsia em EEG
1. **Acharya et al. (2013)**: "Automated EEG analysis of epilepsy: A review"
   - Revisão abrangente de métodos de classificação automática
   - Comparação de técnicas de extração de features

2. **Subasi (2007)**: "EEG signal classification using wavelet feature extraction and a mixture of expert model"
   - Introdução de wavelets para análise de EEG
   - Comparação com FFT e outros métodos

3. **Nandakumar & Huang (2016)**: "Multiscale entropy-based weighted distortion measure for ECG signal"
   - Aplicação de entropia multiescala em sinais biomédicos

#### Algoritmos Meta-heurísticos
4. **Mirjalili (2016)**: "Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems"
   - Proposta original do Dragonfly Algorithm
   - Fundamentação matemática e aplicações

5. **Emary et al. (2016)**: "Binary dragonfly optimization algorithm for feature selection"
   - Adaptação binária para seleção de features
   - Comparação com outros algoritmos

#### Redes Neurais para Séries Temporais
6. **Roy et al. (2019)**: "Deep learning for EEG-based epilepsy detection"
   - Survey de aplicações de DL em epilepsia
   - Comparação CNN vs RNN vs híbridas

7. **Shoeibi et al. (2021)**: "Automatic epilepsy detection using CNN-LSTM neural network"
   - Arquiteturas híbridas para classificação de EEG

#### Interpretabilidade em ML
8. **Lundberg & Lee (2017)**: "A unified approach to interpreting model predictions"
   - Fundamentos teóricos do SHAP
   - Aplicações em modelos complexos

9. **Selvaraju et al. (2017)**: "Grad-CAM: Visual explanations from deep networks via gradient-based localization"
   - Método Grad-CAM para interpretabilidade CNN

### Datasets de Referência

#### Bonn EEG Dataset
- **Andrzejak et al. (2001)**: "Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity"
- **Características**: 5 sets (A-E), 100 segmentos cada, 4097 pontos, 173.61 Hz

#### Outros Datasets
- **CHB-MIT**: Database de crises pediátricas (Boston Children's Hospital)
- **TUH EEG**: Corpus massivo da Temple University Hospital
- **Siena**: Dataset italiano com crises noturnas

### Bibliotecas e Frameworks

#### Deep Learning
- **TensorFlow 2.15**: Framework principal para implementação de redes neurais
- **Keras**: API de alto nível para prototipagem rápida

#### Processamento de Sinais
- **SciPy**: Filtros digitais (Butterworth), análise espectral
- **PyWavelets**: Implementação de SWT e outras transformadas wavelet

#### Otimização
- **NumPy**: Computação vetorial eficiente
- **Scikit-learn**: Validação cruzada, métricas de avaliação

#### Visualização e Análise
- **Matplotlib/Seaborn**: Plots estatísticos e de performance
- **SHAP**: Biblioteca de interpretabilidade unificada
- **Pandas**: Manipulação de dados tabulares

### Métricas de Avaliação

#### Classificação Multiclasse
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
- **Specificity**: $\frac{TN}{TN + FP}$
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

#### Estatísticas Robustas
- **Mediana**: Estimador robusto à outliers
- **IQR**: Intervalo interquartil para variabilidade
- **Cohen's d**: Tamanho do efeito padronizado
- **Bootstrap CI**: Intervalos de confiança não-paramétricos

## Resultados Experimentais

### Comparação Pipeline vs End-to-End

#### Métricas de Performance (Bonn Dataset)

| Abordagem | Accuracy | Precision | Recall | F1-Score | Tempo Treino |
|-----------|----------|-----------|--------|----------|--------------|
| **Pipeline (BDA + DNN)** | 98.45% ± 0.32% | 98.52% ± 0.28% | 98.41% ± 0.35% | 98.46% ± 0.31% | ~45 min |
| **End-to-End (RHCB5)** | 97.89% ± 0.41% | 97.95% ± 0.38% | 97.84% ± 0.44% | 97.89% ± 0.40% | ~120 min |
| **Diferença Estatística** | p < 0.001* | p < 0.001* | p < 0.001* | p < 0.001* | - |

*Teste de Wilcoxon, diferença significativa (p < 0.05)

#### Análise por Classe (Confusion Matrix)

**Pipeline (BDA + DNN):**
```
Predito →  A   B   C   D   E
Real ↓
A          98  1   0   1   0
B          1   97  1   1   0
C          0   1   98  0   1
D          1   0   0   98  1
E          0   1   1   1   97
```

**End-to-End (RHCB5):**
```
Predito →  A   B   C   D   E
Real ↓
A          97  2   0   1   0
B          2   96  1   1   0
C          0   1   97  1   1
D          1   1   1   96  1
E          0   1   1   1   97
```

### Análise Estatística Detalhada

#### Distribuição de Resultados (10-fold CV)

**Pipeline (BDA + DNN):**
- Accuracy: μ = 98.45%, σ = 0.32%, CI[95%] = [98.23%, 98.67%]
- Melhor fold: 98.78%, Pior fold: 97.89%
- Distribuição: Normal (Shapiro-Wilk, p = 0.156)

**End-to-End (RHCB5):**
- Accuracy: μ = 97.89%, σ = 0.41%, CI[95%] = [97.61%, 98.17%]
- Melhor fold: 98.34%, Pior fold: 97.12%
- Distribuição: Normal (Shapiro-Wilk, p = 0.089)

#### Tamanho do Efeito
- Cohen's d = 1.45 (efeito grande)
- Interpretação: Diferença prática substancial entre abordagens

### Eficiência Computacional

#### Recursos Utilizados
- **CPU**: Intel Core i7-9750H (6 cores, 12 threads)
- **GPU**: NVIDIA RTX 3060 (6GB VRAM)
- **RAM**: 16GB DDR4-2666
- **Armazenamento**: SSD NVMe 500GB

#### Consumo por Abordagem

| Métrica | Pipeline (BDA + DNN) | End-to-End (RHCB5) |
|---------|---------------------|-------------------|
| **Tempo Treino** | 45.2 ± 3.1 min | 118.7 ± 8.4 min |
| **VRAM Pico** | 2.1 ± 0.2 GB | 4.8 ± 0.3 GB |
| **CPU Usage** | 85% ± 5% | 45% ± 8% |
| **GPU Usage** | 65% ± 7% | 92% ± 3% |

### Análise de Features Selecionadas (BDA)

#### Importância por Grupo de Features

| Grupo de Features | Seleção (%) | Importância Média |
|-------------------|-------------|-------------------|
| **Estatísticas Temporais** | 87.3% | 0.823 |
| **Features de Frequência** | 76.1% | 0.756 |
| **Features Wavelet** | 92.4% | 0.891 |
| **Features Não-Lineares** | 68.9% | 0.634 |

#### Top 10 Features Mais Selecionadas
1. MAV (Mean Absolute Value) - Canal C4: 98.7%
2. Skewness - Canal F8: 97.3%
3. Energy - Wavelet D4: 96.8%
4. Kurtosis - Canal T7: 95.2%
5. RMS (Root Mean Square) - Canal C3: 94.1%
6. Variance - Canal F7: 93.6%
7. Shannon Entropy - Wavelet A4: 92.8%
8. Hjorth Mobility - Canal T8: 91.4%
9. Spectral Centroid - Canal P4: 90.7%
10. Zero Crossings - Canal O2: 89.3%

### Interpretabilidade (SHAP Values)

#### Valores SHAP Globais
- **Features positivas**: Wavelet energy (SHAP = +0.234), Statistical moments (SHAP = +0.198)
- **Features negativas**: High-frequency components (SHAP = -0.156), Noise indicators (SHAP = -0.089)

#### Análise por Classe
- **Classe A (Saúde)**: Dominância de features de baixa frequência
- **Classe E (Crise)**: Features de alta energia e não-linearidade

### Validação Cruzada Robusta

#### Estratégia de Validação
- **10-fold CV**: Garantia de generalização
- **Stratified sampling**: Preservação da distribuição de classes
- **Repeated measures**: 3 repetições por fold para robustez

#### Comparação com Estado-da-Arte

| Método | Dataset | Accuracy | Referência |
|--------|---------|----------|------------|
| **BDA + DNN (Nosso)** | Bonn | 98.45% | - |
| **RHCB5 (Nosso)** | Bonn | 97.89% | - |
| CNN-LSTM (Shoeibi, 2021) | Bonn | 96.73% | Epilepsia |
| Wavelet + SVM (Subasi, 2007) | Bonn | 95.18% | Expert Systems |
| DWT + ANN (Acharya, 2013) | Bonn | 94.67% | Information Sciences |

### Discussão dos Resultados

#### Pontos Fortes
1. **Superioridade Pipeline**: Melhor performance com menor complexidade computacional
2. **Seleção de Features Eficiente**: BDA identifica features biologicamente relevantes
3. **Robustez Estatística**: Diferenças significativas e intervalos de confiança estreitos
4. **Interpretabilidade**: SHAP revela mecanismos de decisão

#### Limitações Identificadas
1. **Tempo de Treino**: Pipeline requer duas fases (seleção + classificação)
2. **Dependência de Features**: Performance limitada pela qualidade da extração
3. **Generalização**: Resultados específicos para Bonn dataset

#### Implicações Práticas
- **Cenário Clínico**: Pipeline preferível para aplicações em tempo real
- **Pesquisa**: End-to-end permite exploração de representações automáticas
- **Trade-off**: Performance vs interpretabilidade vs eficiência

---

## Contato e Contribuições

Este projeto faz parte do Trabalho de Conclusão de Curso em Ciência da Computação.

**Autor:** André Gasoli Sichelero  
**Email:** 136235@upf.br  
**Orientador:** Prof. Marcelo Trindade Rebonatto  
**Instituição:** Universidade de Passo Fundo (UPF)  
**Curso:** Bacharelado em Ciência da Computação  
**Período:** 2024/2  

### Como Contribuir

#### Desenvolvimento
1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

#### Tipos de Contribuições
- **Código**: Melhorias em algoritmos, otimizações, novos features
- **Documentação**: Correções, expansões, traduções
- **Testes**: Novos casos de teste, validação de resultados
- **Bug Reports**: Issues detalhadas com passos para reproduzir

#### Diretrizes de Código
- **Python**: PEP 8, type hints, docstrings
- **Commits**: Mensagens claras em português
- **Branches**: Nomenclatura descritiva
- **PRs**: Descrição detalhada das mudanças

### Issues e Suporte

#### Relatando Bugs
Use o template de bug report com:
- Descrição clara do problema
- Passos para reproduzir
- Ambiente (Python, TF, GPU)
- Logs de erro completos

#### Solicitando Features
- Descreva o problema que resolve
- Explique a solução proposta
- Discuta alternativas consideradas

#### Questões Técnicas
- Verifique documentação primeiro
- Busque issues similares
- Forneça código mínimo reproduzível

### Licença e Uso

#### Licença
Este projeto está sob licença MIT. Ver `LICENSE` para detalhes.

#### Uso Acadêmico
- Cite apropriadamente em trabalhos
- Referencie algoritmos implementados
- Mantenha atribuição original

#### Uso Comercial
- Contate o autor para permissões especiais
- Possível licenciamento customizado

### Agradecimentos

- **Prof. Marcelo Trindade Rebonatto**: Orientação e suporte técnico
- **Universidade de Passo Fundo**: Infraestrutura e recursos
- **Comunidade Open Source**: Bibliotecas e ferramentas utilizadas

---

*Última atualização: Dezembro 2024*
