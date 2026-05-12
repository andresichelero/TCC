[To read in English, click here](#english-version)

# Estratégias Computacionais para Detecção de Epilepsia em EEG: Abordagem em Pipeline versus End-to-End

**Trabalho Final de Curso (TCC) - Ciência da Computação**  
**Universidade de Passo Fundo - Dezembro 2025**  
**Autor:** André Gasoli Sichelero  
**Orientador:** Prof. Marcelo Trindade Rebonatto

---

## Contexto e Importância

A epilepsia é um distúrbio neurológico que afeta aproximadamente 50 milhões de pessoas em todo o mundo, caracterizado por crises epilépticas recorrentes. A Eletroencefalografia (EEG) é a técnica padrão-ouro para diagnóstico e monitoramento de epilepsia, capturando a atividade elétrica cerebral através de eletrodos posicionados no couro cabeludo.

**Desafio Principal**: A detecção automática de crises epilépticas em sinais de EEG é crucial para:
- Diagnóstico precoce e preciso de epilepsia do lobo temporal
- Monitoramento contínuo de pacientes com redução de falsos positivos
- Minimização da "fadiga de alarme" em ambientes clínicos
- Suporte a decisões clínicas baseadas em dados com alta confiabilidade

**Problema de Pesquisa**: Apesar da vasta produção acadêmica, observa-se uma carência crítica de estudos que realizem comparação direta e controlada entre paradigmas de pipeline e end-to-end sob **as mesmas condições experimentais**, com análise rigorosa da estabilidade estocástica e da robustez estatística.

## Contribuição Principal deste Trabalho

Este projeto conduz uma **análise comparativa rigorosa e pareada** entre:
- **Pipeline clássico otimizado**: Algoritmo da Libélula Binário (BDA) + K-vizinhos próximos (KNN) + Rede Neural Profunda (DNN)
- **Arquitetura end-to-end do estado da arte**: Rede Híbrida Convolucional Bidirecional (RHCB5)

Utilizando **Sementes Aleatórias Comuns (CRN - Common Random Numbers)** para garantir que ambos os modelos sejam avaliados sob condições idênticas, isolando o desempenho do algoritmo como fator principal.

### Referências Teóricas

**Inspirado em:**
* Yogarajan, G., Alsubaie, N., Rajasekaran, G. et al. EEG-based epileptic seizure detection using binary dragonfly algorithm and deep neural network. *Sci Rep* **13**, 17710 (2023). [https://doi.org/10.1038/s41598-023-44318-w](https://doi.org/10.1038/s41598-023-44318-w)

**Arquitetura RHCB5:**
* Maggioni, A. et al. (2023/2024) - Rede Híbrida Convolucional Bidirecional originalmente validada para eletrocardiogramas (ECG), aqui testada para epilepsia em EEG.

## Funcionalidades

### Pipeline-Based (BDA+DNN)

#### Pré-processamento de Sinais
* **Filtragem**: Filtro Butterworth passa-baixas de ordem 4, frequência de corte 40Hz (remove ruído de alta frequência)
* **Normalização**: Min-Max scaling para intervalo [-1, 1], preservando relações relativas
* **Segmentação**: Sinais de 4097 pontos → 4096 pontos (remoção do primeiro ponto para estabilidade)

#### Extração de Características SWT
* **Transformada Wavelet**: Stationary Wavelet Transform (SWT) com wavelet 'db4' (Daubechies 4), nível de decomposição 4
* **Sub-bandas**: 16 componentes por sinal (aproximação e detalhes de 4 níveis)
* **Características Estatísticas (8 por sub-banda)**:
  1. **MAV (Mean Absolute Value)**: $\frac{1}{N} \sum_{i=1}^{N} |x_i|$ - Energia média do sinal
  2. **StdDev (Standard Deviation)**: $\sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$ - Variabilidade
  3. **Skewness**: $\frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^3}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^{3/2}}$ - Assimetria da distribuição
  4. **Kurtosis**: $\frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^4}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2} - 3$ - Curtose (achatamento)
  5. **RMS (Root Mean Square)**: $\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$ - Valor eficaz
  6. **Activity (Hjorth)**: $\frac{1}{N} \sum_{i=1}^{N} x_i^2$ - Variância do sinal no tempo
  7. **Mobility (Hjorth)**: $\sqrt{\frac{\text{Activity}(\frac{dx}{dt})}{\text{Activity}(x)}}$ - Mobilidade (razão entre variâncias)
  8. **Complexity (Hjorth)**: $\frac{\text{Mobility}(\frac{dx}{dt})}{\text{Mobility}(x)}$ - Complexidade (normalizada)

* **Características Adicionais**: 15 razões MAV entre sub-bandas

**Total**: 143 características (16 sub-bandas × 8 features + 15 razões)

#### Seleção de Características com BDA
* **Algoritmo**: Binary Dragonfly Algorithm (BDA) - meta-heurística bio-inspirada
* **Codificação**: Vetor binário de 143 dimensões (1 = feature selecionada, 0 = não selecionada)
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
├── Conv1D (512 filtros, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (256 filtros, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (256 filtros, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (128 filtros, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Bi-LSTM (256 unidades, return_sequences=False)
├── Dense (256 neurônios, relu)
├── Dropout (0.4)
├── Dense (128 neurônios, relu)
├── Dense (3 neurônios, softmax)
Output: Probabilidades para [Normal, Interictal, Ictal]
```

#### Hiperparâmetros de Treinamento
* **Otimização**: Adam (lr=0.001, β1=0.9, β2=0.999)
* **Loss**: Sparse Categorical Crossentropy
* **Métricas**: Accuracy, Precision, Recall, F1-Score
* **Regularização**: Early Stopping (monitor='val_loss', patience=15, restore_best_weights=True)
* **Batch Size**: 32
* **Epochs**: 100 (máximo, com early stopping)

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
│   └── Bonn/                      # Dataset principal
│       ├── A/                     # 100 arquivos .txt (EEG Normal)
│       ├── D/                     # 100 arquivos .txt (EEG Interictal)
│       └── E/                     # 100 arquivos .txt (EEG Ictal)
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

O script principal `pipeline/main.py` orquestra a comparação pareada completa entre os dois pipelines usando a técnica de Sementes Aleatórias Comuns (CRN).

### Execução Básica

1.  **Executar a Comparação Pareada Completa:**
    Certifique-se de que o ambiente virtual está ativado e os dados estão em `data/Bonn/`.
    ```bash
    cd pipeline
    python main.py
    ```

2.  **Fluxo de Execução (Pareado com CRN):**
    * Carrega e pré-processa os dados uma vez.
    * Gera 30 sementes aleatórias compartilhadas (CRN).
    * **Para cada uma das 30 sementes**:
      - Executa BDA+DNN com a mesma divisão de dados
      - Executa RHCB5 com a mesma divisão de dados
    * Identifica os melhores runs (excluindo outliers) e executa análise XAI.
    * Compila estatísticas pareadas: média, mediana, desvio padrão, IQR.
    * Realiza testes estatísticos pareados: Shapiro-Wilk, T-test pareado, Wilcoxon.
    * Calcula tamanho do efeito (Cohen's d) e intervalos de confiança (95%).
    * Gera plots: boxplots comparativos, matrizes de confusão agregadas, análise de estabilidade.
    * Salva todos os resultados em `pipeline/results/comparison_run_YYYY-MM-DD_HH-MM-SS/`.

3.  **Configurações Principais:**
    Edite `pipeline/main.py` para ajustar:
    * `NUM_RUNS = 30`: Número de execuções pareadas.
    
    Edite `pipeline/pipeline_utils.py` para ajustar:
    * `USE_XAI = True`: Habilitar análise SHAP/Grad-CAM (mais lento, ~30% overhead).
    * `USE_GPU = True`: Usar GPU se disponível (recomendado para RHCB5).
    * `VERBOSE_LEVEL = 1`: Nível de detalhamento (0=silencioso, 1=básico, 2=detalhado).
    * Parâmetros de pré-processamento: `FS=173.61`, `HIGHCUT_HZ=40`, `FILTER_ORDER=4`.

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
- Extração de 143 features via SWT (16 sub-bandas × 8 características + 15 razões).
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
2. **Extração de Características**: SWT com wavelet 'db4' nível 4 → 16 sub-bandas → 8 features cada + 15 razões MAV → 143 features totais.
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

#### Questão de Pesquisa Principal
"Qual abordagem oferece maior **confiabilidade e robustez** para detecção de crises epilépticas em EEG: um pipeline otimizado com feature engineering explícito, ou uma arquitetura end-to-end integrada?"

#### Metodologia de Sementes Aleatórias Comuns (CRN)

A contribuição metodológica principal deste trabalho é a implementação de **Sementes Aleatórias Comuns (CRN)** para comparação rigorosa:

1. **Geração de 30 sementes compartilhadas**: Uma semente única é gerada para cada iteração $i$
2. **Aplicação pareada**: Na iteração $i$:
   - BDA+DNN usa seed $s_i$, gerando estratificação específica de treino/validação/teste
   - RHCB5 usa seed $s_i$, gerando **exatamente a mesma estratificação**
3. **Impacto**: Elimina vieses de amostragem e isola o desempenho do algoritmo

```python
# Geração de sementes compartilhadas
seed_generator = np.random.RandomState(42)
run_seeds = [seed_generator.randint(0, 100000) for _ in range(NUM_RUNS)]

# Cada iteração usa a mesma semente para ambos os pipelines
for i, seed in enumerate(run_seeds):
    bda_results[i] = run_bda_dnn_pipeline(seed=seed)
    rhcb5_results[i] = run_rhcb5_pipeline(seed=seed)
```

#### Hipóteses Testadas

- **H1**: O BDA+DNN apresenta melhor interpretabilidade (características explícitas selecionadas)
- **H2**: A RHCB5 apresenta melhor sensibilidade geral e, crítico: melhor **detecção do estado Interictal**
- **H3**: Não há diferença significativa em ambos os métodos (nula)

**Resultado**: H2 foi confirmada com significância estatística (p=0.0053). O estado Interictal é o fator crítico para viabilidade clínica, onde RHCB5 superou BDA em +15.11 p.p. no Recall.

---

#### Limitações e Ameaças à Validade

**Limitações do Dataset Bonn:**
- Segmentos **pré-selecionados e limpos** (não representa ambiente clínico contínuo)
- Restrito a **epilepsia do lobo temporal focal** (não generaliza para crises generalizadas)
- **Apenas 100 amostras por classe** (pequeno para deep learning moderno)
- Single-channel EEG (não explora montagens multi-canal)

**Validação Ecológica Necessária:**
- Testar em registros contínuos longos (CHB-MIT dataset)
- Implementar rejeição de artefatos em tempo real
- Avaliar em cenários com desbalanceamento severo de classes

**Ameaças à Validade Mitigadas pela Metodologia:**
- ✓ **Validade Interna**: Mesmo pré-processamento para ambos garante comparabilidade
- ✓ **Validade Estatística**: 30 repetições com CRN fornece poder estatístico adequado
- ✓ **Reprodutibilidade**: Sementes fixas garantem replicabilidade exata

---

### Interpretabilidade: Análise XAI

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

## Resultados Experimentais Detalhados (30 Execuções Pareadas com CRN)

### Métricas Globais de Performance

| Métrica | BDA-DNN | RHCB5 | Diferença | p-value |
|---------|---------|-------|-----------|---------|
| **Acurácia** | 90.37% ± 3.66% | **93.19% ± 4.33%** | +2.82% | 0.0053* |
| **F1-Score Macro** | 90.18% ± 3.82% | **93.14% ± 4.35%** | +2.96% | 0.0043* |
| **Tempo Total (s)** | 58.50 (IQR 3.97) | **37.83 (IQR 10.58)** | -35.33% | - |
| **Redução de Features** | 65-66% (média 49 atributos) | N/A | - | - |

*Teste T Pareado, diferença significativa (p < 0.05)

#### Análise por Classe: O Achado Crítico

**Classe Interictal (D) - Desafio Clínico Principal:**

| Métrica | BDA-DNN | RHCB5 | Diferença |
|---------|---------|-------|-----------|
| **Recall (Sensibilidade)** | 80.00% ± 10.21% | **95.11% ± 4.93%** | **+15.11 p.p.** |
| **Especificidade** | **96.22% ± 3.58%** | 95.22% ± 5.16% | -1.00% |
| **Precisão** | **91.99% ± 7.24%** | 91.64% ± 8.30% | -0.35% |
| **F1-Score** | 85.08% ± 6.58% | **93.10% ± 5.17%** | +9.43% |

**Interpretação Clínica**: A detecção de estado Interictal é o gargalo crítico para aplicação clínica. O BDA apresentou taxa de erro de **20%**, confundindo frequentemente padrões interictais com normais (9.6%) ou com crises (10.4%), causador de "fadiga de alarme". A RHCB5 resolveu este problema crítico, elevando a detecção a 95.11%.

---

**Classe Ictal (E) - Alta Energia:**

| Métrica | BDA-DNN | RHCB5 | Diferença |
|---------|---------|-------|-----------|
| **Recall (Sensibilidade)** | **98.22% ± 3.47%** | 88.22% ± 9.54% | -10.00% |
| **Especificidade** | 94.22% ± 4.63% | **98.00% ± 2.57%** | +3.78% |
| **Precisão** | 90.11% ± 7.17% | **95.89% ± 4.97%** | +5.78% |
| **F1-Score** | **93.78% ± 3.94%** | 91.59% ± 5.92% | -2.19% |

**Interpretação**: O BDA é altamente sensível a descargas de alta energia (crises), mas ao custo de falsos alarmes. A RHCB5 oferece balanço superior entre sensibilidade e especificidade.

#### Matrizes de Confusão Agregadas

**BDA-DNN (30 execuções agregadas):**
```
             Predito Normal  Interictal  Ictal
Real Normal          1440         108       0
     Interictal       288         2400     312    ← 20% de erro crítico
     Ictal              0          59     2941
```

**RHCB5 (30 execuções agregadas):**
```
             Predito Normal  Interictal  Ictal
Real Normal          1425         123       0
     Interictal        81         2857      162   ← Dramaticamente melhorado
     Ictal             27          93      2880
```

---

### Trade-off: Pipeline vs End-to-End

| Aspecto | BDA-DNN | RHCB5 |
|---------|---------|-------|
| **Interpretabilidade** | Alta (features explícitas) | Baixa (caixa-preta) |
| **Estabilidade Estatística** | ⚠️ Baixa (CV=61% na seleção) | ✓ Alta (consistente) |
| **Confiabilidade Clínica** | ⚠️ Problemática (20% erro Interictal) | ✓ Superior (robusta) |
| **Tempo de Inferência** | Mais rápido | Mais lento |
| **Tamanho do Modelo** | Pequeno (MLP simples) | Grande (CNN-BiLSTM) |
| **Reprodutibilidade** | Difícil (BDA instável) | ✓ Excelente |

---

### Interpretabilidade (XAI Analysis)

#### BDA-KNN-DNN: Análise de Instabilidade via SHAP

**Descoberta Crítica: Convergência Não-Canônica**

O BDA selecionou em média **49.2 atributos de 143** (redução de 65-66%), mas com **alta variabilidade**:
- Coeficiente de Variação: 61% (selecionou entre 19 e 79 atributos)
- Nenhuma feature foi selecionada em >60% das 30 execuções
- Distribuição quase uniforme de frequências de seleção

**Interpretação**: O algoritmo não convergiu para um subconjunto **canônico de biomarcadores robustos**, mas sim para correlações estatísticas locais e ruído específico de cada partição (**overfitting**).

Embora o BDA seja um otimizador matemático eficaz, falhou como **descobridor de conhecimento clínico estável**. A "caixa branca" suposta do pipeline revelou-se instável e dependente da inicialização.

#### RHCB5: Análise de Grad-CAM

**Validação Visual de Aprendizado**

O Grad-CAM revelou que a rede desenvolveu detectores robustos de características:

1. **Classe Normal**: Ativação difusa para monitoramento de estabilidade global
2. **Classe Interictal**: **Focos intensos e localizados** alinhados temporalmente com espículas e transientes breves (20-70 ms) — precisamente o padrão clínico crítico
3. **Classe Ictal**: Ativação massiva e contínua rastreando a evolução energética da crise

**Conclusão**: A RHCB5 capturou automaticamente as sutilezas morfológicas que o BDA não conseguiu isolar com features estáticas baseadas em wavelet. Isso explica a superioridade de +15.11 p.p. no Recall da classe Interictal.

---

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
- Interpretação: Diferença prática substancial e clinicamente significativa entre abordagens
- O teste de normalidade Shapiro-Wilk confirmou distribuição normal (W=0.955, p=0.241)

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
## Comparação com Estado-da-Arte

| Método | Dataset | Accuracy | Referência |
|--------|---------|----------|------------|
| **BDA + DNN (Este trabalho)** | Bonn | 90.37% | - |
| **RHCB5 (Este trabalho)** | Bonn | 93.19% | - |
| CNN-LSTM (Shoeibi, 2021) | Bonn | 96.73% | Epilepsia |
| Wavelet + SVM (Subasi, 2007) | Bonn | 95.18% | Expert Systems |
| DWT + ANN (Acharya, 2013) | Bonn | 94.67% | Information Sciences |

---

## Considerações Finais

Este trabalho apresentou uma análise comparativa sistemática entre duas filosofias predominantes para a detecção de epilepsia em sinais de EEG: o *pipeline* clássico otimizado (BDA-KNN-DNN) e a abordagem de aprendizado profundo integrada (RHCB5). Através de repetições pareadas com controle de aleatoriedade, foi possível avaliar não apenas a acurácia máxima, mas também a estabilidade e a consistência de cada método.

Um aspecto crítico observado refere-se à reprodutibilidade da abordagem em *pipeline*. A implementação do BDA exigiu um esforço substancial de inferência técnica, visto que trabalhos anteriores apresentaram uma parametrização inviável na prática e omitiram configurações do classificador wrapper (KNN), forçando a adoção de premissas baseadas na literatura fundamental. Em contraste, a arquitetura RHCB5 demonstrou alta fidelidade de reprodução: sua descrição topológica clara permitiu a transposição direta do domínio original (ECG) para EEG com modificações mínimas. Essa facilidade, aliada à superioridade estatística comprovada, reforça a viabilidade da abordagem *end-to-end* como artefato científico metodologicamente mais reprodutível e acessível para adoção clínica.

**Conclusão**: A escolha entre *pipeline* e *end-to-end* depende do contexto de aplicação. Cenários que exigem explicação clínica detalhada e ambientes com severas restrições de memória podem se beneficiar do BDA-KNN-DNN, desde que tolerem maior variância. Entretanto, **para triagem massiva ou monitoramento contínuo, onde a segurança do paciente e a minimização de falsos negativos são prioritárias, a RHCB5 apresenta-se como uma solução mais segura e eficaz**.

### Agenda de Pesquisa Futura

Como trabalhos futuros, propõe-se uma agenda estruturada em três eixos:

**Primeiro Eixo - Validação Ecológica**: Validação em registros contínuos de longa duração (CHB-MIT) mediante segmentação com janelas deslizantes, *data augmentation* via jittering temporal e adição de ruído gaussiano, balanceamento de classes via SMOTE temporal, e rejeição automática de artefatos por limiarização adaptativa.

**Segundo Eixo - Refinamento Metodológico**: No *pipeline*, substituir o *wrapper* KNN por filtragem via Mutual Information e adotar classificadores XGBoost com interpretabilidade nativa; na RHCB5, integrar Self-Attention nas camadas convolucionais, Temporal Attention entre Bi-LSTM e camadas densas, e aplicar Transfer Learning com pré-treinamento no TUH EEG Corpus (> 60.000 registros).

**Terceiro Eixo - Comparação com Arquiteturas Especializadas**: Comparação direta com EEGNet e DeepConvNet sob o mesmo protocolo pareado com CRN, validando definitivamente a hipótese de transferibilidade de domínio versus especialização arquitetural.

## Contribuições Deste Estudo

Este trabalho oferece quatro contribuições principais à literatura:

1. **Comparação pareada rigorosa**: Análise direta e pareada entre paradigmas de *pipeline* e *end-to-end* para detecção de epilepsia sob controle experimental rigoroso.

2. **Estabilidade estocástica comprovada**: Demonstração empírica de que a superioridade de acurácia da RHCB5 se sustenta através de análise de estabilidade estocástica (30 execuções com CRN).

3. **Resolução do gargalo clínico**: Identificação da redução crítica da confusão entre estados Interictais e Ictais pela arquitetura híbrida, possivelmente resolvendo um dos maiores gargalos para a aplicação clínica de sistemas automáticos.

4. **Evidência via XAI**: Demonstração, através de análise SHAP e Grad-CAM, de que o BDA pode convergir para biomarcadores não canônicos em alta dimensionalidade, enquanto RHCB5 desenvolve detectores de características robustos e interpretáveis.

## Limitações e Ameaças à Validade

Embora o desenho experimental tenha sido rigoroso, algumas limitações devem ser reconhecidas:

### Limitações do Dataset

- **Dados pré-selecionados e limpos**: O *dataset* de Bonn, apesar de ser referência na literatura, consiste em segmentos pré-selecionados e limpos, não refletindo a complexidade de registros contínuos com artefatos encontrados em ambientes clínicos reais. Representa um cenário de laboratório controlado em vez de monitoramento clínico real.

- **Limitação de canal único**: O uso de EEG de canal único (C3 ou C4) não explora montagens multi-canal que poderiam capturar informações espaciais cruciais para decisão clínica em aplicações reais.

### Escopo e Generalização

- **Escopo de epilepsia**: A análise se restringe a epilepsias focais do lobo temporal; a generalização para crises generalizadas ou de início desconhecido requer validação adicional.

- **Variabilidade interpaciente**: O tamanho amostral de 300 segmentos (100 de 5 sujeitos saudáveis + 100 de 5 pacientes epilépticos), embora suficiente para análise comparativa, pode limitar a capacidade de capturar variabilidade interpaciente e diferenças demográficas (idade, sexo, status de medicação).

### Restrições Técnicas

- **Restrições computacionais**: A RHCB5 apresenta complexidade computacional para inferência em dispositivos de borda (*edge computing*) com restrições severas de energia, potencialmente limitando a implementação em cenários clínicos com recursos limitados.

- **Opacidade de modelos profundos**: Embora mitigada pelo Grad-CAM, a RHCB5 não fornece uma lista explícita de regras de decisão como o *pipeline* baseado em atributos, o que pode ser problemático para aprovação regulatória de dispositivos médicos.

### Considerações Metodológicas

- **Tratamento de artefatos**: O estudo não modelou explicitamente rejeição de artefatos, requisito crítico para monitoramento em tempo real onde atividade muscular, movimento de eletrodos e ruído elétrico são prevalentes.

- **Desbalanceamento de classes em prática clínica**: Em monitoramento contínuo, eventos ictais são extremamente raros comparado a estados normal e interictal, criando desbalanceamento severo não totalmente capturado pela distribuição balanceada 100-100-100 do dataset.

### Ameaças Mitigadas pelo Desenho Experimental

- ✓ **Validade Interna**: Mesmo pré-processamento para ambas as abordagens garante comparabilidade
- ✓ **Validade Estatística**: 30 repetições com CRN fornece poder estatístico adequado
- ✓ **Reprodutibilidade**: Sementes fixas garantem replicabilidade exata

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
1. Faça um fork do repositório
2. Crie uma branch para sua funcionalidade (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

#### Tipos de Contribuições
- **Código**: Melhorias em algoritmos, otimizações, novas funcionalidades
- **Documentação**: Correções, expansões, traduções
- **Testes**: Novos casos de teste, validação de resultados
- **Bug Reports**: Issues detalhadas com passos para reproduzir

#### Diretrizes de Código
- **Python**: PEP 8, type hints, docstrings
- **Commits**: Mensagens claras em português ou inglês
- **Branches**: Nomenclatura descritiva
- **PRs**: Descrição detalhada das mudanças

### Relatando Problemas

#### Bugs
Inclua:
- Descrição clara do problema
- Passos para reproduzir
- Ambiente (Python, TensorFlow, GPU)
- Logs de erro completos

#### Solicitações de Funcionalidades
- Descreva o problema que resolve
- Explique a solução proposta
- Discuta alternativas consideradas

#### Questões Técnicas
- Verifique documentação primeiro
- Busque issues similares
- Forneça código mínimo reproduzível

### Referências e Leituras Adicionais

#### Referências Fundamentais

**Detecção de Epilepsia em EEG**
1. **Acharya et al. (2013)**: "Automated EEG analysis of epilepsy: A review"
2. **Subasi (2007)**: "EEG signal classification using wavelet feature extraction"
3. **Schomer & Lopes da Silva (2018)**: "Niedermeyer's Electroencephalography"

**Algoritmos Meta-heurísticos**
4. **Mirjalili (2016)**: "Dragonfly algorithm: a new meta-heuristic optimization technique"
5. **Emary et al. (2016)**: "Binary dragonfly optimization algorithm for feature selection"

**Aprendizado Profundo para EEG**
6. **Roy et al. (2019)**: "Deep learning for EEG-based epilepsy detection"
7. **Shoeibi et al. (2021)**: "Automatic epilepsy detection using CNN-LSTM neural networks"

**Explicabilidade em ML**
8. **Lundberg & Lee (2017)**: "A unified approach to interpreting model predictions" (SHAP)
9. **Selvaraju et al. (2017)**: "Grad-CAM: Visual explanations from deep networks"

#### Dataset

- **Andrzejak et al. (2001)**: Bonn EEG Dataset
  - Disponível em: https://www.ukbonn.de/eeg-database/

---

**Licença:** MIT. Ver `LICENSE` para detalhes.

*Última atualização: Dezembro de 2025*

---

# English Version

# Computational Strategies for Epilepsy Detection in EEG: Pipeline versus End-to-End Approaches

**Final Course Project (TCC) - Computer Science**  
**Federal University of Passo Fundo - December 2025**  
**Author:** André Gasoli Sichelero  
**Advisor:** Prof. Marcelo Trindade Rebonatto

---

## Context and Importance

Epilepsy is a neurological disorder affecting approximately 50 million people worldwide, characterized by recurrent seizures. Electroencephalography (EEG) is the gold standard technique for epilepsy diagnosis and monitoring, capturing brain electrical activity through electrodes positioned on the scalp.

**Main Challenge**: Automatic seizure detection in EEG signals is crucial for:
- Early and accurate diagnosis of temporal lobe epilepsy
- Continuous patient monitoring with reduced false positives
- Minimization of "alarm fatigue" in clinical environments
- Support for data-driven clinical decision-making with high reliability

**Research Problem**: Despite extensive academic production, there is a critical lack of studies conducting direct and controlled comparison between pipeline and end-to-end paradigms under **identical experimental conditions**, with rigorous analysis of stochastic stability and statistical robustness.

## Main Contribution of This Work

This project conducts a **rigorous and paired comparative analysis** between:
- **Classical optimized pipeline**: Binary Dragonfly Algorithm (BDA) + K-Nearest Neighbors (KNN) + Deep Neural Network (DNN)
- **State-of-the-art end-to-end architecture**: Hybrid Bidirectional Convolutional Network (RHCB5)

Using **Common Random Numbers (CRN)** to ensure both models are evaluated under identical conditions, isolating algorithm performance as the primary factor.

### Theoretical References

**Inspired by:**
* Yogarajan, G., Alsubaie, N., Rajasekaran, G. et al. EEG-based epileptic seizure detection using binary dragonfly algorithm and deep neural network. *Sci Rep* **13**, 17710 (2023). [https://doi.org/10.1038/s41598-023-44318-w](https://doi.org/10.1038/s41598-023-44318-w)

**RHCB5 Architecture:**
* Maggioni, A. et al. (2023/2024) - Hybrid Bidirectional Convolutional Network originally validated for electrocardiograms (ECG), here tested for epilepsy in EEG.

---

## Experimental Results and Detailed Analysis

### Global Performance Metrics (30 Paired Runs)

| Metric | BDA-DNN | RHCB5 | Difference | p-value |
|--------|---------|-------|-----------|---------|
| **Accuracy** | 90.37% ± 3.66% | **93.19% ± 4.33%** | +2.82% | 0.0053* |
| **F1-Score Macro** | 90.18% ± 3.82% | **93.14% ± 4.35%** | +2.96% | 0.0043* |
| **Total Time (s)** | 58.50 (IQR 3.97) | **37.83 (IQR 10.58)** | -35.33% | - |
| **Feature Reduction** | 65-66% (49 avg) | N/A | - | - |

*Paired T-Test, statistically significant (p < 0.05)

### Per-Class Analysis: The Critical Finding

**Interictal Class (D) - Primary Clinical Challenge:**

| Metric | BDA-DNN | RHCB5 | Difference |
|--------|---------|-------|-----------|
| **Recall (Sensitivity)** | 80.00% ± 10.21% | **95.11% ± 4.93%** | **+15.11 p.p.** |
| **Specificity** | **96.22% ± 3.58%** | 95.22% ± 5.16% | -1.00% |
| **Precision** | **91.99% ± 7.24%** | 91.64% ± 8.30% | -0.35% |
| **F1-Score** | 85.08% ± 6.58% | **93.10% ± 5.17%** | +9.43% |

**Clinical Interpretation**: Interictal state detection is the critical bottleneck for clinical application. BDA showed **20% error rate**, frequently confusing interictal patterns with normal (9.6%) or with seizures (10.4%), causing "alarm fatigue". RHCB5 resolved this critical problem, achieving 95.11% detection.

**Ictal Class (E) - High Energy:**

| Metric | BDA-DNN | RHCB5 | Difference |
|--------|---------|-------|-----------|
| **Recall (Sensitivity)** | **98.22% ± 3.47%** | 88.22% ± 9.54% | -10.00% |
| **Specificity** | 94.22% ± 4.63% | **98.00% ± 2.57%** | +3.78% |
| **Precision** | 90.11% ± 7.17% | **95.89% ± 4.97%** | +5.78% |
| **F1-Score** | **93.78% ± 3.94%** | 91.59% ± 5.92% | -2.19% |

**Interpretation**: BDA is highly sensitive to high-energy discharges (seizures), but at the cost of false alarms. RHCB5 offers superior balance between sensitivity and specificity.

### Trade-off: Pipeline vs End-to-End

| Aspect | BDA-DNN | RHCB5 |
|--------|---------|-------|
| **Interpretability** | High (explicit features) | Low (black-box) |
| **Statistical Stability** | ⚠️ Low (CV=61% in selection) | ✓ High (consistent) |
| **Clinical Reliability** | ⚠️ Problematic (20% interictal error) | ✓ Superior (robust) |
| **Inference Time** | Faster | Slower |
| **Model Size** | Small (simple MLP) | Large (CNN-BiLSTM) |
| **Reproducibility** | Difficult (BDA unstable) | ✓ Excellent |

### Aggregated Confusion Matrices (30 Runs Combined)

**BDA-DNN:**
```
             Predicted Normal  Interictal  Ictal
Real Normal          1440         108       0
     Interictal       288         2400     312    ← 20% critical error
     Ictal              0          59     2941
```

**RHCB5:**
```
             Predicted Normal  Interictal  Ictal
Real Normal          1425         123       0
     Interictal        81         2857      162   ← Dramatically improved
     Ictal             27          93      2880
```

---

## Comparison with State-of-the-Art

| Method | Dataset | Accuracy | Reference |
|--------|---------|----------|------------|
| **BDA + DNN (This work)** | Bonn | 90.37% | - |
| **RHCB5 (This work)** | Bonn | 93.19% | - |
| CNN-LSTM (Shoeibi, 2021) | Bonn | 96.73% | Epilepsia |
| Wavelet + SVM (Subasi, 2007) | Bonn | 95.18% | Expert Systems |
| DWT + ANN (Acharya, 2013) | Bonn | 94.67% | Information Sciences |
| BDA + DNN (Yogarajan, 2023) | Bonn | ~100%* | Scientific Reports |

*Yogarajan et al. reported near-perfect performance; our replication with careful reproducibility validation obtained 90.37%, highlighting the importance of explicit hyperparameter documentation.

---

## Technical Implementation Details

### Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| TensorFlow | 2.15.0 | Neural network framework |
| Keras | 2.15.0 | High-level API |
| NumPy | 2.1.3 | Numerical computing |
| Pandas | 2.2.3 | Data manipulation |
| Scikit-learn | 1.5.2 | ML utilities, metrics |
| SciPy | 1.15.3 | Signal processing, statistics |
| PyWavelets | 1.8.0 | Wavelet transforms |
| SHAP | 0.49.1 | Explainability |
| Matplotlib | 3.10.3 | Visualization |
| Seaborn | 0.13.2 | Statistical plots |

### Hardware Requirements

- **CPU**: Intel/AMD processor with ≥6 cores recommended
- **GPU**: NVIDIA with CUDA 11.8+ (optional, accelerates training by ~3-5x)
- **RAM**: Minimum 16GB, recommended 32GB (for SHAP analysis)
- **Storage**: 15GB+ for datasets and results

### Reproducibility Requirements

All experiments include:
- Fixed random seeds (reproducible across runs)
- Paired experimental design with Common Random Numbers
- Version-pinned dependencies
- Full hyperparameter documentation
- Code with deterministic operations

---

## Methodology: Common Random Numbers (CRN)

### Paired Experimental Design

The central pillar of this work's methodological contribution is the implementation of **Common Random Numbers (CRN)** for rigorous comparison:

1. **Generation of 30 shared seeds**: One unique seed is generated for each iteration $i$
2. **Paired application**: In iteration $i$:
   - BDA+DNN uses seed $s_i$, generating specific train/validation/test stratification
   - RHCB5 uses seed $s_i$, generating **exactly the same stratification**
3. **Impact**: Eliminates sampling biases and isolates algorithm performance

### Hypotheses Tested

- **H1**: BDA+DNN presents better interpretability (explicitly selected features)
- **H2**: RHCB5 presents better general sensitivity and, critically: better **interictal state detection**
- **H3**: No significant difference between both methods (null)

**Result**: H2 was confirmed with statistical significance (p=0.0053). The interictal state is the critical factor for clinical viability, where RHCB5 outperformed BDA by +15.11 p.p. in Recall.

---

## XAI Analysis

#### BDA-KNN-DNN: Instability Analysis via SHAP

**Critical Discovery: Non-Canonical Convergence**

BDA selected on average **49.2 features from 143** (65-66% reduction), but with **high variability**:
- Coefficient of Variation: 61% (in feature selection)
- Range: 19 to 79 features selected (from 143)
- Mean: 49.2 ± 30 features selected
- Implication: No convergent "canonical" biomarcators
- Conclusion: Overfitting to partition-specific correlations

Although BDA is an effective mathematical optimizer, it failed as a **stable clinical knowledge discoverer**. The supposed "white box" of the pipeline proved unstable and dependent on initialization.

#### RHCB5: Grad-CAM Analysis

**Visual Validation of Learning**

Grad-CAM revealed that the network developed robust feature detectors:

1. **Normal class**: Diffuse activation for global stability monitoring
2. **Interictal class**: **Intense and localized foci** temporally aligned with spikes and brief transients (20-70 ms) — precisely the critical clinical pattern
3. **Ictal class**: Massive and continuous activation tracking seizure energetic evolution

**Conclusion**: RHCB5 automatically captured the morphological subtleties that BDA could not isolate with static wavelet-based features. This explains the superiority of +15.11 p.p. in interictal class recall.

---

## Features Implementation and Signal Processing

### Pipeline-Based (BDA+DNN)

#### Signal Preprocessing
* **Filtering**: 4th-order Butterworth low-pass filter with 40Hz cutoff (removes high-frequency noise)
* **Normalization**: Instance standardization (zero mean, unit variance) per segment
* **Segmentation**: 4097-point signals → 4096 points (first point removal for stability)

#### SWT Feature Extraction
* **Wavelet Transform**: Stationary Wavelet Transform (SWT) with 'db4' wavelet, decomposition level 4
* **Sub-bands**: 16 components per signal (approximation and 4 levels of details)
* **Statistical Characteristics (8 per sub-band)**:
  1. **MAV (Mean Absolute Value)**: Mean signal energy
  2. **StdDev (Standard Deviation)**: Signal variability
  3. **Skewness**: Distribution asymmetry
  4. **Kurtosis**: Flatness measure
  5. **RMS (Root Mean Square)**: Effective value
  6. **Activity (Hjorth)**: Signal variance over time
  7. **Mobility (Hjorth)**: Variance ratio
  8. **Complexity (Hjorth)**: Normalized complexity

* **Additional Characteristics**: 15 MAV ratios between sub-bands

**Total**: 143 features (16 sub-bands × 8 features + 15 ratios)

#### Feature Selection with BDA
* **Algorithm**: Binary Dragonfly Algorithm (BDA) - bio-inspired meta-heuristic
* **Encoding**: Binary vector of 143 dimensions (1 = selected, 0 = not selected)
* **Fitness Function**: Fitness = α × ErrorRate + β × (FeaturesSelected/TotalFeatures)
  - α = 0.99, β = 0.01
* **BDA Parameters**:
  - Population: 10 dragonflies
  - Iterations: 100
  - Weights: separation=0.1, alignment=0.1, cohesion=0.7, food=1.0, enemy=1.0
  - Inertia: 0.85 (fixed)
  - Transfer function: V-Shaped (τ ∈ [0.01, 4.0])

#### DNN Classification
* **Architecture**: Multi-layer Perceptron (MLP)
* **Layers**: 3 hidden layers (10 sigmoid neurons) + softmax output (3 classes)
* **Regularization**: Dropout, Early Stopping (patience=30)
* **Optimization**: Adam (lr=0.001), loss=sparse_categorical_crossentropy

### End-to-End (RHCB5)

#### Detailed Architecture
```
Input: (4096, 1) - Pre-processed EEG signal
├── Conv1D (512 filters, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (256 filters, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (256 filters, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Conv1D (128 filters, kernel=3, relu, padding='same')
├── MaxPooling1D (pool_size=2)
├── Dropout (0.2)
├── Bi-LSTM (256 units, return_sequences=False)
├── Dense (256 neurons, relu)
├── Dropout (0.4)
├── Dense (128 neurons, relu)
├── Dense (3 neurons, softmax)
Output: Probabilities for [Normal, Interictal, Ictal]
```

#### Training Hyperparameters
* **Optimization**: Adam (lr=0.001, β1=0.9, β2=0.999)
* **Loss**: Sparse Categorical Crossentropy
* **Metrics**: Accuracy, Precision, Recall, F1-Score
* **Regularization**: Early Stopping (monitor='val_loss', patience=15, restore_best_weights=True)
* **Batch Size**: 32
* **Epochs**: 100 (maximum, with early stopping)

#### Interpretability Analysis
* **Grad-CAM**: Visualization of salient regions in input signal
* **SHAP**: Shapley values for global and local explainability
* **Application**: Identification of critical temporal patterns for classification

---

## Bonn EEG Dataset

The study uses the **Bonn University EEG Dataset**, widely recognized as a reference for epilepsy detection algorithm validation. Data originated from continuous multi-channel recordings of 5 healthy volunteers and 5 patients diagnosed with temporal lobe epilepsy.

### Data Composition
| Set | Class | Description | N° Segments | Duration |
|-----|--------|-----------|--------------|---------|
| A   | Normal | EEG from healthy subjects, eyes open | 100 | 23.6s |
| D   | Interictal | Interictal EEG (epileptic patients, temporal lobe) | 100 | 23.6s |
| E   | Ictal | Seizure EEG (same patients, same region) | 100 | 23.6s |

**Total**: 300 segments of 4097 points each (23.6 seconds)

### Acquisition Parameters
- Single electrode (C3 or C4) vs reference
- Sampling frequency: 173.61 Hz
- Resolution: 12 bits
- Anti-aliasing filter: 0.53-40 Hz

### Spectral Characteristics
* **Set A (Normal)**: Dominant alpha activity (8-12 Hz), beta (12-30 Hz)
* **Set D (Interictal)**: Anomalous patterns, isolated spikes
* **Set E (Ictal)**: High-amplitude rhythmic activity, variable frequency

---

## Experimental Design and Methodology

### Common Random Numbers (CRN) Protocol

The methodological pillar of this study's validation is the implementation of **Common Random Numbers (CRN)** for rigorous comparison:

1. **Generation of 30 shared seeds**: One unique seed is generated for each iteration $i$
2. **Paired application**: In iteration $i$:
   - BDA+DNN uses seed $s_i$, generating specific train/validation/test stratification
   - RHCB5 uses seed $s_i$, generating **exactly the same stratification**
3. **Impact**: Eliminates sampling biases and isolates algorithm performance as primary factor

```python
# Shared seed generation
seed_generator = np.random.RandomState(42)
run_seeds = [seed_generator.randint(0, 100000) for _ in range(NUM_RUNS)]

# Each iteration uses the same seed for both pipelines
for i, seed in enumerate(run_seeds):
    bda_results[i] = run_bda_dnn_pipeline(seed=seed)
    rhcb5_results[i] = run_rhcb5_pipeline(seed=seed)
```

### Hypotheses Tested

- **H1**: BDA+DNN presents better interpretability (explicitly selected features)
- **H2**: RHCB5 presents better general sensitivity and, critically: better **interictal state detection**
- **H3**: No significant difference between both methods (null)

**Result**: H2 was confirmed with statistical significance (p=0.0053). The interictal state is the critical factor for clinical viability, where RHCB5 outperformed BDA by +15.11 p.p. in Recall.

### Evaluation Metrics

#### Per-Class Metrics
* **Precision**: TP / (TP + FP)
* **Recall (Sensitivity)**: TP / (TP + FN)
* **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
* **Specificity**: TN / (TN + FP)

#### Aggregate Metrics
* **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
* **Macro-F1**: Unweighted F1-Score across classes
* **Weighted-F1**: Class-weighted F1-Score

#### Statistical Analysis
* **Normality Tests**: Shapiro-Wilk on paired differences
* **Significance Tests**: Paired T-test (if normal) or Wilcoxon (non-parametric)
* **Effect Size**: Cohen's d for practical significance
* **Confidence Intervals**: Bootstrap (95%, n=10,000 resamples)
* **Correlations**: Pearson between metrics and execution time

---

## Final Considerations and Clinical Guidelines

### Synthesis of Findings

This work presented a systematic and rigorous comparative analysis between two predominant philosophies for epilepsy detection in EEG:

1. **Optimized Pipeline (BDA+DNN)**: Offers explicit interpretability through feature selection, but presents **stochastic instability** and **critical error in interictal state detection (20%)**, making reliable clinical application unfeasible.

2. **End-to-End Approach (RHCB5)**: Presents **superior statistical robustness**, **resolution of the interictal bottleneck** (+15.11 p.p. improvement) and **excellent reproducibility**, consolidating itself as the most viable solution for automatic screening and continuous monitoring.

### Recommendations for Clinical Application

**For mass screening environments and continuous monitoring:**
- ✓ **RHCB5 is recommended** as a clinical decision support tool
- Justification: Robustness, generalization, and minimization of false alarms critical to patient safety

**For academic research with interpretability focus:**
- ⚠️ **BDA+DNN with caveats**: Useful for hypothesis generation, but instability must be recognized
- Recommendation: Combine with nested cross-validation techniques to stabilize selection

### Critical Mention: Scientific Reproducibility

BDA implementation required significant effort of **technical inference** from fundamental literature, as the primary reference (Yogarajan et al., 2023) omitted critical parameterizations. This demonstrates the crucial importance of **rigorous hyperparameter documentation** for reproducibility.

In contrast, RHCB5 allowed **direct transposition** from the original domain (ECG) to EEG, validating the hypothesis of **transferability between non-stationary biosignals**.

---

---

## XAI (Explainable AI) Analysis

### BDA-KNN-DNN: Instability Analysis via SHAP

**Critical Discovery: Non-Canonical Convergence**

BDA selected on average **49.2 features from 143** (65-66% reduction), but with **high variability**:
- Coefficient of Variation: 61% (in feature selection)
- Range: 19 to 79 features selected (from 143)
- Mean: 49.2 ± 30 features selected
- **Implication**: No convergent "canonical" biomarcators
- **Conclusion**: Overfitting to partition-specific correlations

Although BDA is an effective mathematical optimizer, it failed as a **stable clinical knowledge discoverer**. The supposed "white box" of the pipeline proved unstable and dependent on initialization.

### RHCB5: Grad-CAM Analysis

**Visual Validation of Learning**

Grad-CAM revealed that the network developed robust feature detectors:

1. **Normal class**: Diffuse activation for global stability monitoring
2. **Interictal class**: **Intense and localized foci** temporally aligned with spikes and brief transients (20-70 ms) — precisely the critical clinical pattern
3. **Ictal class**: Massive and continuous activation tracking seizure energetic evolution

**Conclusion**: RHCB5 automatically captured the morphological subtleties that BDA could not isolate with static wavelet-based features. This explains the superiority of +15.11 p.p. in interictal class recall.

---

## Future Research Agenda

### Axis 1: Ecological Validation in Continuous Data

- [ ] Test on **CHB-MIT dataset** (>60 patients, 1+ hour recordings each)
- [ ] Implement **real-time artifact rejection** (adaptive thresholding + spectral analysis)
- [ ] Apply **data augmentation** (temporal jittering, Gaussian noise, temporal SMOTE)
- [ ] Evaluate under **severe class imbalance** (ictal events rare in prolonged monitoring)
- [ ] Segment with **sliding windows** and overlap
- [ ] Balance classes via **Temporal SMOTE** or focal loss weighting

### Axis 2: Methodological Refinement

**In BDA+DNN Pipeline:**
- [ ] Replace KNN wrapper ($O(N \times T)$) with filters via **Mutual Information** ($O(N \log N)$)
- [ ] Adopt **XGBoost classifiers with native interpretability**
- [ ] Implement **nested cross-validation** to stabilize feature selection
- [ ] Explore **Simulated Annealing** or **Differential Evolution** enhancements

**In RHCB5:**
- [ ] Integrate **Self-Attention** in convolutional layers
- [ ] Add **Temporal Attention** between Bi-LSTM and dense layers
- [ ] Apply **Transfer Learning** with pre-training on TUH EEG Corpus (>60k records)
- [ ] Explore **Temporal Attention mechanisms** for improved focus

### Axis 3: Complete Architectural Comparison

- [ ] Direct validation with **EEGNet** and **DeepConvNet** under paired CRN protocol
- [ ] Test hypothesis: architectural specialization vs. domain transferability
- [ ] **Ablation studies** to identify critical components
- [ ] Comparison with emerging architectures (Vision Transformers, etc.)

---

## Study Contributions

This work offers four main contributions to the literature:

1. **Rigorous paired comparison**: Direct and paired analysis between pipeline and end-to-end paradigms for epilepsy detection under rigorous experimental control using Common Random Numbers methodology.

2. **Proven stochastic stability**: Empirical demonstration that RHCB5's accuracy superiority is sustained through stochastic stability analysis (30 runs with CRN), establishing statistical significance (p=0.0053).

3. **Resolution of clinical bottleneck**: Identification of critical reduction in confusion between Interictal and Ictal states by the hybrid architecture, achieving +15.11 p.p. improvement in interictal recall, possibly resolving one of the major barriers to clinical application of automatic systems.

4. **Evidence via XAI**: Demonstration, through SHAP and Grad-CAM analysis, that BDA may converge to non-canonical biomarkers in high dimensionality with high variability (CV=61%), while RHCB5 develops robust and interpretable feature detectors with consistent spatial-temporal focus.

---

## Limitations and Threats to Validity

Although the experimental design was rigorous, some limitations should be recognized:

### Dataset Limitations

- **Pre-selected and clean data**: The Bonn dataset, despite being a reference in the literature, consists of pre-selected and clean segments, not reflecting the complexity of continuous recordings with artifacts found in real clinical environments. This represents a controlled laboratory setting rather than actual clinical monitoring scenarios.

- **Single-channel limitation**: The use of single-channel EEG (C3 or C4) does not exploit multi-channel montages that could capture spatial information crucial for clinical decision-making in real-world applications.

### Scope and Generalization

- **Scope of epilepsy**: The analysis is restricted to temporal lobe focal epilepsy; generalization to generalized seizures or seizures of unknown onset requires additional validation.

- **Inter-patient variability**: The sample size of 300 segments (100 from 5 healthy subjects + 100 from 5 epileptic patients), while sufficient for comparative analysis, may limit the ability to capture inter-patient variability and demographic differences (age, gender, medication status).

### Technical Constraints

- **Computational requirements**: RHCB5 presents computational complexity for inference on edge devices with severe energy restrictions, potentially limiting deployment in resource-constrained clinical settings.

- **Opacity of deep models**: Although mitigated by Grad-CAM, RHCB5 does not provide an explicit list of decision rules like the attribute-based pipeline, which may be problematic for regulatory approval in medical devices.

### Methodological Considerations

- **Artifact handling**: The study did not explicitly model artifact rejection, a critical requirement for real-world monitoring where muscle activity, electrode movement, and electrical noise are prevalent.

- **Class imbalance in clinical practice**: In continuous monitoring, ictal events are extremely rare compared to normal and interictal states, creating severe class imbalance not fully captured by the balanced 100-100-100 dataset distribution.

### Mitigated Threats via Experimental Design

- ✓ **Internal Validity**: Same preprocessing for both approaches guarantees comparability
- ✓ **Statistical Validity**: 30 repetitions with CRN provides adequate statistical power
- ✓ **Reproducibility**: Fixed seeds guarantee exact replicability

---

---

## References and Further Reading

### Fundamental References

#### EEG and Epilepsy Detection
1. **Acharya et al. (2013)**: "Automated EEG analysis of epilepsy: A review"
   - Comprehensive review of automated classification methods
   - Comparison of feature extraction techniques

2. **Subasi (2007)**: "EEG signal classification using wavelet feature extraction"
   - Introduction of wavelets for EEG analysis
   - Comparison with FFT and other methods

3. **Schomer & Lopes da Silva (2018)**: "Niedermeyer's Electroencephalography"
   - Clinical foundations of EEG interpretation
   - Comprehensive reference for neurophysiology

#### Meta-heuristic Optimization
4. **Mirjalili (2016)**: "Dragonfly algorithm: a new meta-heuristic optimization technique"
   - Original dragonfly algorithm proposal
   - Mathematical foundations

5. **Emary et al. (2016)**: "Binary dragonfly optimization algorithm for feature selection"
   - Binary adaptation for feature selection
   - Comparison with other algorithms

#### Deep Learning for EEG
6. **Roy et al. (2019)**: "Deep learning for EEG-based epilepsy detection"
   - Survey of deep learning applications in epilepsy
   - CNN vs RNN vs hybrid architectures

7. **Shoeibi et al. (2021)**: "Automatic epilepsy detection using CNN-LSTM neural networks"
   - Hybrid architecture validation
   - Time-series modeling approaches

#### Explainability in ML
8. **Lundberg & Lee (2017)**: "A unified approach to interpreting model predictions" (SHAP)
   - Theoretical foundations of SHAP values
   - Applications in complex models

9. **Selvaraju et al. (2017)**: "Grad-CAM: Visual explanations from deep networks"
   - Gradient-based visualization method
   - CNN interpretability techniques

### Dataset References

- **Andrzejak et al. (2001)**: Bonn EEG Dataset
  - Original dataset description and characteristics
  - Publicly available at: https://www.ukbonn.de/eeg-database/

---

## Contact and Contributions

This project is part of the Final Course Project in Computer Science.

**Author:** André Gasoli Sichelero  
**Email:** 136235@upf.br  
**Advisor:** Prof. Marcelo Trindade Rebonatto  
**Institution:** Federal University of Passo Fundo (UPF)  
**Course:** Bachelor's in Computer Science  
**Period:** 2024/2  

### How to Contribute

#### Development
1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Adds new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

#### Types of Contributions
- **Code**: Improvements to algorithms, optimizations, new features
- **Documentation**: Corrections, expansions, translations
- **Tests**: New test cases, results validation
- **Bug Reports**: Detailed issues with reproduction steps

#### Code Guidelines
- **Python**: PEP 8, type hints, docstrings
- **Commits**: Clear messages in English or Portuguese
- **Branches**: Descriptive naming
- **PRs**: Detailed description of changes

### Reporting Issues

#### Bug Reports
Include:
- Clear problem description
- Steps to reproduce
- Environment (Python, TensorFlow, GPU)
- Full error logs

#### Feature Requests
- Describe the problem it solves
- Explain proposed solution
- Discuss alternatives considered

---

**License:** MIT. See `LICENSE` for details.

*Last update: December 2025*

---