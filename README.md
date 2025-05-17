[To read in English, click here](#english-version)

# Detecção de Crises Epilépticas Baseada em EEG usando BDFA/BPSO e DNN

Este projeto implementa e avalia pipelines para a detecção automática de crises epilépticas a partir de sinais de Eletroencefalograma (EEG). Ele utiliza a Transformada Wavelet Estacionária (SWT) para extração de características, algoritmos de otimização meta-heurísticos binários - Binary Dragonfly Algorithm (BDA) e Binary Particle Swarm Optimization (BPSO) - para seleção de características, e uma Rede Neural Profunda (DNN) para classificação. O objetivo é classificar os sinais de EEG em três categorias: Normal, Interictal (entre crises) e Ictal (durante a crise), utilizando o dataset público da Universidade de Bonn.

Este trabalho é inspirado e busca implementar conceitos apresentados no artigo:
* Yogarajan, G., Alsubaie, N., Rajasekaran, G. et al. EEG-based epileptic seizure detection using binary dragonfly algorithm and deep neural network. *Sci Rep* **13**, 17710 (2023). [https://doi.org/10.1038/s41598-023-44318-w](https://doi.org/10.1038/s41598-023-44318-w)

## Funcionalidades

* Carregamento e pré-processamento de dados do dataset BONN (Conjuntos A, D, E).
* Filtragem de sinal (Butterworth passa-baixas 0-40Hz) e normalização Min-Max.
* Extração de 9 características estatísticas e de Hjorth de 5 sub-bandas da SWT (wavelet 'db4', nível 4), totalizando 45 características.
    * Características: Valor Médio Absoluto (MAV), Desvio Padrão, Assimetria (Skewness), Curtose, Potência RMS, Razão dos MAVs (com MAV(cA4) como denominador), Atividade, Mobilidade e Complexidade.
* Seleção de características otimizada usando:
    * Binary Dragonfly Algorithm (BDA)
    * Binary Particle Swarm Optimization (BPSO) (para comparação)
* Classificação dos estados de EEG usando uma Rede Neural Profunda (DNN) do tipo Perceptron Multicamadas (MLP).
* Avaliação comparativa das pipelines BDA+DNN e BPSO+DNN em termos de acurácia, sensibilidade, especificidade e F1-score.

## Estrutura do Projeto
```
epilepsy_detection_project/
|-- data/                  # Diretório para o dataset BONN
|   |-- Set A/             # Arquivos .txt para EEG Normal
|   |-- Set D/             # Arquivos .txt para EEG Interictal
|   |-- Set E/             # Arquivos .txt para EEG Ictal
|-- src/                   # Código fonte dos módulos
|   |-- init.py
|   |-- data_loader.py       # Carregamento e pré-processamento de dados
|   |-- feature_extractor.py # Extração de características SWT
|   |-- dnn_model.py         # Definição do modelo DNN
|   |-- fitness_function.py  # Função de aptidão unificada para otimizadores
|   |-- bda.py               # Implementação do Binary Dragonfly Algorithm
|   |-- bpso.py              # Implementação do Binary Particle Swarm Optimization
|   |-- utils.py             # Funções utilitárias (métricas, plots)
|-- results/               # Saída de resultados, modelos salvos, gráficos
|-- main.py                # Script principal para executar o pipeline completo
|-- README.md              # Este arquivo
|-- requirements.txt
```

## Configuração e Instalação

### Pré-requisitos
* Python 3.10
* pip
* Git

### Passos

1.  **Clonar o Repositório (se aplicável):**
    ```bash
    git clone [https://github.com/andresichelero/TCC.git](https://github.com/andresichelero/TCC.git)
    cd epilepsy_detection_project
    ```

2.  **Criar e Ativar um Ambiente Virtual:**
    ```bash
    python -m venv venv
    # No Windows
    .\venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar Dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo (pacotes essenciais):
    ```
    numpy
    pandas
    scipy
    scikit-learn
    pywavelets
    tensorflow>=2.15 # Use uma versão compatível com sua GPU/CUDA
    matplotlib
    tqdm
    ```
    Então, instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração da GPU (Opcional, para TensorFlow):**
    Para utilizar uma GPU NVIDIA com TensorFlow:
    * Instale os drivers NVIDIA mais recentes.
    * Instale o CUDA Toolkit e o cuDNN compatíveis com sua versão do TensorFlow e driver. Consulte a [documentação oficial do TensorFlow sobre suporte a GPU](https://www.tensorflow.org/install/gpu).
    O script `main.py` tentará usar a GPU automaticamente se configurada corretamente.

## Uso

O script principal `main.py` orquestra todo o pipeline.

1.  **Configurar Parâmetros (Opcional):**
    Você pode ajustar os parâmetros principais diretamente no arquivo `main.py`, como:
    * `T_MAX_ITER_OPTIMIZERS`: Número de iterações para BDA e BPSO (recomenda-se 100, conforme o artigo, mas levará cerca de **1000 min** para uma rodada completa - testado com Ryzen 5600X & RTX 3070).
    * `N_AGENTS_OPTIMIZERS`: Tamanho da população para BDA/BPSO (10, conforme o artigo).
    * Parâmetros específicos do BDA e BPSO (conforme detalhado abaixo).
    * Parâmetros de treinamento da DNN (epochs, batch size, patience para EarlyStopping).
    * `RANDOM_SEED` para reprodutibilidade.

2.  **Executar o Script Principal:**
    Certifique-se de que seu ambiente virtual está ativado.
    ```bash
    python main.py
    ```

3.  **O que Esperar:**
    * O script irá carregar os dados, pré-processá-los, extrair as 45 características SWT.
    * Em seguida, executará o BDA e o BPSO para selecionar os subconjuntos de características ótimos. A função de fitness treinará e validará modelos DNN internamente.
    * Após a seleção, modelos DNN finais serão treinados usando os conjuntos de características selecionados por BDA e BPSO.
    * Os modelos finais serão avaliados no conjunto de teste.
    * O output no console mostrará o progresso, logs de depuração (se `DEBUG_FEATURES = True` em `feature_extractor.py`), resultados de fitness, e as métricas finais de desempenho.
    * Uma curva de convergência dos otimizadores será plotada (se `matplotlib` estiver configurado para funcionar em seu ambiente).
    * Resultados detalhados e os modelos DNN finais treinados serão salvos no diretório `results/`.

## Descrição dos Módulos (`src/`)

* `data_loader.py`: Contém funções para carregar os dados do dataset BONN, aplicar pré-processamento (filtragem Butterworth, normalização Min-Max) e dividir os dados em conjuntos de treino, validação e teste.
* `feature_extractor.py`: Responsável pela extração das 9 características estatísticas e de Hjorth das 5 sub-bandas obtidas pela Transformada Wavelet Estacionária (SWT).
* `dnn_model.py`: Define a arquitetura da Rede Neural Profunda (MLP com 3 camadas ocultas de 10 neurônios sigmoides cada) e sua compilação.
* `fitness_function.py`: Implementa a função de aptidão unificada usada pelos algoritmos BDA e BPSO. Esta função treina e avalia a DNN com um subconjunto de características para guiar o processo de otimização.
* `bda.py`: Implementação do Binary Dragonfly Algorithm (BDA) para seleção de características.
* `bpso.py`: Implementação do Binary Particle Swarm Optimization (BPSO) para seleção de características.
* `utils.py`: Funções utilitárias, incluindo cálculo de métricas de classificação (acurácia, especificidade, relatório de classificação) e plotagem de curvas de convergência.

## Parâmetros dos Otimizadores (Conforme Artigo e Implementação)

### Binary Dragonfly Algorithm (BDA)
Baseado em Yogarajan et al. (2023) e na implementação:
* `population_size (N)`: 10
* `iterations (T)`: 100 (50 recomendado)
* `s` (peso da separação): 0.1
* `a` (peso do alinhamento): 0.1
* `c_cohesion` (peso da coesão): 0.7
* `f_food` (fator de atração pela comida): 1.0
* `e_enemy` (fator de distração do inimigo): 1.0
* `w_inertia` (peso de inércia): 0.85 (fixo)
* `tau_min`: 0.01 (para função de transferência V-Shaped)
* `tau_max`: 4.0 (para função de transferência V-Shaped)
* A atualização da posição da libélula usa uma função V-Shaped (ex: `abs(tanh(Passo / tau))`) para determinar a probabilidade de inverter um bit.
* Os componentes de Separação, Alinhamento e Coesão são calculados considerando as outras libélulas na população.

### Binary Particle Swarm Optimization (BPSO)
Baseado no PDF de implementação e valores comuns:
* `population_size (N)`: 10
* `iterations (T)`: 100 (50 recomendado)
* `w_max` (inércia máxima): 0.9
* `w_min` (inércia mínima): 0.4 (peso de inércia decresce linearmente)
* `c1` (coeficiente cognitivo): 2.0
* `c2` (coeficiente social): 2.0
* `Vmax` (limite da velocidade): 4.0 (opcional, mas recomendado)
* A atualização da posição da partícula usa uma função de transferência Sigmoide aplicada à velocidade para determinar a probabilidade do bit ser 1.

### Função de Aptidão (Comum a ambos)
* $Fitness = \alpha \cdot \text{taxaDeErro} + \beta \cdot (\text{numFeaturesSel} / \text{totalNumFeatures})$
* $\alpha = 0.99$
* $\beta = 0.01$

## Possíveis Melhorias e Trabalhos Futuros

* **Ajuste Fino de Hiperparâmetros:** Otimizar os parâmetros do BDA, BPSO e da arquitetura/treinamento da DNN.
* **Explorar Diferentes Wavelets e Níveis de Decomposição SWT.**
* **Testar com Outros Datasets de EEG.**
* **Comparar com Outras Meta-heurísticas de Seleção de Características.**
* **Utilizar Arquiteturas de DNN Mais Avançadas:** Como Redes Neurais Convolucionais (CNNs) ou Redes Neurais Recorrentes (RNNs/LSTMs), que podem ser mais adequadas para dados de série temporal como EEG.

---
<a id="english-version"></a>
## English Version

# EEG-Based Epileptic Seizure Detection using BDA/BPSO and DNN

This project implements and evaluates pipelines for the automatic detection of epileptic seizures from Electroencephalogram (EEG) signals. It utilizes the Stationary Wavelet Transform (SWT) for feature extraction, binary meta-heuristic optimization algorithms - Binary Dragonfly Algorithm (BDA) and Binary Particle Swarm Optimization (BPSO) - for feature selection, and a Deep Neural Network (DNN) for classification. The goal is to classify EEG signals into three categories: Normal, Interictal (between seizures), and Ictal (during a seizure), using the public University of Bonn dataset.

This work is inspired by and aims to implement concepts presented in the paper:
* Yogarajan, G., Alsubaie, N., Rajasekaran, G. et al. EEG-based epileptic seizure detection using binary dragonfly algorithm and deep neural network. *Sci Rep* **13**, 17710 (2023). [https://doi.org/10.1038/s41598-023-44318-w](https://doi.org/10.1038/s41598-023-44318-w)

## Features

* Loading and preprocessing data from the BONN dataset (Sets A, D, E).
* Signal filtering (Butterworth low-pass 0-40Hz) and Min-Max normalization.
* Extraction of 9 statistical and Hjorth features from 5 SWT sub-bands (wavelet 'db4', level 4), totaling 45 features.
    * Features: Mean Absolute Value (MAV), Standard Deviation, Skewness, Kurtosis, RMS Power, Ratio of MAVs (with MAV(cA4) as denominator), Activity, Mobility, and Complexity.
* Optimized feature selection using:
    * Binary Dragonfly Algorithm (BDA)
    * Binary Particle Swarm Optimization (BPSO) (for comparison)
* Classification of EEG states using a Deep Neural Network (DNN) of the Multilayer Perceptron (MLP) type.
* Comparative evaluation of BDA+DNN and BPSO+DNN pipelines in terms of accuracy, sensitivity, specificity, and F1-score.

## Project Structure
```
epilepsy_detection_project/
|-- data/                      # Directory for the BONN dataset
|   |-- Set A/                 # .txt files for Normal EEG
|   |-- Set D/                 # .txt files for Interictal EEG
|   |-- Set E/                 # .txt files for Ictal EEG
|-- src/                       # Source code of the modules
|   |-- init.py
|   |-- data_loader.py         # Data loading and preprocessing
|   |-- feature_extractor.py   # SWT feature extraction
|   |-- dnn_model.py           # DNN model definition
|   |-- fitness_function.py    # Unified fitness function for optimizers
|   |-- bda.py                 # Implementation of the Binary Dragonfly Algorithm
|   |-- bpso.py                # Implementation of the Binary Particle Swarm Optimization
|   |-- utils.py               # Utility functions (metrics, plots)
|-- results/                   # Output of results, saved models, plots
|-- main.py                    # Main script to run the complete pipeline
|-- README.md                  # This file
|-- requirements.txt
```

## Setup and Installation

### Prerequisites
* Python 3.10
* pip
* Git

### Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone [https://github.com/andresichelero/TCC.git](https://github.com/andresichelero/TCC.git)
    cd epilepsy_detection_project
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (essential packages):
    ```
    numpy
    pandas
    scipy
    scikit-learn
    pywavelets
    tensorflow>=2.15 # Use a version compatible with your GPU/CUDA
    matplotlib
    tqdm
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU Configuration (Optional, for TensorFlow):**
    To use an NVIDIA GPU with TensorFlow:
    * Install the latest NVIDIA drivers.
    * Install CUDA Toolkit and cuDNN compatible with your TensorFlow version and driver. Refer to the [official TensorFlow documentation on GPU support](https://www.tensorflow.org/install/gpu).
    The `main.py` script will attempt to use the GPU automatically if configured correctly.

## Usage

The main script `main.py` orchestrates the entire pipeline.

1.  **Configure Parameters (Optional):**
    You can adjust the main parameters directly in the `main.py` file, such as:
    * `T_MAX_ITER_OPTIMIZERS`: Number of iterations for BDA and BPSO (100 is recommended as per the paper, but it will take about **1000 min** for a complete round - tested with Ryzen 5600X & RTX 3070).
    * `N_AGENTS_OPTIMIZERS`: Population size for BDA/BPSO (10, as per the paper).
    * Specific parameters for BDA and BPSO (as detailed below).
    * DNN training parameters (epochs, batch size, patience for EarlyStopping).
    * `RANDOM_SEED` for reproducibility.

2.  **Run the Main Script:**
    Make sure your virtual environment is activated.
    ```bash
    python main.py
    ```

3.  **What to Expect:**
    * The script will load the data, preprocess it, and extract the 45 SWT features.
    * Next, it will run BDA and BPSO to select the optimal feature subsets. The fitness function will train and validate DNN models internally.
    * After selection, final DNN models will be trained using the feature sets selected by BDA and BPSO.
    * The final models will be evaluated on the test set.
    * The console output will show progress, debug logs (if `DEBUG_FEATURES = True` in `feature_extractor.py`), fitness results, and final performance metrics.
    * A convergence curve for the optimizers will be plotted (if `matplotlib` is configured to work in your environment).
    * Detailed results and the trained final DNN models will be saved in the `results/` directory.

## Module Descriptions (`src/`)

* `data_loader.py`: Contains functions to load data from the BONN dataset, apply preprocessing (Butterworth filtering, Min-Max normalization), and split the data into training, validation, and test sets.
* `feature_extractor.py`: Responsible for extracting the 9 statistical and Hjorth features from the 5 sub-bands obtained by the Stationary Wavelet Transform (SWT).
* `dnn_model.py`: Defines the architecture of the Deep Neural Network (MLP with 3 hidden layers of 10 sigmoid neurons each) and its compilation.
* `fitness_function.py`: Implements the unified fitness function used by the BDA and BPSO algorithms. This function trains and evaluates the DNN with a subset of features to guide the optimization process.
* `bda.py`: Implementation of the Binary Dragonfly Algorithm (BDA) for feature selection.
* `bpso.py`: Implementation of the Binary Particle Swarm Optimization (BPSO) for feature selection.
* `utils.py`: Utility functions, including calculation of classification metrics (accuracy, specificity, classification report) and plotting convergence curves.

## Optimizer Parameters (As per Paper and Implementation)

### Binary Dragonfly Algorithm (BDA)
Based on Yogarajan et al. (2023) and the implementation:
* `population_size (N)`: 10
* `iterations (T)`: 100 (50 recommended)
* `s` (separation weight): 0.1
* `a` (alignment weight): 0.1
* `c_cohesion` (cohesion weight): 0.7
* `f_food` (food attraction factor): 1.0
* `e_enemy` (enemy distraction factor): 1.0
* `w_inertia` (inertia weight): 0.85 (fixed)
* `tau_min`: 0.01 (for V-Shaped transfer function)
* `tau_max`: 4.0 (for V-Shaped transfer function)
* The dragonfly position update uses a V-Shaped function (e.g., `abs(tanh(Step / tau))`) to determine the probability of flipping a bit.
* The Separation, Alignment, and Cohesion components are calculated considering the other dragonflies in the population.

### Binary Particle Swarm Optimization (BPSO)
Based on implementation PDF and common values:
* `population_size (N)`: 10
* `iterations (T)`: 100 (50 recommended)
* `w_max` (maximum inertia): 0.9
* `w_min` (minimum inertia): 0.4 (inertia weight decreases linearly)
* `c1` (cognitive coefficient): 2.0
* `c2` (social coefficient): 2.0
* `Vmax` (velocity limit): 4.0 (optional, but recommended)
* The particle position update uses a Sigmoid transfer function applied to the velocity to determine the probability of the bit being 1.

### Fitness Function (Common to both)
* $Fitness = \alpha \cdot \text{errorRate} + \beta \cdot (\text{numSelectedFeatures} / \text{totalNumFeatures})$
* $\alpha = 0.99$
* $\beta = 0.01$

## Possible Improvements and Future Work

* **Hyperparameter Fine-Tuning:** Optimize BDA, BPSO parameters, and DNN architecture/training.
* **Explore Different Wavelets and SWT Decomposition Levels.**
* **Test with Other EEG Datasets.**
* **Compare with Other Feature Selection Metaheuristics.**
* **Utilize More Advanced DNN Architectures:** Such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs/LSTMs), which may be more suitable for time-series data like EEG.
