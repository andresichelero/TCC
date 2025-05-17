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
    git clone https://github.com/andresichelero/TCC.git
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
