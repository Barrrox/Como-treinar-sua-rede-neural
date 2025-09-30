# Projeto de Classificação de Obras de Arte

Este projeto utiliza uma Rede Neural Convolucional (CNN) com módulos inspirados na arquitetura Inception para classificar imagens de obras de arte em diferentes movimentos artísticos.

Todo o pipeline, desde o pré-processamento dos dados até o treinamento e a otimização de hiperparâmetros, é controlado por um arquivo de configuração central (`config.yaml`), permitindo a fácil execução e reprodutibilidade dos experimentos.

## Estrutura do Projeto

```
/
├── Dados/                  # Contém os datasets brutos e processados (.npy)
├── Treinamento/
│   ├── megaformat.py       # Script para pré-processar as imagens brutas
│   ├── treinamento_inception.py  # Script principal para treinar o modelo
│   ├── otimizacao_parametros_inception.py # Script para buscar os melhores hiperparâmetros
│   ├── carregar_modelo.py  # Script para avaliar um modelo já treinado
│   └── utilidade.py        # Módulo auxiliar (ex: carregar config)
│
├── config.yaml             # Arquivo central de configuração de parâmetros
└── requirements.txt        # (Recomendado) Lista de dependências do projeto
```

## Guia de Execução

Siga os passos abaixo para preparar o ambiente e treinar o modelo.

### 1. Preparação do Ambiente

Antes de executar os scripts, certifique-se de que todas as bibliotecas necessárias estão instaladas. É altamente recomendado criar um ambiente virtual.

```bash
# Instale as dependências
pip install tensorflow scikit-learn numpy opencv-python pyyaml tqdm matplotlib
```

### 2. Configuração do Experimento (`config.yaml`)

O arquivo `config.yaml` é o centro de controle do projeto. Antes de treinar, ajuste os parâmetros nesta seção conforme sua necessidade.

Para evitar problemas, realize todos os 4 procedimentos de treinamento (formatação do database, otimização, treinamento e carregamento de modelo) utilizando os mesmos parâmetros configurados previamente no **config.yaml**.

#### Parâmetros Chave para Alterar:

* **`dados`**:
    * `caminho_base_de_dados`: Caminho para a pasta contendo as imagens originais, separadas por classes.
    * `arquivo_imagens_treino` / `arquivo_labels_treino`: Nomes dos arquivos `.npy` que serão gerados pelo pré-processamento e usados no treinamento. **Mude esses nomes para cada novo dataset**.
    * `tamanho_imagem`: A resolução para a qual as imagens serão redimensionadas (ex: 224 para 224x224).
    * `num_canais`: `3` para imagens coloridas (RGB) ou `1` para escala de cinza.

* **`treinamento`**:
    * `tam_testes` / `tam_validacao`: Proporção dos dados a ser separada para os conjuntos de teste e validação (ex: 0.1 para 10%).
    * `epocas`: Número máximo de épocas para o treinamento.
    * `batch_size`: Quantidade de imagens processadas por vez em cada etapa do treinamento.
    * `early_stopping_patience`: Limite de épocas que o treinamento executará sem melhora do `val_loss`. 

* **`modelo`**:
    * `caminho_melhor_modelo`: Nome do arquivo `.keras` para salvar o modelo com a melhor performance durante o treinamento. **Mude esse nome para cada novo experimento** para não sobrescrever modelos anteriores.

### 3. Execução do Pipeline

Os scripts devem ser executados a partir da **pasta raiz do projeto** , utilizando a flag `-m` do Python para que as importações funcionem corretamente.

#### Passo 1: Pré-processamento dos Dados

Este script lê as imagens da pasta definida em `caminho_base_de_dados`, as processa (crop, resize) e as salva nos arquivos `.npy` definidos em `arquivo_imagens_treino` e `arquivo_labels_treino`. **Lembre-se** de escolher o tamanho do resize e o número de canais no arquivo de configuração.

```bash
python ./Treinamento/megaformat.py
```

#### Passo 2: Treinamento do Modelo

Após o pré-processamento, este script carrega os arquivos `.npy`, monta o modelo e inicia o treinamento com os parâmetros definidos em `config.yaml`.

```bash
python ./Treinamento/treinamento_inception.py
```

O modelo com o melhor desempenho será salvo no caminho especificado por `caminho_melhor_modelo` nas configurações.

### 3. Otimização de Hiperparâmetros

Se desejar encontrar uma combinação otimizada de parâmetros (como `batch_size`, `optimizer`, etc.), você pode executar o script de otimização. Ele usará as configurações da seção `otimizacao` do `config.yaml`.

```bash
python ./Treinamento/otimizacao_parametros_inception.py
```
> **Atenção:** Este processo pode ser muito demorado.

### 4. Avaliação de um Modelo Salvo

Para carregar um modelo já treinado e visualizar sua matriz de confusão e predições, execute o seguinte comando:

```bash
python ./Treinamento/carregar_modelo.py
```
> Certifique-se de que o parâmetro `caminho_melhor_modelo` no `config.yaml` aponta para o arquivo `.keras` que você deseja carregar.