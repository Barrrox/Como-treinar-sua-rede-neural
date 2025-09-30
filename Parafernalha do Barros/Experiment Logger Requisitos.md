# Experiment Logger

Esse documento visa organizar os requisitos e implementação do Experiment Logger, um código para guardar as informações como parâmetros e resultados de um treinamento. 

# Requisitos

## RF1: Interface do Logger

O Logger deve ser implmentado como uma classe ExperimentLogger que será usado em diferentes etapadas do código (inicio, meio, fim)

## RF2: Geração de experimentos

O Logger deve ter um método para gerar um novo experimento, criando uma pasta nova para guardar o novo experimento.

O sistema DEVE ter métodos para registrar os seguintes artefatos dentro do diretório do experimento:

- RF2.1 - Arquitetura do Modelo: Salvar o sumário da arquitetura do modelo (model.summary()) em um arquivo de texto legível (ex: arquitetura.txt).

- RF2.2 - Parâmetros e Configuração: Salvar um dicionário contendo todos os hiperparâmetros (taxa de aprendizado, épocas, batch size, etc.) e configurações (split dos dados, nome do modelo base) em um arquivo formato JSON (ex: parametros.json).

- RF2.3 - Histórico de Treinamento: Gerar e salvar um gráfico (.png) contendo as curvas de acurácia e perda ao longo das épocas. O gráfico DEVE comparar as métricas de treino e validação. (Correção: A métrica do conjunto de teste é um ponto único, não uma curva por época).

- RF2.4 - Matriz de Confusão: Gerar e salvar uma imagem (.png) da matriz de confusão, calculada sobre o conjunto de teste.

- RF2.5 - Relatório de Resultados: Salvar um resumo com as métricas finais de performance em um formato legível (pode ser no mesmo .json dos parâmetros), incluindo tempo total de treinamento, acurácia e perda finais no conjunto de validação, acurácia e perda finais no conjunto de teste.

## RF3: Gerenciamento de Arquivos

O diretório raiz para todos os experimentos DEVE ser "../Experimentos".

Cada novo experimento DEVE gerar um subdiretório único. O nome do subdiretório DEVE ser baseado em um timestamp (ex: 20250930_143055) para garantir unicidade e ordem cronológica.