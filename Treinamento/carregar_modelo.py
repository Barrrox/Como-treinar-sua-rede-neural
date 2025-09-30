"""
Esse código carrega o modelo e o conjunto de treino para gerar uma
matriz de confusão e fazer a previsão de 9 imagens, uma de cada
movimento artístico

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split # Importar a função de divisão
import matplotlib.pyplot as plt
from matplotlib.widgets import Button # Importar o widget de botão
import numpy as np
import random
from sklearn.metrics import accuracy_score
import os # Importado para construir os caminhos de forma segura


# Carrega as configs/parametros 
from utilidade import carregar_config
config = carregar_config()

# O YAML lida bem com chaves numéricas, então a conversão é direta.
label_para_classe = config['geral']['classes_map']


def carregar_modelo():
    """
    Carrega um modelo Keras treinado, avalia sua performance com uma matriz de confusão
    e exibe uma janela interativa com a predição de 9 imagens, uma de cada categoria.
    """
    caminho_modelo = config['modelo']['caminho_melhor_modelo']
    model = load_model(caminho_modelo)

    model.summary()

    # --- Etapa 1: Geração da Matriz de Confusão ---
    
    caminho_imagens = config['dados']['arquivo_imagens_treino']
    caminho_labels = config['dados']['arquivo_labels_treino']
    
    # Carrega as imagens e os rótulos do conjunto de dados completo.
    imagens = np.load(caminho_imagens)
    labels = np.load(caminho_labels)

    # Separa os dados em um conjunto de treino (para a etapa 2) e um de teste (para a matriz).
    # Este passo é idêntico ao do script de treinamento para garantir consistência.
    x_train, x_test, y_train, y_test = train_test_split(
        imagens, labels, 
        test_size=config["treinamento"]["tam_testes"], 
        random_state=config["geral"]["random_seed"], 
        stratify=labels
    )
    
    # Normaliza os valores dos pixels do conjunto de teste para o intervalo [0, 1].
    x_test_normalized = x_test / 255.0

    # Realiza as predições no conjunto de teste.
    y_pred = np.argmax(model.predict(x_test_normalized), axis=1)

    # Calcula a acurácia comparando os rótulos reais (y_test) com os previstos (y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia (calculada manualmente) no conjunto de teste: {accuracy * 100:.2f}%")

    # Gera a matriz de confusão.
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Pega os nomes das classes do dicionário para usar como rótulos.
    nomes_das_classes = list(label_para_classe.values())

    # 1. Cria uma figura e eixos com tamanho customizado para a matriz de confusão.
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))

    # 2. Cria o objeto de exibição da matriz com os nomes das classes.
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nomes_das_classes)

    # 3. Plota a matriz nos eixos que criamos (ax=ax_cm).
    disp.plot(cmap='Blues', xticks_rotation=40, ax=ax_cm)

    # 3.1 
    plt.setp(ax_cm.get_xticklabels(), ha="right", rotation_mode="anchor")

    # 4. Define o título nos eixos específicos da matriz
    ax_cm.set_title("Matriz de Confusão")

    # 5. Ajusta o layout para garantir que tudo (incluindo os rótulos) caiba na figura
    fig_cm.tight_layout()

    print("Matriz de confusão exibida com sucesso.")

    # --- Etapa 2: Janela de Predição Interativa com 9 Imagens ---

    # Encontra as classes únicas no conjunto de treino.
    classes_to_show = np.unique(y_train)
        
    # Cria uma nova figura para exibir as 9 imagens em uma grade 3x3.
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle('Predição de 9 Imagens (Uma de Cada Categoria)', fontsize=16)

    # Achata o array de eixos para facilitar a iteração.
    axes = axes.flatten()

    def atualizar_grid(event):
        """
        Função chamada pelo botão. Limpa o grid e preenche com 9 novas
        imagens aleatórias e suas predições.
        """
        for i, class_label in enumerate(classes_to_show):
            ax = axes[i]
            ax.clear() # Limpa o subplot antes de desenhar a nova imagem

            # Encontra TODOS os índices de imagens que correspondem à classe atual.
            indices_da_classe = np.where(y_train == class_label)[0]
            # Escolhe um índice ALEATÓRIO dentro dessa lista.
            idx = random.choice(indices_da_classe)
            
            image = x_train[idx]
            real_class = y_train[idx]

            # Prepara a imagem para o modelo (normaliza e adiciona dimensão de batch).
            image_normalized = image / 255.0
            image_for_prediction = np.expand_dims(image_normalized, axis=0)

            # Faz a predição.
            prediction = model.predict(image_for_prediction)
            predicted_class = np.argmax(prediction)

            # --- ALTERAÇÃO INICIA AQUI ---
            # Verifica o número de canais da configuração para plotar corretamente.
            if config['dados']['num_canais'] == 1:
                # Se for 1 canal, exibe em escala de cinza.
                # np.squeeze remove a dimensão do canal (ex: 224,224,1 -> 224,224)
                ax.imshow(np.squeeze(image), cmap='gray')
            else:
                # Se forem 3 canais, exibe como a imagem colorida padrão.
                ax.imshow(image)
            # --- ALTERAÇÃO TERMINA AQUI ---
            
            ax.set_title(f"Real: {label_para_classe[real_class]}\nPredita: {label_para_classe[predicted_class]}")
            ax.axis('off') # Remove os eixos para uma visualização limpa.
        
        # Redesenha a figura para que as alterações apareçam.
        fig.canvas.draw_idle()

    # Ajusta o layout para criar espaço para o botão e entre as imagens.
    fig.subplots_adjust(bottom=0.2, hspace=0.4)

    # Define a área onde o botão será criado. Formato: [esquerda, baixo, largura, altura].
    ax_botao = fig.add_axes([0.4, 0.05, 0.2, 0.075])

    # Cria o widget do botão e atribui um nome a ele.
    botao_proximo = Button(ax_botao, 'Próximas Imagens')

    # Associa a função 'atualizar_grid' ao evento de clique do botão.
    botao_proximo.on_clicked(atualizar_grid)
    
    # Chama a função uma vez no início para já exibir o primeiro grid de imagens.
    atualizar_grid(None)
    
    # Exibe todas as janelas geradas (matriz e a grade de imagens).
    plt.show()

def main():
    """
    Ponto de entrada principal do script.
    """
    carregar_modelo()

# Garante que o script só será executado quando chamado diretamente.
if __name__ == "__main__":
    main()