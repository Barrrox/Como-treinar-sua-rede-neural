"""
Esse código varre o banco de dados inteiro, executa uma operação de crop central e
redimensiona a imagem para 128x128. No fim, ele agrupa todas as imagens em
um único arquivo do numpy (.npy).

Versão Otimizada: Utiliza funções vetorizadas do OpenCV e NumPy para um
processamento muito mais rápido, mantendo a saída idêntica à original.

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

import cv2
import numpy as np
import random
from pathlib import Path
from time import time
import os  # Importado para construir os caminhos de arquivo de forma segura
# tqdm é uma biblioteca excelente para criar barras de progresso.
# Instale com: pip install tqdm
from tqdm import tqdm

# Carrega as configs/parametros 
# (Assumindo que utilidade.py está no diretório correto conforme conversamos)
from utilidade import carregar_config
config = carregar_config()


#
def formatImageRGB(imagemDesf: np.ndarray) -> np.ndarray:
    """
    Realiza um corte quadrado (1:1) aleatório, redimensiona e converte de BGR para RGB
    usando operações vetorizadas de alta performance do OpenCV.
    """
    height, width, _ = imagemDesf.shape
    startY, startX = 0, 0

    outRes = config['dados']['tamanho_imagem']
    
    # 1. Lógica de corte simplificada
    if height == width:
        outScale = height
    elif height > width:
        outScale = width
        startY = int((height - width) * random.random())
    else: # width > height
        outScale = height
        startX = int((width - height) * random.random())

    # 2. Corte (crop) vetorizado usando slicing do NumPy (extremamente rápido)
    imagem_cortada = imagemDesf[startY:startY + outScale, startX:startX + outScale]

    # 3. Redimensionamento (resize) usando a função otimizada do OpenCV
    imagem_redimensionada = cv2.resize(imagem_cortada, (outRes, outRes))
    
    # 4. Inversão de canais BGR para RGB (OpenCV é BGR por padrão) de forma vetorizada
    #    A operação de slicing do NumPy [..., ::-1] é uma alternativa muito rápida também.
    output = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB)
    
    return output


def formatImageBnW(imagemDesf: np.ndarray) -> np.ndarray:
    """
    Realiza crop central, redimensiona e converte a imagem para escala de cinza.
    Esta versão usa funções otimizadas do OpenCV e NumPy.
    """
    # --- 1. Crop Central (Lógica idêntica, implementação vetorizada) ---
    height, width = imagemDesf.shape[:2]
    lado_menor = min(height, width)
    
    # Calcula os pontos de início para o corte central
    inicio_y = (height - lado_menor) // 2
    inicio_x = (width - lado_menor) // 2
    
    # Realiza o corte usando slicing do NumPy, que é instantâneo
    imagem_cortada = imagemDesf[inicio_y : inicio_y + lado_menor, inicio_x : inicio_x + lado_menor]

    # --- 2. Redimensionamento (Lógica idêntica, implementação otimizada) ---
    outRes = config['dados']['tamanho_imagem']
    # O loop manual original é um redimensionamento por "vizinho mais próximo".
    # A interpolação cv2.INTER_NEAREST replica exatamente essa lógica.
    imagem_redimensionada = cv2.resize(imagem_cortada, (outRes, outRes), interpolation=cv2.INTER_NEAREST)

    # --- 3. Conversão para Escala de Cinza (Lógica idêntica, implementação vetorizada) ---
    # Acessa os canais BGR da imagem. OpenCV lê em BGR.
    # imagem_redimensionada[:, :, 0] -> Canal Azul (B)
    # imagem_redimensionada[:, :, 1] -> Canal Verde (G)
    # imagem_redimensionada[:, :, 2] -> Canal Vermelho (R)
    
    # A fórmula original era: 0.299*R + 0.587*G + 0.114*B.
    # Aplicamos a mesma fórmula em todo o array de uma só vez.
    outimage_2d = (0.299 * imagem_redimensionada[:, :, 2] +
                   0.587 * imagem_redimensionada[:, :, 1] +
                   0.114 * imagem_redimensionada[:, :, 0])

    # Converte o resultado para o tipo de dado original (inteiro de 8 bits)
    outimage_2d = outimage_2d.astype(np.uint8)

    # A saída original tinha o shape (224, 224, 1). Adicionamos essa dimensão extra.
    outimage = np.expand_dims(outimage_2d, axis=-1)

    return outimage


def getFolder(path: str) -> str:
    return Path(path).parent.name


def formatDataSet(caminhoParaODataSet):
    """
    Formata o dataset inteiro de imagens, salva e retorna em forma de uma lista de matrizes.
    """
    classe_para_label = {
        "Baroque": 0, "Cubism": 1, "Expressionism": 2, "Impressionism": 3,
        "Minimalism": 4, "Post_Impressionism": 5, "Realism": 6,
        "Romanticism": 7, "Symbolism": 8
    }

    diretorio_raiz = Path(caminhoParaODataSet)
    # Usamos .rglob para pegar imagens em subpastas, filtrando por extensões comuns
    extensions = ['*.jpg', '*.jpeg', '*.png']
    lista_de_caminhos_imagens = [p for ext in extensions for p in diretorio_raiz.rglob(ext)]

    lista_de_matrizes = []
    lista_de_classes = []
    print(f"Encontradas {len(lista_de_caminhos_imagens)} imagens.")

    # A barra de progresso do tqdm torna a espera mais informativa
    for caminho_imagem in tqdm(lista_de_caminhos_imagens, desc="Processando Imagens"):
        try:
            # Abre a imagem
            imagem = cv2.imread(str(caminho_imagem))

            # Verifica se a imagem foi carregada corretamente
            if imagem is None:
                print(f"\nAviso: Não foi possível ler o arquivo {caminho_imagem}, pulando.")
                continue

            # Chama a função de formatação dependendo do número de canais
            # 3 = RGB, 1 = escala de cinza
            
            num_canais = config["dados"]["num_canais"]

            if num_canais == 1:
                imagemFormatada = formatImageBnW(imagem)
            elif num_canais == 3:
                imagemFormatada = formatImageRGB(imagem)
            
            
            lista_de_matrizes.append(imagemFormatada)
            
            classe = getFolder(str(caminho_imagem))
            lista_de_classes.append(classe_para_label[classe])

        except Exception as e:
            print(f"\nErro ao processar o arquivo {caminho_imagem}: {e}")
    

    caminho_saida_imagens = config['dados']['arquivo_imagens_treino']
    caminho_saida_labels = config['dados']['arquivo_labels_treino']
    
    print("Salvando arquivos .npy...")
    np.save(caminho_saida_imagens, lista_de_matrizes)
    np.save(caminho_saida_labels, lista_de_classes)
    print("Arquivos salvos com sucesso.")

    return lista_de_matrizes, lista_de_classes

# --- Execução do Script ---
if __name__ == "__main__":
    inicio = time()
    
    
    formatDataSet(config["dados"]["caminho_base_de_dados"])
    
    print(f"Tempo de execução: {time() - inicio:.2f} segundos")