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
from pathlib import Path
from time import time
# tqdm é uma biblioteca excelente para criar barras de progresso.
# Instale com: pip install tqdm
from tqdm import tqdm

def formatImage(imagemDesf):
	# definir pontos de incio e fim de um espaço de crop 1:1 centrado
	height, width, channels = imagemDesf.shape
	startY = 0
	startX = 0
	outScale = 0

	if height > width:
		outScale = width
		startX = 0

		cropOffset = int((height - width) * 0.5)
		startY = cropOffset

	if width > height:
		outScale = height
		startY = 0

		cropOffset = int((width - height) * 0.5)
		startX = cropOffset


	# criar imagem de saída
	outRes = 224
	output = np.zeros((outRes, outRes, 3), dtype=np.uint8)
	outimage = np.zeros((outRes, outRes, 1), dtype=np.uint8)


	for y in range(outRes):
		for x in range(outRes):
			output[y, x] = imagemDesf[int((y/outRes)*outScale) + startY, int((x/outRes)*outScale) + startX]		
			outimage[y, x] = 0.299*output[y, x, 2] + 0.587*output[y, x, 1] + 0.114*output[y, x, 0]

	# FOI!
	return outimage


def formatImage1(imagemDesf: np.ndarray) -> np.ndarray:
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
    outRes = 224
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

            # Chama a função otimizada
            imagemFormatada = formatImage(imagem)
            
            lista_de_matrizes.append(imagemFormatada)
            
            classe = getFolder(str(caminho_imagem))
            lista_de_classes.append(classe_para_label[classe])

        except Exception as e:
            print(f"\nErro ao processar o arquivo {caminho_imagem}: {e}")
    
    # Salva os arquivos .npy com os mesmos nomes
    print("Salvando arquivos .npy...")
    np.save('imagens_treino_BnW.npy', lista_de_matrizes)
    np.save('labels_treino_BnW.npy', lista_de_classes)
    print("Arquivos salvos com sucesso.")

    return lista_de_matrizes, lista_de_classes

# --- Execução do Script ---
if __name__ == "__main__":
    inicio = time()
    formatDataSet("./BaseDeDados")
    print(f"Tempo de execução: {time() - inicio:.2f} segundos")