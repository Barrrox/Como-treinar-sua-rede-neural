"""
Arquivo que contém códigos para necessidades gerais do projeto

"""

import yaml

def carregar_config(caminho_config='config.yaml'):
    """Lê o arquivo de configuração YAML e o retorna como um dicionário."""
    with open(caminho_config, 'r') as f:
        return yaml.safe_load(f)