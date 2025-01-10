import os
import time
import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor

import sys
print("Versão do Python em uso:", sys.version)

# Diretório onde os notebooks estão armazenados
notebook_dir = 'C:/Users/Renan Torres/OneDrive/Documentos/Github/Resultados_PSO/Notebooks'  # caminho absoluto real

# Lista de notebooks a serem executados, em ordem

notebooks = [
    'Ensemble_1.ipynb', 'Ensemble_2.ipynb','Ensemble_3.ipynb', 
    'Ensemble_4.ipynb', 'Ensemble_5.ipynb', 'Ensemble_6.ipynb', 
    'Ensemble_7.ipynb', 'Ensemble_8.ipynb', 'Ensemble_9.ipynb'
]

for nb_name in notebooks:
    nb_path = os.path.join(notebook_dir, nb_name)
    print(f"Executando {nb_name}...")

    try:
        # Carrega o notebook
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Configura o preprocessador
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        
        # Salva o notebook após execução
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"Concluído {nb_name}.")
    
    except Exception as e:
        print(f"Erro ao executar {nb_name}: {e}")

    # Pausa entre notebooks para evitar problemas de memória e garantir separação dos kernels
    time.sleep(2)

