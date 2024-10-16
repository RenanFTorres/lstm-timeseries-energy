import pandas as pd
import json
import os

here = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(here, 'config.json')
filename_duas_unas = os.path.join(here, './dataset/EEAB_DUAS_UNAS.csv')
filename_pasta_duas_unas = os.path.join(here, './dataset')

# Carregar o JSON de configuração
with open(filename, 'r') as f:
    config = json.load(f)

dataset_config = config['dataset']

# Carregar o dataset 'duas_unas' e combinar 'Data' e 'Hora' para formar um timestamp
df = pd.read_csv(filename_duas_unas)

# Criar coluna de timestamp a partir de 'Data' e 'Hora'
df['timestamp'] = pd.to_datetime(df['Data'], format='%d/%m/%Y') + pd.to_timedelta(df['Hora'] - 1, unit='h')

# Definir o índice como a coluna 'timestamp'
df.set_index('timestamp', inplace=True)

# Remover as colunas originais de 'Data' e 'Hora' já que temos o 'timestamp'
df.drop(columns=['Data', 'Hora'], inplace=True)
        
# Função para agrupar por horas
def group_by_hours(df, hours_list):
    for hours in hours_list:
        # Resample para o somatório a cada 'hours' horas
        grouped_df = df.resample(f'{hours}H').sum()
        
        # Usar o diretório 'dataset' existente e criar a pasta 'duas_unas' dentro dele
        output_dir = os.path.join(filename_pasta_duas_unas, 'duas_unas/hours')
        os.makedirs(output_dir, exist_ok=True) # Cria 'duas_unas' dentro de 'dataset', se não existir
        
        # Salvar arquivo CSV
        output_file = os.path.join(output_dir, f'grouped_{hours}_hours.csv')
        grouped_df.to_csv(output_file)
        print(f'Salvo: {output_file}')

# Função para agrupar por dias
def group_by_days(df, days_list):
    for days in days_list:
        # Resample para o somatório a cada 'days' dias
        grouped_df = df.resample(f'{days}D').sum()
        
        # Usar o diretório 'dataset' existente e criar a pasta 'duas_unas' dentro dele
        output_dir = os.path.join(filename_pasta_duas_unas, 'duas_unas/days')
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar arquivo CSV
        output_file = os.path.join(output_dir, f'grouped_{days}_days.csv')
        grouped_df.to_csv(output_file)
        print(f'Salvo: {output_file}')

# Listas de horas e dias para agrupar
hours_list = dataset_config['group_by']['hours']
days_list = dataset_config['group_by']['days']

# Executar os agrupamentos
group_by_hours(df, hours_list)
group_by_days(df, days_list)

