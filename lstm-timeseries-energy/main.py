import pandas as pd
import json
import os

here = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(here, 'config.json')
filename_duas_unas = os.path.join(here, './dataset/EEAB_DUAS_UNAS_1_24.csv')

# Carregar o JSON de configuração
with open(filename, 'r') as f:
    config = json.load(f)

dataset_config = config['dataset']

# Carregar o dataset 'duas_unas'
df = pd.read_csv(filename_duas_unas, parse_dates=True, index_col='timestamp')  # Ajuste o index_col conforme necessário

# Função para agrupar dados por dias e horas
def group_data(df, days, hours):
    # Resample para dados diários
    daily_resample = df.resample(f'{days}D').mean()  # Pode mudar para sum() ou outra função de agregação
    # Para cada hora, podemos recortar um subset da granularidade original
    hourly_resample = daily_resample.resample(f'{hours}H').mean()
    return hourly_resample

# Iterar sobre as combinações de agrupamento
for day in dataset_config['group_by']['days']:
    for hour in dataset_config['group_by']['hours']:
        # Agrupar os dados
        grouped_df = group_data(df, day, hour)
        
        # Criar diretórios para salvar os arquivos, se necessário
        output_dir =  os.path.join(here,f'./dataset/duas_unas')
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar arquivo agrupado
        output_file = os.path.join(output_dir, f'grouped_days_{day}_hours_{hour}.csv')
        grouped_df.to_csv(output_file)
        
        print(f'Salvo: {output_file}')
