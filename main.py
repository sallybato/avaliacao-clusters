
'''libs para o inicio da implementação'''
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans

dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', sep= ',')


dados_num = dados.drop(columns=['Gender','family_history_with_overweight',
                             'FAVC','CAEC','SMOKE','SCC', 'CALC','MTRANS', 'NObeyesdad'])

dados_cat = dados[['Gender','family_history_with_overweight',
                               'FAVC','CAEC','SMOKE','SCC', 'CALC','MTRANS', 'NObeyesdad']]

#separação dos textos/numeros do arquivo csv

scaler = MinMaxScaler() #prepara a ferramenta 

normalizador = scaler.fit(dados_num) #leitura dos numeros para descobrir quem é o maior ou menor

dados_num_norm = normalizador.fit_transform(dados_num) #troca os numeros originais pelos espremidos

dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)
dados_cat_norm = pd.get_dummies(dados_cat,prefix_sep='_', dtype= int)
dados_norm = dados_num_norm.join(dados_cat_norm, how='left')

#iniciando o agrupamento de KMeans para hiperparametrizar 

modelo_kmeans = KMeans(n_clusters=7, random_state=42)
modelo_kmeans.fit(dados_norm)
dados['cluster'] = modelo_kmeans.labels_

print(dados['cluster'].value_counts())

nomes_colunas = list(dados_num.columns) + list(dados_cat_norm.columns)
centroides = pd.DataFrame(modelo_kmeans.cluster_centers_, columns= nomes_colunas)

#separa os numeros e categorias
cent_num = centroides[dados_num.columns]
cent_cat = centroides[dados_cat_norm.columns]

cent_num_real = pd.DataFrame(normalizador.inverse_transform(cent_num), columns=dados_num.columns)
#desnormalizando os numeros (voltando para os valores originais)
# Para cada categoria, pega o nome da coluna com maior valor
cent_cat_real = pd.DataFrame(index=range(len(cent_cat)))

for coluna in dados_cat.columns:
    # Filtra só as colunas do one-hot que pertencem a essa categoria
    colunas_categoria = [c for c in dados_cat_norm.columns if c.startswith(coluna + '_')]
    # Pega o nome da opção com maior valor e remove o prefixo
    cent_cat_real[coluna] = cent_cat[colunas_categoria].idxmax(axis=1).str.replace(coluna + '_', '', regex=False)

descricao_clusters = cent_num_real.join(cent_cat_real)
descricao_clusters.index.name = 'Cluster'
print(descricao_clusters)

#inicio de inferencia de paciente desconhecido como exemplo

paciente = pd.DataFrame([{
    'Age': 25,
    'Height': 1.70,
    'Weight': 80,
    'FCVC' :2,
    'NCP': 3,
    'CH2O': 2,
    'FAF': 1,
    'TUE': 1,
    'Gender': 'Male',
    'family_history_with_overweight': 'yes',
    'FAVC': 'yes',
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'SCC': 'no',
    'CALC': 'no',
    'MTRANS': 'Public_Transportation',
    'NObeyesdad': 'Normal_Weight'
}])

#separa e normaliza o paciente como no treinamento
pac_num = paciente[dados_num.columns]
pac_cat = paciente[dados_cat.columns]

pac_num_norm = pd.DataFrame(normalizador.transform(pac_num), columns=dados_num.columns)
pac_cat_norm = pd.get_dummies(pac_cat, prefix_sep='_', dtype=int)

#garante que o paciente tem as mesmas colunas do treino
pac_cat_norm = pac_cat_norm.reindex(columns=dados_cat_norm.columns, fill_value=0)

pac_norm = pac_num_norm.join(pac_cat_norm)

#momento de inicialização da inferencia
cluster_paciente = modelo_kmeans.predict(pac_norm)
print(f"\nO paciente pertence ao Cluster: {cluster_paciente[0]}")