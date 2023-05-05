# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:17:25 2023

@author: Daniel Neres
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/Daniel Neres/Documents/unicsul/IA/dados.csv', sep = ';')

dic_internacao_obito = {'INTERNADO':1,
                        'NAO INTERNADO':0,
                        'INTERNADO UTI':1,
                        'NAO INTERNADO UTI':0,
                        'SIM':1,
                        'NAO':0,
                        }

df['internacao'] = df['internacao'].map(dic_internacao_obito)
df['internacao_uti'] = df['internacao_uti'].map(dic_internacao_obito)
df['obito'] = df['obito'].map(dic_internacao_obito)

df['CANSACO'] = df['sintomas'].apply(lambda x: 1 if 'CANSACO' in str(x) else 0)
df['CEFALEIA'] = df['sintomas'].apply(lambda x: 1 if 'CEFALEIA' in str(x) else 0)
df['CONGESTAO_NASAL'] = df['sintomas'].apply(lambda x: 1 if 'CONGESTAO NASAL' in str(x) else 0)
df['CORIZA'] = df['sintomas'].apply(lambda x: 1 if 'CORIZA' in str(x) else 0)
df['DIARREIA'] = df['sintomas'].apply(lambda x: 1 if 'DIARREIA' in str(x) else 0)
df['DISPNEIA'] = df['sintomas'].apply(lambda x: 1 if 'DISPNEIA' in str(x) else 0)
df['DOR_NO_CORPO'] = df['sintomas'].apply(lambda x: 1 if 'DOR NO CORPO' in str(x) else 0)
df['DOR_DE_GARGANTA'] = df['sintomas'].apply(lambda x: 1 if 'DOR DE GARGANTA' in str(x) else 0)
df['FEBRE'] = df['sintomas'].apply(lambda x: 1 if 'FEBRE' in str(x) else 0)
df['TOSSE'] = df['sintomas'].apply(lambda x: 1 if 'TOSSE' in str(x) else 0)
df['MIALGIA'] = df['sintomas'].apply(lambda x: 1 if 'MIALGIA' in str(x) else 0)

del df['sintomas']

df['ASMA'] = df['comorbidades'].apply(lambda x: 1 if 'ASMA,' in str(x) else 0)
df['CANCER'] = df['comorbidades'].apply(lambda x: 1 if 'CANCER' in str(x) else 0)
df['DIABETES'] = df['comorbidades'].apply(lambda x: 1 if 'DIABETES' in str(x) else 0)
df['DOENCA_HEMATOLOGICA_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA HEMATOLOGICA CRONICA' in str(x) else 0)
df['DOENCA_CARDIOVASCULAR_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA CARDIOVASCULAR CRONICA' in str(x) else 0)
df['DOENCA_RENAL_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA RENAL CRONICA' in str(x) else 0)
df['DOENCA_NEUROLOGICA_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA NEUROLOGICA CRONICA' in str(x) else 0)
df['DOENCA_HEPATICA_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA HEPATICA CRONICA' in str(x) else 0)
df['DOENCA_PNEUMATICA_CRONICA'] = df['comorbidades'].apply(lambda x: 1 if 'DOENCA PNEUMATICA CRONICA' in str(x) else 0)
df['OBESIDADE'] = df['comorbidades'].apply(lambda x: 1 if 'OBESIDADE' in str(x) else 0)
df['HIPERTENSAO'] = df['comorbidades'].apply(lambda x: 1 if 'HIPERTENSAO' in str(x) else 0)
df['IMUNODEPRESSAO'] = df['comorbidades'].apply(lambda x: 1 if 'IMUNODEPRESSAO' in str(x) else 0)
df['SINDROME_DE_DOWN'] = df['comorbidades'].apply(lambda x: 1 if 'SINDROME DE DOWN' in str(x) else 0)

del df['comorbidades']

X = df.drop(['obito','sexo'], axis=1).fillna(0)

y = df['obito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calcula o R² e o MSE usando o conjunto de teste
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('R²:', r2)
print('MSE:', mse)


import matplotlib.pyplot as plt

# plotar gráfico de dispersão dos valores reais versus as previsões do modelo
plt.scatter(y_test, y_pred)

# adicionar linha diagonal para comparar com a previsão perfeita
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)

# definir rótulos dos eixos
plt.xlabel('Valores reais')
plt.ylabel('Previsões do modelo')

# mostrar o gráfico
plt.show()

















