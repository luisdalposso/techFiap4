###################
### IMPORTAÇÕES ###
###################

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

################################
### COLETA DOS DADOS DA AÇÃO ###
################################

simbolo = 'NFLX'  # Nome da ação
data_inicio = '2018-01-01'
data_fim = '2024-07-20'
dados_brutos = yf.download(simbolo, start=data_inicio, end=data_fim)
dados_fechamento = dados_brutos[['Close']]

# Normaliza os valores de fechamento entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = normalizador.fit_transform(dados_fechamento)

##########################################
### PREPARAÇÃO DOS DADOS PARA O TREINO ###
##########################################

entradas, saidas = [], []
passos_tempo = 60

for i in range(passos_tempo, len(dados_normalizados)):
    entradas.append(dados_normalizados[i - passos_tempo:i, 0])
    saidas.append(dados_normalizados[i, 0])

entradas, saidas = np.array(entradas), np.array(saidas)
entradas = np.reshape(entradas, (entradas.shape[0], entradas.shape[1], 1))

# Divide os dados em treino e teste (80/20)
limite_treino = int(0.8 * len(entradas))
x_treino, x_teste = entradas[:limite_treino], entradas[limite_treino:]
y_treino, y_teste = saidas[:limite_treino], saidas[limite_treino:]

############################
### CONSTRUÇÃO DO MODELO ###
############################

modelo = Sequential()
modelo.add(LSTM(50, return_sequences=True, input_shape=(x_treino.shape[1], 1)))
modelo.add(LSTM(50))
modelo.add(Dense(1))

modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(x_treino, y_treino, epochs=10, batch_size=32)

###############################
### AVALIAÇÃO DO DESEMPENHO ###
###############################

previsoes = modelo.predict(x_teste)
previsoes = normalizador.inverse_transform(previsoes)
y_teste_real = normalizador.inverse_transform(y_teste.reshape(-1, 1))

mae = mean_absolute_error(y_teste_real, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste_real, previsoes))
mape = np.mean(np.abs((y_teste_real - previsoes) / y_teste_real)) * 100

print(f"MAE (Erro Absoluto Médio): {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}")
print(f"MAPE (Erro Percentual Absoluto Médio): {mape:.2f}%")

# Salva o modelo treinado
modelo.save('modelo_lstm.h5')

##############################################
### API PARA FAZER PREVISÕES EM TEMPO REAL ###
##############################################

# Define o formato esperado para entrada da API
class EntradaPrecos(BaseModel):
    prices: list[float]

# Inicializa a API
app = FastAPI()

@app.post("/predict/")
def prever_proximo_valor(dados: EntradaPrecos):
    precos = np.array(dados.prices).reshape(-1, 1)
    precos_normalizados = normalizador.transform(precos)

    x_novo = []
    x_novo.append(precos_normalizados[-passos_tempo:, 0])
    x_novo = np.array(x_novo)
    x_novo = np.reshape(x_novo, (x_novo.shape[0], x_novo.shape[1], 1))

    modelo_carregado = load_model('modelo_lstm.h5')
    previsao_normalizada = modelo_carregado.predict(x_novo)
    previsao_final = normalizador.inverse_transform(previsao_normalizada)

    return {"proximo_preco_fechamento": float(previsao_final[0, 0])}

# Inicia o servidor local
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)