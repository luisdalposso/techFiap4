
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Coleta os dados
simbolo = 'NFLX'
data_inicio = '2018-01-01'
data_fim = '2025-07-01'
dados_brutos = yf.download(simbolo, start=data_inicio, end=data_fim)
dados_fechamento = dados_brutos[['Close']]

# Normaliza os valores de fechamento entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = normalizador.fit_transform(dados_fechamento)

# Prepara os dados com janela de 60 dias
entradas, saidas = [], []
passos_tempo = 60
for i in range(passos_tempo, len(dados_normalizados)):
    entradas.append(dados_normalizados[i - passos_tempo:i, 0])
    saidas.append(dados_normalizados[i, 0])

entradas, saidas = np.array(entradas), np.array(saidas)
entradas = np.reshape(entradas, (entradas.shape[0], entradas.shape[1], 1))

# Divide treino e teste
limite_treino = int(0.8 * len(entradas))
x_treino, x_teste = entradas[:limite_treino], entradas[limite_treino:]
y_treino, y_teste = saidas[:limite_treino], saidas[limite_treino:]

# Cria e treina modelo LSTM
modelo = Sequential()
modelo.add(LSTM(50, return_sequences=True, input_shape=(x_treino.shape[1], 1)))
modelo.add(LSTM(50))
modelo.add(Dense(1))
modelo.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=3)
modelo.fit(x_treino, y_treino, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Avaliação
previsoes = modelo.predict(x_teste)
previsoes = normalizador.inverse_transform(previsoes)
y_teste_real = normalizador.inverse_transform(y_teste.reshape(-1, 1))

mae = mean_absolute_error(y_teste_real, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste_real, previsoes))
mape = np.mean(np.abs((y_teste_real - previsoes) / y_teste_real)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Salva modelo e normalizador
modelo.save('modelo_lstm.h5')
np.save('normalizador.npy', normalizador.fit(dados_fechamento))

joblib.dump(normalizador, "normalizador.gz")