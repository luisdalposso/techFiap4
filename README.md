###############################
#### TECH CHALLENGE 4 FIAP ####
###############################


# 📈 Previsão de Preço de Ações com LSTM e FastAPI - 

Este projeto usa redes neurais LSTM para prever o próximo valor de fechamento de uma ação com base nos últimos 60 valores.

---

## 🚀 Tecnologias Utilizadas

- [Python](https://www.python.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Pandas & NumPy](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [yFinance](https://pypi.org/project/yfinance/)
- [Uvicorn](https://www.uvicorn.org/)

---

## ⚙️ Etapas do Código

### 1. **Coleta dos Dados**
O código utiliza o pacote 'yfinance' para baixar dados históricos da ação da **Netflix (NFLX)** desde 2018 até uma data de corte. 
Utiliza-se apenas a coluna de **preço de fechamento**.

'''python
dados_brutos = yf.download(simbolo, start=data_inicio, end=data_fim)
dados_fechamento = dados_brutos[['Close']]
'''

---

### 2. **Normalização**
Os dados são normalizados com 'MinMaxScaler', convertendo os valores para o intervalo entre 0 e 1. Isso facilita o aprendizado do modelo.

'''python
normalizador = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = normalizador.fit_transform(dados_fechamento)
'''

---

### 3. **Preparação dos Dados**
Cada amostra usada no treinamento tem 60 dias consecutivos (janelas deslizantes). O objetivo é prever o valor do 61º dia com base nos 60 anteriores.

'''python
for i in range(passos_tempo, len(dados_normalizados)):
    entradas.append(dados_normalizados[i - passos_tempo:i, 0])
    saidas.append(dados_normalizados[i, 0])
'''

---

### 4. **Criação e Treinamento do Modelo**
O modelo é composto por duas camadas 'LSTM' e uma camada final 'Dense'. A rede é treinada para minimizar o erro quadrático médio (MSE).

'''python
modelo = Sequential()
modelo.add(LSTM(50, return_sequences=True, input_shape=(x_treino.shape[1], 1)))
modelo.add(LSTM(50))
modelo.add(Dense(1))
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(x_treino, y_treino, epochs=10, batch_size=32)
'''

---

### 5. **Avaliação do Modelo**
Depois do treino, o modelo é avaliado com métricas como MAE, RMSE e MAPE. Isso ajuda a entender o quão preciso ele está na tarefa de previsão.

'''python
mae = mean_absolute_error(y_teste_real, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste_real, previsoes))
mape = np.mean(np.abs((y_teste_real - previsoes) / y_teste_real)) * 100
'''

---

### 6. **API de Previsão**
A API FastAPI permite enviar uma lista de 60 preços de fechamento e retornar o próximo valor previsto pelo modelo treinado.

Exemplo de payload:
'''json
{
  "prices": [101.0, 102.5, 99.8, ..., 105.6]
}
'''

A rota disponível:
'''
POST /predict/
'''

---

### 7. **Execução do Servidor**
Para rodar a API localmente:
'''bash
uvicorn script:app --reload
'''

---

## 📝 Como Executar Localmente

1. Clone este repositório:
   '''bash
   git clone git@github.com:luisdalposso/techFiap4.git
   cd techFiap4
   '''

2. Crie um ambiente virtual:
   '''bash
   python3 -m venv .venv
   source .venv/bin/activate
   '''

3. Instale as dependências:
   '''bash
   pip install -r requirements.txt
   '''

4. Execute o script para treinar o modelo:
   '''bash
   python script.py
   '''

5. Inicie a API:
   '''bash
   uvicorn script:app --reload
   '''

6. Teste a API no navegador em: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🤖 Objetivo

Este projeto foi criado com fins educacionais e pode ser expandido para qualquer outro ativo financeiro disponível no Yahoo Finance. A LSTM foi escolhida por ser ideal para **séries temporais** e capturar padrões de longo prazo entre dados históricos.

---

## 📂 Observação

Certifique-se de manter a pasta '.venv/' e arquivos '.h5' fora do repositório. Eles estão devidamente ignorados no '.gitignore'.

---

## 📬 Contato

**Luis Dalposso**  
[GitHub](https://github.com/luisdalposso) | [LinkedIn](https://www.linkedin.com/in/luisdalposso)