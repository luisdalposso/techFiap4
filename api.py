
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn
import joblib
import os

# Carrega o normalizador salvo
normalizador: MinMaxScaler = joblib.load("normalizador.gz")
modelo = load_model("modelo_lstm.h5")
passos_tempo = 60

class EntradaPrecos(BaseModel):
    prices: list[float]

app = FastAPI()

@app.post("/predict/")
def prever_proximo_valor(dados: EntradaPrecos):
    precos = np.array(dados.prices).reshape(-1, 1)
    precos_normalizados = normalizador.transform(precos)

    x_novo = [precos_normalizados[-passos_tempo:, 0]]
    x_novo = np.reshape(x_novo, (1, passos_tempo, 1))

    previsao_normalizada = modelo.predict(x_novo)
    previsao_final = normalizador.inverse_transform(previsao_normalizada)

    return {"proximo_preco_fechamento": float(previsao_final[0, 0])}

if __name__ == "__main__":
    import os
    porta = int(os.environ.get("PORT", 8080))
    uvicorn.run("api:app", host="0.0.0.0", port=porta)