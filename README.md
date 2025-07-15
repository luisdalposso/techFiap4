# Previsão de Valores com LSTM

Este projeto realiza a previsão de valores utilizando uma rede neural LSTM (Long Short-Term Memory). Ele abrange o treinamento do modelo, normalização dos dados, avaliação de desempenho e disponibilização da API para inferência.

---

## 📁 Estrutura de Arquivos

- `treinamento.py`: contém o código de pré-processamento, treinamento, avaliação do modelo LSTM e geração dos arquivos `.h5`, `.npy` e `.pkl`.
- `api.py`: API desenvolvida com FastAPI para disponibilizar o modelo treinado em produção.
- `requirements.txt`: bibliotecas necessárias.
- `README.md`: este documento.

---

## 🧠 Treinamento do Modelo

### Objetivo
Treinar uma rede LSTM com base em séries temporais para prever valores futuros. O modelo é avaliado por métricas como MAE, RMSE e MAPE.

### Etapas
1. **Leitura dos dados** do CSV.
2. **Normalização** dos valores com `MinMaxScaler`.
3. **Divisão** dos dados em treino e teste.
4. **Criação da estrutura de entrada** para a LSTM com `look_back=3`.
5. **Definição do modelo LSTM** usando Keras:
   - Camada LSTM com 50 neurônios
   - Camada `Dense` de saída
6. **Treinamento** com 100 épocas.
7. **Avaliação** dos erros (MAE, RMSE e MAPE).
8. **Salvar os arquivos** necessários:
   - `modelo_lstm.h5`: modelo treinado
   - `normalizador.npy`: escalador MinMaxScaler
   - `look_back.pkl`: parâmetro usado

---

## 🚀 API com FastAPI

A API lê o modelo treinado e oferece um endpoint para previsão baseado em três valores anteriores.

### Endpoint
- `POST /prever`
- Entrada (JSON):
```json
{
  "valores": [120.5, 122.0, 121.2]
}
```
- Saída:
```json
{
  "valor_previsto": 123.87
}
```

---

## ☁️ Deploy na Nuvem (Google Cloud Run)

### Ferramentas Utilizadas
- Google Cloud SDK
- Docker
- Google Container Registry (GCR)
- Cloud Run

### Passos Realizados

1. **Criação do projeto** e ativação do serviço `Cloud Build`:
```bash
gcloud builds submit --tag gcr.io/api-previsao-acoes/modelo-lstm-api
```

2. **Criação do Dockerfile**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
```

3. **Deploy no Cloud Run**:
```bash
gcloud run deploy modelo-lstm-api   --image gcr.io/api-previsao-acoes/modelo-lstm-api   --platform managed   --region us-central1   --memory=1Gi   --timeout=600s   --allow-unauthenticated
```

4. **Resultado**:
- URL pública: https://modelo-lstm-api-304859573791.us-central1.run.app

---

## 📈 Monitoramento

Configurado automaticamente via Google Cloud Console:
- **Cloud Logging**: rastreia erros e tempo de resposta.
- **Cloud Monitoring**: análise de CPU, memória e latência.

---

## ✅ Teste via CURL

```bash
curl -X POST https://modelo-lstm-api-304859573791.us-central1.run.app/prever -H "Content-Type: application/json" -d '{"valores": [120.5, 122.0, 121.2]}'
```

---

## 🔒 .gitignore

Evita subir arquivos desnecessários:
```
.venv/
__pycache__/
*.pyc
.env
*.h5
*.npy
*.pkl
```

---

## 📋 Observações

- A LSTM foi treinada apenas uma vez para não consumir recursos excessivos.
- O deploy usa 1Gi de memória e tempo de inicialização de 600s por conta do tamanho dos modelos.