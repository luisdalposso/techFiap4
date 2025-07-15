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

A API lê o modelo treinado e oferece um endpoint para previsão

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
curl -X POST https://modelo-lstm-api-304859573791.us-central1.run.app/predict/   -H "Content-Type: application/json"   -d '{"prices": [190.1, 191.3, 192.5, 193.2, 194.8, 195.1, 196.4, 197.8, 198.9, 199.3, 200.1, 201.0, 202.4, 203.5, 204.6, 205.1, 206.7, 207.5, 208.1, 209.0, 210.2, 211.5, 212.0, 213.6, 214.7, 215.1, 216.2, 217.0, 218.4, 219.1, 220.3, 221.0, 222.5, 223.1, 224.2, 225.0, 226.3, 227.0, 228.6, 229.4, 230.1, 231.0, 232.2, 233.1, 234.0, 235.5, 236.3, 237.0, 238.4, 239.5, 240.3, 241.0, 242.6, 243.4, 244.1, 245.2, 246.0, 247.1, 248.5, 249.0]}'
```
---

## 🔒 .gcloudignore
Evitar subir arquivos desnecessários, mantendo os arquivos de treinamento necessário para o projeto na Cloud
```
.git
.gitignore
__pycache__/
*.pyc
.venv/
.env
*.secret
```

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