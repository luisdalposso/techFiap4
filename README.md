
# Previsão de Preço de Ações com LSTM

Este projeto utiliza uma rede neural do tipo LSTM para prever o próximo valor de fechamento de uma ação com base nos 60 últimos valores.

---

## 📁 Estrutura de Arquivos

- `treinamento.py`: script responsável por baixar os dados históricos da ação, preparar os dados, treinar o modelo e salvá-lo em disco.
- `api.py`: disponibiliza uma API com FastAPI que carrega o modelo treinado e realiza previsões a partir de 60 valores recebidos como entrada.
- `valores.py`: gera valores aleatórios simulados para testes da API.
- `lstm_modelo.h5`: arquivo gerado contendo o modelo treinado.
- `requirements.txt`: dependências do projeto.

---

## ✅ Requisitos

- Python 3.10 ou superior
- Ambiente virtual com as dependências instaladas:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Como Treinar o Modelo

```bash
python treinamento.py
```

Isso irá:
1. Baixar dados históricos da ação (atualmente: Netflix - `NFLX`)
2. Normalizar os dados
3. Treinar o modelo com LSTM
4. Avaliar a performance (MAE, RMSE, MAPE)
5. Salvar o modelo em `lstm_modelo.h5`

---

## 🧠 Como Usar a API

Com o modelo treinado salvo em disco, execute:

```bash
python api.py
```

A API será executada localmente em `http://127.0.0.1:8000`.

### Exemplo de entrada (JSON)

```json
{
  "precos": [115.2, 112.4, 117.5, ..., 110.1]
}
```

- A lista deve conter exatamente 60 valores.

---

## 🧪 Testando com cURL

```bash
curl -X POST "http://127.0.0.1:8000/prever/" \
     -H "Content-Type: application/json" \
     -d '{"precos": [115.2, 112.4, 117.5, ..., 110.1]}'
```

---

## 📊 Métricas de Avaliação

Na execução do treinamento, as seguintes métricas são exibidas:

- **MAE (Erro Absoluto Médio)**
- **RMSE (Raiz do Erro Quadrático Médio)**
- **MAPE (Erro Percentual Absoluto Médio)**

---
