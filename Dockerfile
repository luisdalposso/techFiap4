FROM python:3.10-slim

# Evita prompts durante instalação
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cria pasta de trabalho
WORKDIR /app

# Copia apenas o necessário
COPY requirements.txt .

# Instala dependências
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia os demais arquivos
COPY . .

# Expõe a porta correta esperada pelo Cloud Run
EXPOSE 8080

# Comando para iniciar a API (usa a variável de ambiente PORT como fallback)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]