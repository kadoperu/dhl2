# Dockerfile — Proxy FastAPI para Qwen2.5-VL-32B-Instruct-AWQ (DGX Spark)
#
# Este Dockerfile construye SOLO el proxy liviano que escucha en el puerto 8499
# y reenvía las peticiones al backend vLLM (ver docker-compose.yml).
# python:3.12-slim tiene builds oficiales para arm64, así que funciona nativo en el Spark.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vl_server.py .

EXPOSE 8499

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8499/health || exit 1

CMD ["uvicorn", "vl_server:app", "--host", "0.0.0.0", "--port", "8499"]