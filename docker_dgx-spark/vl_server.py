"""
vl_server.py
Proxy FastAPI para Qwen2.5-VL-32B-Instruct-AWQ servido por vLLM en NVIDIA DGX Spark.

Endpoints:
  GET  /health                 -> healthcheck del proxy + verifica el backend vLLM
  GET  /v1/models               -> passthrough a vLLM
  POST /v1/chat/completions     -> passthrough OpenAI-compatible (soporta streaming)
  POST /v1/vision/analyze       -> endpoint simplificado: imagen (URL o archivo) + prompt

Variables de entorno:
  VLLM_BASE_URL    URL interna del servidor vLLM (default: http://vllm:8000)
  MODEL_NAME       Nombre del modelo tal como lo espera vLLM
  PROXY_PORT       Puerto de escucha del proxy (default: 8499)
  REQUEST_TIMEOUT  Timeout en segundos para requests al backend (default: 300)
"""

import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=REQUEST_TIMEOUT)
    yield
    await client.aclose()


app = FastAPI(
    title="Qwen2.5-VL Proxy (DGX Spark)",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Modelos de datos ----------

class ChatMessage(BaseModel):
    role: str
    content: Any  # str o lista de bloques (texto/imagen), formato OpenAI-compatible


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: list[ChatMessage]
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 1024
    stream: bool = False


# ---------- Utilidades internas ----------

def _build_vision_messages(image_content: str, prompt: str, is_url: bool) -> list[dict]:
    image_block = (
        {"type": "image_url", "image_url": {"url": image_content}}
        if is_url
        else {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}}
    )
    return [
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def _forward_chat_completion(payload: dict) -> httpx.Response:
    assert client is not None
    return await client.post("/v1/chat/completions", json=payload)


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    assert client is not None
    try:
        r = await client.get("/health", timeout=5.0)
        backend_ok = r.status_code == 200
    except httpx.HTTPError:
        backend_ok = False

    status = "ok" if backend_ok else "backend_unreachable"
    code = 200 if backend_ok else 503
    return JSONResponse(
        status_code=code,
        content={"proxy": "ok", "vllm_backend": status, "model": MODEL_NAME, "ts": time.time()},
    )


@app.get("/v1/models")
async def list_models():
    assert client is not None
    r = await client.get("/v1/models")
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    payload = req.model_dump()
    payload.setdefault("model", MODEL_NAME)

    if req.stream:

        async def event_stream():
            assert client is not None
            async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=body.decode())
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    r = await _forward_chat_completion(payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(status_code=200, content=r.json())


@app.post("/v1/vision/analyze")
async def vision_analyze(
    prompt: str = Form(default="Describe esta imagen en detalle."),
    max_tokens: int = Form(default=512),
    temperature: float = Form(default=0.2),
    image_url: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """Endpoint simplificado: manda una imagen (URL o archivo subido) + un prompt de texto."""
    if not image_url and not file:
        raise HTTPException(status_code=400, detail="Debes enviar 'image_url' o subir 'file'.")

    if file is not None:
        raw = await file.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        messages = _build_vision_messages(b64, prompt, is_url=False)
    else:
        messages = _build_vision_messages(image_url, prompt, is_url=True)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = await _forward_chat_completion(payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        text = None

    return {"response": text, "raw": data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vl_server:app", host="0.0.0.0", port=int(os.getenv("PROXY_PORT", "8499")))