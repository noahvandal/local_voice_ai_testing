"""
Server module providing API endpoints for LLM, Kokoro TTS, and Parakeet ASR.
It also mounts a simple Gradio web UI to interact with the same capabilities.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import io
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from gradio.routes import mount_gradio_app

# --- Internal imports -------------------------------------------------------
from .llm.llm_host import HuggingFaceLLMHost
from .tts.kokoro_text_to_speech import KokoroTextToSpeech
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

app = FastAPI(title="VoiceLatency API", description="Endpoints for LLM, TTS, and ASR", version="0.1.0")

# ---------------------------------------------------------------------------
# Model initialisation (performed once at startup)
# ---------------------------------------------------------------------------

# 1. LLM host ----------------------------------------------------------------
llm_host = HuggingFaceLLMHost(model_name="Qwen/Qwen3-1.7B", use_unsloth=False)

# 2. Kokoro TTS --------------------------------------------------------------
tts_engine = KokoroTextToSpeech()

# 3. Parakeet ASR ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").to(DEVICE)
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.compute_timestamps = True
    asr_model.change_decoding_strategy(decoding_cfg)
except Exception as e:
    # If the model cannot be loaded at import-time, raise a clear error so that
    # the user is aware immediately rather than on first request.
    raise RuntimeError(f"Parakeet ASR model failed to load: {e}") from e

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class LLMRequest(BaseModel):
    prompt: str
    enable_thinking: bool = True

class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a 1-D or 2-D torch tensor (channels x samples) to WAV bytes."""
    audio_np = audio_tensor.squeeze().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/")
async def read_root():
    """Health-check root endpoint."""
    return {"message": "Hello, World!"}

# --- LLM --------------------------------------------------------------------

@app.post("/llm/query", tags=["LLM"])
async def llm_query(payload: LLMRequest):
    """Query the language model with a text prompt."""
    try:
        response = llm_host.query(payload.prompt, enable_thinking=payload.enable_thinking)
        return {"response": response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# --- TTS --------------------------------------------------------------------

@app.post("/tts/generate", tags=["TTS"], response_class=StreamingResponse)
async def tts_generate(payload: TTSRequest):
    """Generate speech from text. Returns WAV audio."""
    try:
        # Configure voice & speed if provided by user
        if payload.voice:
            tts_engine.voice = payload.voice
        if payload.speed:
            tts_engine.speed = payload.speed

        result = tts_engine.generate_speech(payload.text)
        wav_bytes = _tensor_to_wav_bytes(result["tts_speech"], result["sample_rate"])
        return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# --- ASR --------------------------------------------------------------------

@app.post("/asr/transcribe", tags=["ASR"])
async def asr_transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file using Parakeet ASR."""
    try:
        contents = await file.read()
        data, samplerate = sf.read(io.BytesIO(contents), dtype="float32")
        if data.ndim > 1:  # Convert to mono
            data = data.mean(axis=1)
        audio_tensor = torch.from_numpy(data).to(DEVICE)

        hypotheses = asr_model.transcribe(audio=audio_tensor, return_hypotheses=True)
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        if not hypotheses:
            return {"text": ""}

        hypothesis = hypotheses[0]
        word_timestamps = []
        if hasattr(hypothesis, "timestamp") and isinstance(hypothesis.timestamp, dict):
            word_timestamps = hypothesis.timestamp.get("word", [])

        return {
            "text": hypothesis.text,
            "word_timestamps": [
                {"word": w.word, "start": w.start, "end": w.end} for w in word_timestamps
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# Gradio Web UI (mounted at /gradio)
# ---------------------------------------------------------------------------

# Helper functions for Gradio -------------------------------------------------

def _gradio_llm(prompt: str):
    return llm_host.query(prompt)

def _gradio_tts(text: str):
    result = tts_engine.generate_speech(text)
    return result["sample_rate"], result["tts_speech"].cpu().numpy()

def _gradio_asr(audio):
    """`audio` comes in as (sample_rate, ndarray) from gradio."""
    if audio is None:
        return ""
    sample_rate, data = audio
    if sample_rate != 16000:
        # naive resample via numpy – for demo purposes only.
        factor = 16000 / sample_rate
        data = np.interp(
            np.arange(0, len(data) * factor) / factor,
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    audio_tensor = torch.from_numpy(data.astype("float32")).to(DEVICE)
    hypotheses = asr_model.transcribe(audio=audio_tensor, return_hypotheses=True)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    if not hypotheses:
        return ""
    return hypotheses[0].text

# Build Blocks UI ------------------------------------------------------------

def _build_gradio_ui():
    with gr.Blocks(title="VoiceLatency Demo") as demo:
        gr.Markdown("# VoiceLatency Demo\nInteract with the Language Model, Text-to-Speech, and ASR capabilities.")
        with gr.Tab("LLM"):
            prompt_box = gr.Textbox(lines=5, label="Prompt")
            llm_btn = gr.Button("Generate")
            llm_out = gr.Textbox(label="Response")
            llm_btn.click(_gradio_llm, inputs=prompt_box, outputs=llm_out)
        with gr.Tab("TTS"):
            tts_text = gr.Textbox(lines=5, label="Text")
            tts_btn = gr.Button("Generate Speech")
            tts_audio = gr.Audio(label="Audio Output", type="numpy")
            tts_btn.click(_gradio_tts, inputs=tts_text, outputs=tts_audio)
        with gr.Tab("ASR"):
            asr_audio_in = gr.Audio(source="upload", type="numpy", label="Upload Audio")
            asr_btn = gr.Button("Transcribe")
            asr_out = gr.Textbox(label="Transcription")
            asr_btn.click(_gradio_asr, inputs=asr_audio_in, outputs=asr_out)
    return demo

# Mount the Gradio UI on the FastAPI application -----------------------------

demo_app = _build_gradio_ui()
app = mount_gradio_app(app, demo_app, path="/gradio")
