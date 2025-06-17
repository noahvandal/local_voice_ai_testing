"""
Server module providing API endpoints for LLM, Kokoro TTS, and Parakeet ASR.
It also mounts a simple Gradio web UI to interact with the same capabilities.
"""

# NOTE: This file now exposes a factory function `create_app` for Uvicorn's
# `--factory` / `factory=True` mode so the heavyweight models are only loaded
# in the worker process (not in the reloader parent).

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import io
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from gradio.routes import mount_gradio_app
import time
from datetime import datetime
import base64
import os

# ---------------------------------------------------------------------------
# Pydantic request models and shared schemas (defined at module level so they
# can be imported elsewhere without executing heavy model loads).
# ---------------------------------------------------------------------------

class LLMRequest(BaseModel):
    prompt: str
    enable_thinking: bool = True

class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None

# Latency report schema (shared)
class LatencyReport(BaseModel):
    asr_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    total_ms: float

# Pipeline response schema (shared)
class PipelineResponse(BaseModel):
    transcription: str
    llm_response: str
    audio_base64: str
    latencies: LatencyReport

# ---------------------------------------------------------------------------
# Factory to build the FastAPI app (and load heavy models) -------------------
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the FastAPI application with all routes and Gradio UI."""

    # --- Internal imports (heavy) -----------------------------------------
    from .conversation_manager import ConversationManager

    # Instantiate ConversationManager and (optionally) start real-time capture
    cm = ConversationManager()
    if os.getenv("CM_START", "0") == "1":
        try:
            cm.start()
            print("ConversationManager started (background audio capture)")
        except Exception as exc:
            print(f"Failed to start ConversationManager: {exc}")

    llm_host = cm.llm
    tts_engine = cm.tts
    asr_model = cm.asr.model
    ASR_DEVICE = next(asr_model.parameters()).device

    app = FastAPI(
        title="VoiceLatency API",
        description="Endpoints for LLM, TTS, and ASR",
        version="0.1.0",
    )

    # ---------------- Helper functions ------------------------------------

    def _tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
        audio_np = audio_tensor.squeeze().cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def _process_pipeline(audio_np: np.ndarray, samplerate: int):
        lat: dict[str, float] = {}
        t0 = time.perf_counter()

        # ASR
        asr_start = time.perf_counter()
        if samplerate != 16000:
            factor = 16000 / samplerate
            audio_np = np.interp(
                np.arange(0, len(audio_np) * factor) / factor,
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.float32)
            samplerate = 16000
        audio_tensor = torch.from_numpy(audio_np.astype("float32")).to(ASR_DEVICE)
        hypotheses = asr_model.transcribe(audio=audio_tensor, return_hypotheses=True)
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        transcription = hypotheses[0].text if hypotheses else ""
        lat["asr_ms"] = (time.perf_counter() - asr_start) * 1000.0

        # LLM
        llm_start = time.perf_counter()
        llm_response = llm_host.query(transcription)
        lat["llm_ms"] = (time.perf_counter() - llm_start) * 1000.0

        # TTS
        tts_start = time.perf_counter()
        tts_out = tts_engine.generate_speech(llm_response)
        wav_bytes = _tensor_to_wav_bytes(tts_out["tts_speech"], tts_out["sample_rate"])
        lat["tts_ms"] = (time.perf_counter() - tts_start) * 1000.0

        lat["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return transcription, llm_response, wav_bytes, lat

    # --------------------- API routes -------------------------------------

    @app.get("/")
    async def read_root():
        return {"message": "Hello, World!"}

    @app.post("/llm/query", tags=["LLM"])
    async def llm_query(payload: LLMRequest):
        received_at = datetime.utcnow().isoformat() + "Z"
        t0 = time.perf_counter()
        try:
            response = llm_host.query(payload.prompt, enable_thinking=payload.enable_thinking)
            proc_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "received_at": received_at,
                "processing_ms": proc_ms,
                "response": response,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/tts/generate", tags=["TTS"], response_class=StreamingResponse)
    async def tts_generate(payload: TTSRequest):
        try:
            if payload.voice:
                tts_engine.voice = payload.voice
            if payload.speed:
                tts_engine.speed = payload.speed

            result = tts_engine.generate_speech(payload.text)
            wav_bytes = _tensor_to_wav_bytes(result["tts_speech"], result["sample_rate"])
            return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/asr/transcribe", tags=["ASR"])
    async def asr_transcribe(file: UploadFile = File(...)):
        received_at = datetime.utcnow().isoformat() + "Z"
        t0 = time.perf_counter()
        try:
            contents = await file.read()
            data, samplerate = sf.read(io.BytesIO(contents), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Ensure audio is at 16kHz for the ASR model
            if samplerate != 16000:
                factor = 16000 / samplerate
                data = np.interp(
                    np.arange(0, len(data) * factor) / factor,
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
                samplerate = 16000
            audio_tensor = torch.from_numpy(data).to(ASR_DEVICE)

            hypotheses = asr_model.transcribe(audio=audio_tensor, return_hypotheses=True)
            if isinstance(hypotheses, tuple):
                hypotheses = hypotheses[0]
            if not hypotheses:
                proc_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "received_at": received_at,
                    "processing_ms": proc_ms,
                    "text": "",
                }

            hypothesis = hypotheses[0]
            word_timestamps = []
            if hasattr(hypothesis, "timestamp") and isinstance(hypothesis.timestamp, dict):
                word_timestamps = hypothesis.timestamp.get("word", [])

            proc_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "received_at": received_at,
                "processing_ms": proc_ms,
                "text": hypothesis.text,
                "word_timestamps": [
                    {"word": w.word, "start": w.start, "end": w.end} for w in word_timestamps
                ],
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/pipeline/process", tags=["Pipeline"])
    async def pipeline_process(file: UploadFile = File(...)):
        received_at = datetime.utcnow().isoformat() + "Z"
        t0_api = time.perf_counter()
        try:
            contents = await file.read()
            data, samplerate = sf.read(io.BytesIO(contents), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            transcription, llm_resp, wav_bytes, lat = _process_pipeline(data, samplerate)
            encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")
            api_proc_ms = (time.perf_counter() - t0_api) * 1000.0
            return {
                "received_at": received_at,
                "processing_ms": api_proc_ms,
                "transcription": transcription,
                "llm_response": llm_resp,
                "audio_base64": encoded_audio,
                "latencies": lat,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------- Gradio helpers & UI ------------------------------

    def _gradio_llm(prompt: str, use_thinking: bool):
        return llm_host.query(prompt, enable_thinking=use_thinking)

    def _gradio_tts(text: str):
        result = tts_engine.generate_speech(text)
        return result["sample_rate"], result["tts_speech"].cpu().numpy()

    def _gradio_asr(audio):
        if audio is None:
            return ""
        if isinstance(audio, dict):
            # some gradio versions send dict
            audio = audio.get('data')
        if audio is None:
            return ""
        sample_rate, data = audio
        if sample_rate != 16000:
            factor = 16000 / sample_rate
            data = np.interp(
                np.arange(0, len(data) * factor) / factor,
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        audio_tensor = torch.from_numpy(data.astype("float32")).to(ASR_DEVICE)
        hypotheses = asr_model.transcribe(audio=audio_tensor, return_hypotheses=True)
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        if not hypotheses:
            return ""
        return hypotheses[0].text

    def _gradio_pipeline(audio):
        if audio is None:
            empty = np.zeros(1, dtype=np.float32)
            return "", "", (16000, empty)
        sample_rate, data = audio
        transcription, llm_resp, wav_bytes, lat = _process_pipeline(data, sample_rate)
        try:
            wav_np, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception:
            wav_np = np.zeros(1, dtype=np.float32)
        return transcription, llm_resp, (16000, wav_np)

    def _build_gradio_ui():
        with gr.Blocks(title="VoiceLatency Demo") as demo:
            gr.Markdown("# VoiceLatency Demo\nInteract with the Language Model, Text-to-Speech, and ASR capabilities.")
            with gr.Tab("LLM"):
                prompt_box = gr.Textbox(lines=5, label="Prompt")
                llm_btn = gr.Button("Generate")
                llm_use_thinking = gr.Checkbox(label="Use thinking", value=True)
                llm_out = gr.Textbox(label="Response")
                llm_btn.click(_gradio_llm, inputs=[prompt_box, llm_use_thinking], outputs=llm_out)
            with gr.Tab("TTS"):
                tts_text = gr.Textbox(lines=5, label="Text")
                tts_btn = gr.Button("Generate Speech")
                tts_audio = gr.Audio(label="Audio Output", type="numpy")
                tts_btn.click(_gradio_tts, inputs=tts_text, outputs=tts_audio)
            with gr.Tab("ASR"):
                asr_audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or upload audio")
                cached_audio = gr.State()
                asr_audio_in.change(lambda a: a, inputs=asr_audio_in, outputs=cached_audio, queue=False)
                asr_btn = gr.Button("Transcribe")
                asr_out = gr.Textbox(label="Transcription")
                asr_btn.click(_gradio_asr, inputs=cached_audio, outputs=asr_out)
            with gr.Tab("Conversation"):
                conv_status = gr.Markdown("**Click start to open the real-time conversation (mic required).**")
                start_btn = gr.Button("Start Conversation", variant="primary")
                stop_btn = gr.Button("Stop Conversation")
                refresh_btn = gr.Button("🔄 Manual Refresh (Debug)", variant="secondary")
                
                # Debug info
                debug_info = gr.Textbox(label="Debug Info", interactive=False)
                
                chatbot = gr.Chatbot(height=400, show_copy_button=True)
                conversation_history = gr.State([])  # Store conversation history in Gradio state

                def _get_debug_info():
                    """Get debug information about queue status."""
                    try:
                        queue_size = cm.ui_queue.qsize()
                        is_running = cm.is_running
                        chat_hist_len = len(cm.chat_history)
                        return f"Queue size: {queue_size}, Running: {is_running}, Chat history: {chat_hist_len}"
                    except Exception as e:
                        return f"Error getting debug info: {e}"

                def _start_conv():
                    print("DEBUG: _start_conv called")
                    if not cm.is_running:
                        print("DEBUG: Starting conversation manager...")
                        cm.start()  # This will clear queues and chat history
                        return (
                            "🟢 Conversation running. Speak into the microphone.",
                            [],  # reset conversation history
                            [],  # clear chatbot
                            _get_debug_info()  # update debug info
                        )
                    print("DEBUG: Conversation manager already running")
                    return (
                        "⚠️ Conversation already running.",
                        gr.update(),  # don't change conversation history
                        gr.update(),  # don't change chatbot
                        _get_debug_info()  # update debug info
                    )

                def _stop_conv():
                    print("DEBUG: _stop_conv called")
                    if cm.is_running:
                        print("DEBUG: Stopping conversation manager...")
                        cm.stop()
                        return "🔴 Conversation stopped.", _get_debug_info()
                    print("DEBUG: Conversation manager was not running")
                    return "Conversation not running.", _get_debug_info()

                def _poll_ui_queue(current_history):
                    """Poll the UI queue for new conversation items and update the chatbot."""
                    try:
                        print("DEBUG: _poll_ui_queue called")
                        
                        if not cm.is_running:
                            print("DEBUG: Conversation not running, returning no update")
                            return gr.update(), current_history, _get_debug_info()

                        # Check for new items in the UI queue
                        new_items = []
                        try:
                            while True:
                                ui_item = cm.ui_queue.get_nowait()
                                new_items.append(ui_item)
                                print(f"DEBUG: Got UI item: {ui_item}")
                        except:
                            pass  # Queue is empty, which is expected
                        
                        if not new_items:
                            print("DEBUG: No new UI items, returning no update")
                            return gr.update(), current_history, _get_debug_info()
                        
                        # Add new items to conversation history
                        updated_history = current_history.copy() if current_history else []
                        for item in new_items:
                            conversation_pair = [item['user_text'], item['assistant_text']]
                            updated_history.append(conversation_pair)
                            print(f"DEBUG: Added conversation pair: {conversation_pair}")
                        
                        print(f"DEBUG: Updating chatbot with {len(updated_history)} total conversations")
                        return updated_history, updated_history, _get_debug_info()
                    
                    except Exception as e:
                        print(f"ERROR in _poll_ui_queue: {e}")
                        return gr.update(), current_history, _get_debug_info()

                start_btn.click(
                    _start_conv, 
                    outputs=[conv_status, conversation_history, chatbot, debug_info]
                )
                stop_btn.click(
                    _stop_conv, 
                    outputs=[conv_status, debug_info]
                )
                
                # Manual refresh for debugging
                refresh_btn.click(
                    _poll_ui_queue,
                    inputs=[conversation_history],
                    outputs=[chatbot, conversation_history, debug_info]
                )

                print("DEBUG: Manual refresh configured - use the refresh button to test")

        return demo

    demo_app = _build_gradio_ui()
    app = mount_gradio_app(app, demo_app, path="/gradio")

    return app

# ---------------------------------------------------------------------------
# Script entry point --------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn, os

    reload_flag = os.getenv("RELOAD", "0") == "1"
    port = int(os.getenv("PORT", 8000))

    if reload_flag:
        uvicorn.run(
            "src.server:create_app",
            factory=True,
            host="0.0.0.0",
            port=port,
            reload=True,
        )
    else:
        # Instantiate app once and run without the reloader parent
        uvicorn.run(create_app(), host="0.0.0.0", port=port)