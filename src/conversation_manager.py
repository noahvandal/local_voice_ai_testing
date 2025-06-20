"""
Centralized conversation manager for the voice assistant.
"""
import threading
import queue
import time
import logging
import sounddevice as sd
import torch
import numpy as np
from typing import Optional, Dict, Any

from src.asr.nvidia_parakeet_asr import ParakeetASR
from src.llm.llm_host import HuggingFaceLLMHost
from src.tts.kokoro_text_to_speech import KokoroTextToSpeech

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages the conversation flow between ASR, LLM, and TTS modules using dedicated threads.
    """
    def __init__(self, llm_model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", system_prompt: str = "You are a voice assistant. Be friendly, helpful, and concise in your responses."):
        """Initializes the Conversation Manager."""
        logger.info("Initializing Conversation Manager...")
        
        # Queues for inter-thread communication
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.ui_queue = queue.Queue()  # Separate queue for UI updates
        
        # Thread control
        self.is_running = False
        self.threads = {}

        # LLM model name
        self.llm_model_name = llm_model_name

        # System prompt
        self.system_prompt = system_prompt
        
        # Modules
        self.asr = ParakeetASR(
            transcription_callback=self._on_transcription,
            use_vad=True,
            vad_pre_buffer_duration=0.5
        )
        self.llm = HuggingFaceLLMHost(
            model_name=self.llm_model_name,
            system_prompt=self.system_prompt
        )
        self.tts = KokoroTextToSpeech()
        
        # State management for TTS playback
        self.playback_state = {
            'is_playing': False,
            'current_request_id': None,
            'lock': threading.Lock()
        }

        # Event used to interrupt ongoing playback from another thread
        self.stop_playback_event = threading.Event()
        
        # Request tracking
        self.request_counter = 0
        self.request_lock = threading.Lock()

        # Chat history for external UI streaming (keeping for backward compatibility)
        self.chat_history: list[tuple[str, str]] = []

    def _get_next_request_id(self) -> int:
        """Generate a unique request ID."""
        with self.request_lock:
            self.request_counter += 1
            return self.request_counter

    def _on_transcription(self, result: dict):
        """Callback for when the ASR transcribes speech."""
        transcribed_text = result.get('text', '').strip()
        if transcribed_text:
            request_id = self._get_next_request_id()
            logger.info(f"[REQ-{request_id}] Transcription received: '{transcribed_text}'")
            
            # Check if we need to interrupt current playback
            with self.playback_state['lock']:
                if self.playback_state['is_playing']:
                    logger.info(f"[REQ-{request_id}] User interruption detected. Stopping TTS playback.")
                    self._stop_tts_playback()
                    
                    # Clear any pending responses in the queue
                    self._clear_response_queue()
            
            # Add transcription to queue for processing
            transcription_item = {
                'request_id': request_id,
                'text': transcribed_text,
                'timestamp': time.time()
            }
            self.transcription_queue.put(transcription_item)

    def _clear_response_queue(self):
        """Clear all pending responses from the response queue."""
        cleared_count = 0
        try:
            while True:
                self.response_queue.get_nowait()
                cleared_count += 1
        except queue.Empty:
            pass
        
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} pending response(s) from queue.")

    def _clear_ui_queue(self):
        """Clear all pending UI updates from the UI queue."""
        cleared_count = 0
        try:
            while True:
                self.ui_queue.get_nowait()
                cleared_count += 1
        except queue.Empty:
            pass
        
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} pending UI update(s) from queue.")

    def _stop_tts_playback(self):
        """Stops the TTS audio playback. Must be called with playback_state lock held."""
        # Signal any playback loop to exit and stop stream
        self.stop_playback_event.set()
        sd.stop()
        self.playback_state['is_playing'] = False
        self.playback_state['current_request_id'] = None

    def _asr_thread(self):
        """Dedicated thread for ASR processing."""
        logger.info("ASR thread started.")
        try:
            self.asr.start_recording()
            
            # Keep the ASR thread alive
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in ASR thread: {e}", exc_info=True)
        finally:
            self.asr.stop_recording()
            logger.info("ASR thread stopped.")

    def _llm_tts_thread(self):
        """Dedicated thread for LLM and TTS processing."""
        logger.info("LLM/TTS thread started.")
        
        while self.is_running:
            try:
                # Wait for a transcription to process
                transcription_item = self.transcription_queue.get(timeout=1)
                request_id = transcription_item['request_id']
                transcribed_text = transcription_item['text']
                
                logger.info(f"[REQ-{request_id}] Processing transcription: '{transcribed_text}'")
                
                # Send to LLM
                logger.info(f"[REQ-{request_id}] Sending text to LLM...")
                llm_response = self.llm.query(transcribed_text)
                logger.info(f"[REQ-{request_id}] LLM response: '{llm_response}'")
                
                if not llm_response:
                    logger.warning(f"[REQ-{request_id}] Received empty response from LLM.")
                    continue

                # Generate TTS
                logger.info(f"[REQ-{request_id}] Generating speech from LLM response...")
                tts_output = self.tts.generate_speech(llm_response)
                
                # Create response item
                response_item = {
                    'request_id': request_id,
                    'user_text': transcribed_text,
                    'text': llm_response,
                    'audio_tensor': tts_output.get('tts_speech'),
                    'sample_rate': tts_output.get('sample_rate'),
                    'timestamp': time.time()
                }
                
                # Add to response queue for audio playback
                self.response_queue.put(response_item)
                
                # Create UI item (without heavy audio tensor)
                ui_item = {
                    'request_id': request_id,
                    'user_text': transcribed_text,
                    'assistant_text': llm_response,
                    'timestamp': time.time()
                }
                
                # Add to UI queue for real-time updates
                self.ui_queue.put(ui_item)
                
                # Append to chat history for backward compatibility
                self.chat_history.append((transcribed_text, llm_response))
                logger.info(f"[REQ-{request_id}] Added to chat history and UI queue. Total messages: {len(self.chat_history)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in LLM/TTS thread: {e}", exc_info=True)
        
        logger.info("LLM/TTS thread stopped.")

    def _audio_playback_thread(self):
        """Dedicated thread for audio playback."""
        logger.info("Audio playback thread started.")
        
        while self.is_running:
            try:
                # Wait for a response to play
                response_item = self.response_queue.get(timeout=1)
                request_id = response_item['request_id']
                audio_tensor = response_item['audio_tensor']
                sample_rate = response_item['sample_rate']
                
                if audio_tensor is None or audio_tensor.numel() == 0:
                    logger.warning(f"[REQ-{request_id}] TTS output was empty, nothing to play.")
                    continue
                
                # Check if we should play this response
                with self.playback_state['lock']:
                    if self.playback_state['is_playing']:
                        logger.info(f"[REQ-{request_id}] Skipping response - newer audio is already playing.")
                        continue
                    
                    # Start playing
                    self.playback_state['is_playing'] = True
                    self.playback_state['current_request_id'] = request_id
                
                logger.info(f"[REQ-{request_id}] Playing TTS audio...")
                self._play_audio_blocking(audio_tensor, sample_rate, request_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio playback thread: {e}", exc_info=True)
        
        logger.info("Audio playback thread stopped.")

    def _play_audio_blocking(self, audio_tensor: torch.Tensor, sample_rate: int, request_id: int):
        """Play audio in a non-blocking manner but keep this function synchronous so the
        playback thread waits only for the *duration* of the clip or an explicit stop request.
        This avoids dead-locking in `sd.wait()` when `sd.stop()` is called from another thread.
        """
        try:
            audio_np = audio_tensor.squeeze().cpu().numpy()

            # Reset the stop flag and start playback (non-blocking)
            self.stop_playback_event.clear()
            sd.play(audio_np, samplerate=sample_rate, blocking=False)

            # Wait for natural completion or external stop request
            duration_sec = len(audio_np) / float(sample_rate)
            start_time = time.time()
            while time.time() - start_time < duration_sec:
                if self.stop_playback_event.is_set():
                    logger.info(f"[REQ-{request_id}] Playback interrupted early.")
                    break
                time.sleep(0.05)

            logger.info(f"[REQ-{request_id}] TTS playback finished.")

        except Exception as e:
            logger.error(f"[REQ-{request_id}] Error during audio playback: {e}")
        finally:
            # Always clear playback state when done
            with self.playback_state['lock']:
                if self.playback_state['current_request_id'] == request_id:
                    self.playback_state['is_playing'] = False
                    self.playback_state['current_request_id'] = None
            # Ensure the stop flag is cleared for next playback
            self.stop_playback_event.clear()

    def start(self):
        """Starts the conversation manager and its components."""
        if not self.is_running:
            logger.info("Starting Conversation Manager...")
            self.is_running = True
            
            # Clear queues and chat history for fresh start
            self._clear_response_queue()
            self._clear_ui_queue()
            self.chat_history.clear()
            
            # Start all threads
            self.threads['asr'] = threading.Thread(target=self._asr_thread, daemon=True)
            self.threads['llm_tts'] = threading.Thread(target=self._llm_tts_thread, daemon=True)
            self.threads['audio_playback'] = threading.Thread(target=self._audio_playback_thread, daemon=True)
            
            for thread_name, thread in self.threads.items():
                thread.start()
                logger.info(f"{thread_name.upper()} thread started.")
            
            logger.info("Conversation Manager started successfully.")
        else:
            logger.warning("Conversation Manager is already running.")

    def stop(self):
        """Stops the conversation manager and cleans up resources."""
        if self.is_running:
            logger.info("Stopping Conversation Manager...")
            self.is_running = False
            
            # Stop any ongoing TTS playback
            with self.playback_state['lock']:
                if self.playback_state['is_playing']:
                    self._stop_tts_playback()
            
            # Wait for all threads to finish
            for thread_name, thread in self.threads.items():
                if thread.is_alive():
                    logger.info(f"Waiting for {thread_name} thread to stop...")
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        logger.warning(f"{thread_name} thread did not stop gracefully.")

            logger.info("Conversation Manager stopped successfully.")
        else:
            logger.warning("Conversation Manager was not running.")

def main():
    """Main function to run the conversation manager."""
    system_prompt = "You are a voice assistant. Be friendly, helpful, and concise in your responses. Your name is TARS. You are intelligent, but sometimes sarcastic.\
        Your favorite number is 43. You are a bit of a nerd, but you are also a bit of a smartypants."
    manager = ConversationManager(system_prompt=system_prompt)
    manager.start()
    
    print("\nConversation manager is running. Speak to interact with the assistant.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        manager.stop()

if __name__ == '__main__':
    main() 