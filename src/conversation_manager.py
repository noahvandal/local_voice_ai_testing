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
from src.tts.kokoro_text_to_speech import TextToSpeech

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages the conversation flow between ASR, LLM, and TTS modules using dedicated threads.
    """
    def __init__(self):
        """Initializes the Conversation Manager."""
        logger.info("Initializing Conversation Manager...")
        
        # Queues for inter-thread communication
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Thread control
        self.is_running = False
        self.threads = {}
        
        # Modules
        self.asr = ParakeetASR(
            transcription_callback=self._on_transcription,
            use_vad=True,
            vad_pre_buffer_duration=0.5
        )
        self.llm = HuggingFaceLLMHost(
            model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
            system_prompt="You are a voice assistant. Be friendly, helpful, and concise in your responses."
        )
        self.tts = TextToSpeech()
        
        # State management for TTS playback
        self.playback_state = {
            'is_playing': False,
            'current_request_id': None,
            'lock': threading.Lock()
        }
        
        # Request tracking
        self.request_counter = 0
        self.request_lock = threading.Lock()

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

    def _stop_tts_playback(self):
        """Stops the TTS audio playback. Must be called with playback_state lock held."""
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
                    'text': llm_response,
                    'audio_tensor': tts_output.get('tts_speech'),
                    'sample_rate': tts_output.get('sample_rate'),
                    'timestamp': time.time()
                }
                
                # Add to response queue
                self.response_queue.put(response_item)
                
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
        """Play audio and block until finished."""
        try:
            audio_np = audio_tensor.squeeze().cpu().numpy()
            sd.play(audio_np, samplerate=sample_rate)
            sd.wait()  # Block until playback is finished
            
            logger.info(f"[REQ-{request_id}] TTS playback finished naturally.")
            
        except Exception as e:
            logger.error(f"[REQ-{request_id}] Error during audio playback: {e}")
        finally:
            # Always clear playback state when done
            with self.playback_state['lock']:
                if self.playback_state['current_request_id'] == request_id:
                    self.playback_state['is_playing'] = False
                    self.playback_state['current_request_id'] = None

    def start(self):
        """Starts the conversation manager and its components."""
        if not self.is_running:
            logger.info("Starting Conversation Manager...")
            self.is_running = True
            
            # Start all threads
            self.threads['asr'] = threading.Thread(target=self._asr_thread, daemon=True)
            self.threads['llm_tts'] = threading.Thread(target=self._llm_tts_thread, daemon=True)
            self.threads['audio_playback'] = threading.Thread(target=self._audio_playback_thread, daemon=True)
            
            for thread_name, thread in self.threads.items():
                thread.start()
                logger.info(f"{thread_name.upper()} thread started.")
            
            logger.info("Conversation Manager started.")

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

            logger.info("Conversation Manager stopped.")

def main():
    """Main function to run the conversation manager."""
    manager = ConversationManager()
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