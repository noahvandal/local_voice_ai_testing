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

from src.asr.nvidia_parakeet_asr import ParakeetASR
from src.llm.llm_host import HuggingFaceLLMHost
from src.tts.kokoro_text_to_speech import TextToSpeech

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages the conversation flow between ASR, LLM, and TTS modules.
    """
    def __init__(self):
        """Initializes the Conversation Manager."""
        logger.info("Initializing Conversation Manager...")
        
        # Queues for inter-thread communication
        self.transcription_queue = queue.Queue()
        
        # Modules
        # The ASR uses VAD, so it will call the callback on end-of-speech.
        self.asr = ParakeetASR(
            transcription_callback=self._on_transcription,
            use_vad=True,
            vad_pre_buffer_duration=0.5
        )
        self.llm = HuggingFaceLLMHost(
            model_name="Qwen/Qwen3-1.7B",
            system_prompt="You are a voice assistant. Be friendly, helpful, and concise in your responses."
        )
        self.tts = TextToSpeech()
        
        # State management for TTS playback
        self.is_playing_tts = False
        self.playback_lock = threading.Lock()
        
        # Main processing thread
        self.is_running = False
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)

    def _on_transcription(self, result: dict):
        """Callback for when the ASR transcribes speech."""
        transcribed_text = result.get('text', '').strip()
        if transcribed_text:
            logger.info(f"Transcription received: '{transcribed_text}'")
            
            # If the user speaks while TTS is playing, interrupt it.
            with self.playback_lock:
                if self.is_playing_tts:
                    logger.info("User interruption detected. Stopping TTS playback.")
                    self.stop_tts_playback()
            
            self.transcription_queue.put(transcribed_text)

    def stop_tts_playback(self):
        """Stops the TTS audio playback."""
        sd.stop()
        with self.playback_lock:
            self.is_playing_tts = False

    def _play_audio_interruptible(self, audio_tensor: torch.Tensor, sample_rate: int):
        """Plays audio in a non-blocking, interruptible manner."""
        
        def finished_callback():
            logger.info("TTS playback finished naturally.")
            with self.playback_lock:
                self.is_playing_tts = False

        with self.playback_lock:
            self.is_playing_tts = True
        
        audio_np = audio_tensor.squeeze().cpu().numpy()
        sd.play(audio_np, samplerate=sample_rate, callback=finished_callback)

    def _process_loop(self):
        """The main loop for processing transcriptions and generating responses."""
        while self.is_running:
            try:
                # 1. Wait for a transcription from the ASR module
                transcribed_text = self.transcription_queue.get(timeout=1)
                
                # 2. Send the transcribed text to the LLM for a response
                logger.info("Sending text to LLM...")
                llm_response = self.llm.query(transcribed_text)
                logger.info(f"LLM response: '{llm_response}'")
                
                if not llm_response:
                    logger.warning("Received empty response from LLM.")
                    continue

                # 3. Generate speech from the LLM's response
                logger.info("Generating speech from LLM response...")
                tts_output = self.tts.generate_speech(llm_response)
                audio_tensor = tts_output.get('tts_speech')
                sample_rate = tts_output.get('sample_rate')
                
                # 4. Play the generated audio
                if audio_tensor is not None and audio_tensor.numel() > 0:
                    logger.info("Playing TTS audio...")
                    self._play_audio_interruptible(audio_tensor, sample_rate)
                else:
                    logger.warning("TTS output was empty, nothing to play.")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                
    def start(self):
        """Starts the conversation manager and its components."""
        if not self.is_running:
            logger.info("Starting Conversation Manager...")
            self.is_running = True
            
            # Start the ASR recording
            self.asr.start_recording()
            
            # Start the main processing thread
            self.processing_thread.start()
            logger.info("Conversation Manager started.")

    def stop(self):
        """Stops the conversation manager and cleans up resources."""
        if self.is_running:
            logger.info("Stopping Conversation Manager...")
            self.is_running = False
            
            # Stop ASR
            self.asr.stop_recording()
            
            # Stop any ongoing TTS playback
            self.stop_tts_playback()
            
            # Wait for the processing thread to finish
            if self.processing_thread.is_alive():
                self.processing_thread.join()

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