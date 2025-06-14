"""
Nvidia 0.6B Parakeet ASR.
"""
import os
import torch
import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, List, Tuple, Callable
import nemo.collections.asr as nemo_asr
from collections import deque
import logging
from omegaconf import open_dict
from .silero_realtime import SileroRealTime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParakeetASR:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        format: int = pyaudio.paFloat32,
        window_duration: float = 3.0,  # Duration of rolling window in seconds
        transcription_callback: Optional[Callable[[Dict], None]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_vad: bool = False,
        vad_pre_buffer_duration: float = 0.5  # Seconds of pre-buffer
    ):
        """
        Initialize the Parakeet ASR system for real-time transcription.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            chunk_size: Number of frames per buffer
            channels: Number of audio channels
            format: Audio format (default: 32-bit float)
            window_duration: Duration of audio window for processing
            transcription_callback: Optional callback function for transcription results
            device: Device to run the model on ("cuda" or "cpu")
            use_vad: Whether to use Silero VAD for speech detection
            vad_pre_buffer_duration: Seconds of audio to keep before speech starts
        """
        logger.info("Initializing ParakeetASR...")
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.window_duration = window_duration
        self.transcription_callback = transcription_callback
        self.device = device
        self.use_vad = use_vad
        self.vad_pre_buffer_duration = vad_pre_buffer_duration
        
        # Initialize VAD if enabled. This may override chunk_size.
        self.vad = None
        if self.use_vad:
            logger.info("Initializing Silero VAD...")
            vad_chunk_size = 512  # Silero VAD requires specific chunk sizes
            self.chunk_size = vad_chunk_size
            self.vad = SileroRealTime(
                sample_rate=self.sample_rate,
                window_size_samples=vad_chunk_size,
                threshold_on=0.4,
                threshold_off=0.8,
                min_speech_duration_ms=100,
                min_silence_duration_ms=500,
                standalone=False
            )

        # Calculate number of chunks in window
        self.chunks_in_window = int(window_duration * sample_rate / self.chunk_size)
        logger.info(f"Window size: {self.chunks_in_window} chunks")
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Initialize Parakeet model
        logger.info("Loading Parakeet model...")
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2"
            )
            self.model = self.model.to(self.device)

            # Update decoding config for timestamps
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.compute_timestamps = True
            self.model.change_decoding_strategy(decoding_cfg)

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Buffer for storing audio chunks
        self.buffer_lock = threading.Lock()
        if self.use_vad:
            self.audio_buffer = deque()  # No maxlen when using VAD
            # Calculate pre-buffer size in chunks
            pre_buffer_chunks = int(self.vad_pre_buffer_duration * self.sample_rate / self.chunk_size) if self.vad_pre_buffer_duration > 0 else 0
            self.pre_speech_buffer = deque(maxlen=pre_buffer_chunks)
        else:
            self.audio_buffer = deque(maxlen=self.chunks_in_window)
        
        logger.info("Initialization complete")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio chunks in real-time."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        if self.is_recording:
            try:
                if self.use_vad and self.vad:
                    audio_chunk_np = np.frombuffer(in_data, dtype=np.float32)
                    is_speech = self.vad.process_chunk(audio_chunk_np)
                    
                    with self.buffer_lock:
                        if is_speech:
                            # If audio_buffer is empty, it's the start of a new utterance.
                            # Prepend the pre-speech buffer.
                            if not self.audio_buffer:
                                logger.debug(f"Speech start detected. Prepending {len(self.pre_speech_buffer)} chunks.")
                                self.audio_buffer.extend(list(self.pre_speech_buffer))
                                self.pre_speech_buffer.clear()
                            
                            self.audio_buffer.append(in_data)
                        else:  # Not speech
                            # If buffer has content, speech just ended
                            if self.audio_buffer:
                                buffer_copy = list(self.audio_buffer)
                                self.audio_buffer.clear()
                                
                                if buffer_copy:
                                    logger.debug(f"VAD detected end of speech, processing buffer of {len(buffer_copy)} chunks...")
                                    threading.Thread(target=self._process_audio_buffer, args=(buffer_copy,), daemon=True).start()

                            # Keep filling the pre-speech buffer during silence
                            self.pre_speech_buffer.append(in_data)

                else: # Original behavior without VAD
                    with self.buffer_lock:
                        # Add new chunk to buffer
                        self.audio_buffer.append(in_data)
                        logger.debug(f"Added chunk to buffer. Buffer size: {len(self.audio_buffer)}")
                        
                        # Process if we have enough data
                        if len(self.audio_buffer) >= self.chunks_in_window:
                            logger.debug("Processing audio buffer...")
                            # Process in a separate thread to avoid blocking
                            threading.Thread(target=self._process_audio_buffer, daemon=True).start()
            except Exception as e:
                logger.error(f"Error in audio callback: {str(e)}")
        
        return (in_data, pyaudio.paContinue)

    def _process_audio_buffer(self, audio_data_list: Optional[List[bytes]] = None):
        """Process the current audio buffer and transcribe audio."""
        try:
            if audio_data_list:
                # Using data passed from VAD
                audio_data = b''.join(audio_data_list)
            else:
                # Using the shared buffer (non-VAD rolling window)
                with self.buffer_lock:
                    # Combine chunks into a single audio array
                    if not self.audio_buffer:
                        return
                    audio_data = b''.join(self.audio_buffer)

            if not audio_data:
                return

            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Reshape if stereo
            if self.channels == 2:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np.copy()).to(self.device)
            
            logger.debug("Processing audio with Parakeet...")
            
            # Transcribe with timestamps
            hypotheses = self.model.transcribe(
                audio=audio_tensor,
                return_hypotheses=True
            )

            # For transducer models, the output can be a tuple
            if isinstance(hypotheses, tuple) and len(hypotheses) > 0:
                hypotheses = hypotheses[0]
            
            if hypotheses and len(hypotheses) > 0:
                hypothesis = hypotheses[0]
                word_timestamps = []

                if hasattr(hypothesis, 'timestamp') and isinstance(hypothesis.timestamp, dict):
                    word_timestamps = hypothesis.timestamp.get('word', [])
                
                # Create result dictionary
                transcription_result = {
                    'timestamp': time.time(),
                    'text': hypothesis.text,
                    'word_timestamps': word_timestamps
                }
                
                logger.debug(f"Transcription: {hypothesis.text}")
                
                # Call callback if provided
                if self.transcription_callback:
                    self.transcription_callback(transcription_result)
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

    def start_recording(self):
        """Start recording and transcribing audio."""
        if not self.is_recording:
            logger.info("Starting audio recording...")
            self.is_recording = True
            
            try:
                # List available audio devices
                logger.info("Available audio devices:")
                for i in range(self.audio.get_device_count()):
                    dev_info = self.audio.get_device_info_by_index(i)
                    logger.info(f"Device {i}: {dev_info['name']}")
                
                # Try to open the default input device
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback,
                    start=False  # Don't start the stream immediately
                )
                
                # Start the stream
                self.stream.start_stream()
                logger.info("Audio stream started successfully")
                
            except Exception as e:
                logger.error(f"Error starting audio stream: {str(e)}")
                self.is_recording = False
                if self.stream:
                    self.stream.close()
                    self.stream = None
                raise

    def stop_recording(self):
        """Stop recording and transcribing audio."""
        if self.is_recording:
            logger.info("Stopping audio recording...")
            self.is_recording = False
            
            if self.vad:
                self.vad.stop()

            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.error(f"Error stopping audio stream: {str(e)}")
            self.stream = None
            logger.info("Audio recording stopped")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


if __name__ == "__main__":
    def transcription_callback(result):
        """Callback function to handle transcription updates."""
        # Clear previous line
        print("\033[K", end='', flush=True)
        
        # Print timestamp
        timestamp = result.get('timestamp', time.time())
        print(f"\r[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] ", end='', flush=True)
        
        # Print transcription with word timestamps
        text = result.get('text', '')
        word_timestamps = result.get('word_timestamps', [])
        
        print(f"Text: {text}")
        if word_timestamps:
            print("\nWord timestamps:")
            for word_info in word_timestamps:
                start_time = word_info.start
                end_time = word_info.end
                word = word_info.word
                print(f"{start_time:.2f}s - {end_time:.2f}s: {word}")
        
        print()  # New line for next update

    print("Initializing Parakeet ASR...")
    try:
        asr = ParakeetASR(
            window_duration=3.0,  # 3 second window
            transcription_callback=transcription_callback,
            use_vad=True,  # Enable VAD
            vad_pre_buffer_duration=0.5  # Add 0.5 seconds of audio before speech starts
        )
        
        print("\nStarting audio recording and transcription...")
        print("Press Ctrl+C to stop recording\n")
        
        asr.start_recording()
        
        while True:
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if 'asr' in locals():
            asr.stop_recording()
        print("\nRecording stopped.")