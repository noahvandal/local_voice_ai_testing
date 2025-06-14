import pyaudio
import wave
import numpy as np
import threading
import queue
import time
import os
from typing import Optional, Dict, List, Tuple
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeEmotionAnalyzer:
    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        format: int = pyaudio.paFloat32,
        window_duration: float = 3.0,  # Duration of rolling window in seconds
        smoothing_window: int = 3,     # Number of chunks to average for smoothing
        emotion_callback: Optional[callable] = None
    ):
        """
        Initialize the real-time emotion analyzer using SenseVoice.
        """
        logger.info("Initializing RealTimeEmotionAnalyzer...")
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.window_duration = window_duration
        self.smoothing_window = smoothing_window
        self.emotion_callback = emotion_callback
        
        # Calculate number of chunks in window
        self.chunks_in_window = int(window_duration * sample_rate / chunk_size)
        logger.info(f"Window size: {self.chunks_in_window} chunks")
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Initialize SenseVoice model
        logger.info("Loading SenseVoice model...")
        try:
            self.model = AutoModel(
                model=model_id,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={
                    "max_single_segment_time": 30000,
                    "threshold": 0.3,  # Lower threshold for more sensitive detection
                    "min_speech_duration": 0.1,  # Shorter minimum speech duration
                    "min_silence_duration": 0.1  # Shorter minimum silence duration
                },
                device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                disable_update=True  # Disable update check to speed up initialization
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Buffer for storing audio chunks
        self.audio_buffer = deque(maxlen=self.chunks_in_window)
        self.buffer_lock = threading.Lock()
        
        # Recent emotion results for smoothing
        self.recent_emotions = deque(maxlen=smoothing_window)
        
        # Emotion labels mapping (SenseVoice format)
        self.emotion_labels = {
            "HAPPY": "happy",
            "SAD": "sad",
            "ANGRY": "angry",
            "NEUTRAL": "neutral",
            "FEARFUL": "fearful",
            "DISGUSTED": "disgusted",
            "SURPRISED": "surprised"
        }
        
        logger.info("Initialization complete")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio chunks in real-time."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        if self.is_recording:
            try:
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

    def _process_audio_buffer(self):
        """Process the current audio buffer and analyze emotions."""
        try:
            with self.buffer_lock:
                # Combine chunks into a single audio array
                audio_data = b''.join(self.audio_buffer)
                
                # Save temporary WAV file
                temp_wav = "temp_audio.wav"
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)
                
                logger.debug("Processing audio with SenseVoice...")
                
                # Analyze emotions using SenseVoice
                result = self.model.generate(
                    input=temp_wav,
                    cache={},
                    language="en",  # Explicitly set language to English
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15
                )
                
                if result and len(result) > 0:
                    raw_text = result[0].get('text', '')

                    # Extract emotion from the raw_text
                    emotion = 'NEUTRAL'  # Default value
                    # The emotion tags from SenseVoice are like <|HAPPY|>
                    for label in self.emotion_labels.keys():
                        if f'<|{label.upper()}|>' in raw_text.upper():
                            emotion = label
                            break
                    
                    text = rich_transcription_postprocess(raw_text)
                    
                    # Add to recent emotions for smoothing
                    self.recent_emotions.append(emotion)
                    
                    # Get most common emotion from recent results
                    if self.recent_emotions:
                        from collections import Counter
                        emotion_counts = Counter(self.recent_emotions)
                        smoothed_emotion = emotion_counts.most_common(1)[0][0]
                    else:
                        smoothed_emotion = emotion
                    
                    # Create result dictionary
                    emotion_result = {
                        'timestamp': time.time(),
                        'emotion': smoothed_emotion,
                        'text': text,
                        'raw_emotion': emotion
                    }
                    
                    logger.debug(f"Emotion detected: {smoothed_emotion}")
                    
                    # Call callback if provided
                    if self.emotion_callback:
                        self.emotion_callback(emotion_result)
                
                # Clean up temp file
                os.remove(temp_wav)
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

    def start_recording(self):
        """Start recording and analyzing audio."""
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
        """Stop recording and analyzing audio."""
        if self.is_recording:
            logger.info("Stopping audio recording...")
            self.is_recording = False
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
        self.audio.terminate()


if __name__ == "__main__":
    def emotion_callback(result):
        """Callback function to handle emotion updates."""
        # Clear previous line
        print("\033[K", end='', flush=True)
        
        # Print timestamp
        timestamp = result.get('timestamp', time.time())
        print(f"\r[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] ", end='', flush=True)
        
        # Print emotion and text
        emotion = result.get('emotion', 'NEUTRAL')
        text = result.get('text', '')
        print(f"Emotion: {emotion.upper():<10} Text: {text}", end='', flush=True)
        
        print()  # New line for next update

    print("Initializing Real-Time Emotion Analyzer...")
    try:
        analyzer = RealTimeEmotionAnalyzer(
            window_duration=3.0,  # 3 second window
            smoothing_window=3,   # Average over 3 chunks
            emotion_callback=emotion_callback
        )
        
        print("\nStarting audio recording and emotion analysis...")
        print("Press Ctrl+C to stop recording\n")
        
        analyzer.start_recording()
        
        while True:
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if 'analyzer' in locals():
            analyzer.stop_recording()
        print("\nRecording stopped.")