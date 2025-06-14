"""
Realtime implementation of Silero VAD for audio streaming
"""

import torch
import pyaudio
import numpy as np
from typing import Optional, Callable
from silero_vad import load_silero_vad
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SileroRealTime:
    def __init__(self, 
                 sample_rate: int = 16000,
                 threshold_on: float = 0.5,
                 threshold_off: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100,
                 window_size_samples: int = 512,
                 smoothing_window: int = 5,
                 speech_callback: Optional[Callable[[bool, float], None]] = None,
                 standalone: bool = True):
        """
        Initialize the Silero VAD real-time detector.
        
        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz)
            threshold_on: Probability threshold to start detecting speech (0.0 to 1.0)
            threshold_off: Probability threshold to stop detecting speech (0.0 to 1.0)
            min_speech_duration_ms: Minimum speech duration in milliseconds to trigger speech state
            min_silence_duration_ms: Minimum silence duration in milliseconds to end speech state
            window_size_samples: Number of samples per window (must be 512 for 16kHz, 256 for 8kHz)
            smoothing_window: Number of probabilities to average for smoothing
            speech_callback: Optional callback for speech state changes
            standalone: If True, manages its own PyAudio stream.
        """
        self.sample_rate = sample_rate
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_callback = speech_callback
        self.standalone = standalone
        
        # Load Silero VAD model
        self.model = load_silero_vad()
        
        # Initialize audio stream only in standalone mode
        if self.standalone:
            self.audio = pyaudio.PyAudio()
            self.stream = None
        
        self.is_running = False
        
        # State tracking
        self.current_speech_state = False
        self.last_speech_start_time = 0
        self.last_silence_start_time = 0
        
        # Smoothing
        self.smoothing_window = smoothing_window
        self.probability_buffer = deque(maxlen=smoothing_window)
        
    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process a single audio chunk and update speech state.
        
        Args:
            audio_chunk: A numpy array of audio data.
            
        Returns:
            True if speech is detected in the current state, False otherwise.
        """
        speech_prob = self.model(torch.from_numpy(audio_chunk), self.sample_rate).item()
        
        self.probability_buffer.append(speech_prob)
        smoothed_prob = sum(self.probability_buffer) / len(self.probability_buffer) if self.probability_buffer else 0.0
        
        current_time = time.time()

        if not self.current_speech_state:
            # We are in a silence state, check for speech start
            if smoothed_prob > self.threshold_on:
                if self.last_speech_start_time == 0:
                    self.last_speech_start_time = current_time
                
                time_since_speech_start = (current_time - self.last_speech_start_time) * 1000
                if time_since_speech_start >= self.min_speech_duration_ms:
                    self.current_speech_state = True
                    self.last_silence_start_time = 0
                    if self.speech_callback:
                        self.speech_callback(True, smoothed_prob)
            else:
                self.last_speech_start_time = 0
        else:
            # We are in a speech state, check for silence start
            if smoothed_prob < self.threshold_off:
                if self.last_silence_start_time == 0:
                    self.last_silence_start_time = current_time
                
                time_since_silence_start = (current_time - self.last_silence_start_time) * 1000
                if time_since_silence_start >= self.min_silence_duration_ms:
                    self.current_speech_state = False
                    self.last_speech_start_time = 0
                    if self.speech_callback:
                        self.speech_callback(False, smoothed_prob)
            else:
                self.last_silence_start_time = 0

        return self.current_speech_state
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Standalone audio callback."""
        if not self.standalone:
            return (in_data, pyaudio.paContinue)
        
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.process_chunk(audio_data)
            
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        """Start real-time speech detection in standalone mode."""
        if not self.standalone:
            logger.error("Cannot start in non-standalone mode.")
            return

        if self.is_running:
            return
            
        self.is_running = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.window_size_samples,
            stream_callback=self._audio_callback
        )
        
        logger.info("Started real-time speech detection...")
        
    def stop(self):
        """Stop real-time speech detection in standalone mode."""
        if not self.standalone:
            return

        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.standalone:
            self.audio.terminate()
        logger.info("Stopped real-time speech detection.")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    def speech_callback(is_speech, probability):
        print(f"\rSpeech detected: {is_speech} (Prob: {probability:.2f})", end="", flush=True)

    with SileroRealTime(
        speech_callback=speech_callback
    ) as vad:
        print("Listening... (Press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")

