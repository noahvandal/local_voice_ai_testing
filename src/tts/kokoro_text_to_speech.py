"""
Kokoro text to speech.
"""
import os
import sys
import torch
import sounddevice as sd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from kokoro import KPipeline

class KokoroTextToSpeech:
    def __init__(
        self,
        lang_code: str = 'a',  # 'a' for American English
        speed: float = 1.0,
        voice: str = 'af_heart'
    ):
        """
        Initialize the TextToSpeech class with Kokoro model.
        
        Args:
            lang_code: Language code ('a' for American English, 'b' for British English, etc.)
            speed: Speech speed multiplier
            voice: Voice to use for speech generation
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self.speed = speed
        self.voice = voice
        self.sample_rate = 24000  # Kokoro's default sample rate

    def generate_speech(
        self,
        text: str,
        split_pattern: str = r'\n+'
    ) -> Dict[str, Any]:
        """
        Generate speech from text using Kokoro.
        
        Args:
            text: Text to convert to speech
            split_pattern: Pattern to split text into chunks
            
        Returns:
            Dictionary containing the generated speech tensor and metadata
        """
        try:
            # Generate speech
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=split_pattern
            )
            
            # Collect all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            
            # Combine all chunks
            combined_audio = np.concatenate(audio_chunks)
            
            return {
                'tts_speech': torch.from_numpy(combined_audio),
                'sample_rate': self.sample_rate,
                'text': text,
                'voice': self.voice
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating speech: {str(e)}")

    def play_audio(self, audio_tensor: torch.Tensor) -> None:
        """
        Play audio directly using sounddevice.
        
        Args:
            audio_tensor: Audio tensor to play
        """
        # Convert to numpy array and ensure it's the right shape
        audio_np = audio_tensor.squeeze().cpu().numpy()
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=0)  # Convert to mono if stereo
            
        # Play the audio
        sd.play(audio_np, self.sample_rate)
        sd.wait()  # Wait until audio is finished playing

def main():
    """Interactive text-to-speech conversion."""
    print("Initializing Text-to-Speech system...")
    tts = KokoroTextToSpeech(
        lang_code='a',  # American English
        speed=1.0,
        voice='af_heart'
    )
    print("System ready!")
    print(f"Using voice: {tts.voice}")
    
    while True:
        try:
            # Get text input
            print("\nEnter text to convert to speech (or 'quit' to exit):")
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not text:
                print("Please enter some text.")
                continue
            
            # Generate and play speech
            result = tts.generate_speech(text)
            tts.play_audio(result['tts_speech'])
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()