"""
Sesame text to speech module using Speechmatics' Sesame model.
"""
import os
import sys
import torch
import sounddevice as sd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from speechmatics.models import SesameTTS

class TextToSpeech:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "sesame-csm-1b",
        api_key: Optional[str] = None
    ):
        """
        Initialize the TextToSpeech class with Sesame model.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            model_name: Name of the Sesame model to use
            api_key: Optional Speechmatics API key for cloud inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY")
        
        if not self.api_key:
            raise ValueError("Speechmatics API key is required. Set it via api_key parameter or SPEECHMATICS_API_KEY environment variable.")
            
        self.model = SesameTTS(
            model_name=model_name,
            api_key=self.api_key,
            device=self.device
        )
        self.sample_rate = 24000  # Sesame's default sample rate

    def generate_speech(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate speech from text using Sesame.
        
        Args:
            text: Text to convert to speech
            speaker_id: Optional speaker ID for voice cloning
            emotion: Optional emotion to apply to the speech
            
        Returns:
            Dictionary containing the generated speech tensor and metadata
        """
        try:
            # Generate speech using the model
            audio = self.model.generate(
                text=text,
                speaker_id=speaker_id,
                emotion=emotion
            )
            
            return {
                'tts_speech': audio,
                'sample_rate': self.sample_rate,
                'text': text,
                'speaker_id': speaker_id,
                'emotion': emotion
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
    
    # Get API key from environment or user input
    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        print("Please enter your Speechmatics API key:")
        api_key = input("> ").strip()
    
    tts = TextToSpeech(api_key=api_key)
    print("System ready!")
    
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