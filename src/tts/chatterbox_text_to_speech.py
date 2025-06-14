"""
Text to speech module using Chatterbox.
"""
import os
import sys
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

class TextToSpeech:
    def __init__(
        self,
        device: str = "cuda",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        reference_audio: Optional[str] = None
    ):
        """
        Initialize the TextToSpeech class with Chatterbox model.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            exaggeration: Emotion exaggeration control (0.0 to 1.0)
            cfg_weight: Configuration weight for speech generation (0.0 to 1.0)
            reference_audio: Path to reference audio file for voice cloning
        """
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.sample_rate = 24000  # Chatterbox's default sample rate
        self.reference_audio = reference_audio

    def generate_speech(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate speech from text using Chatterbox.
        
        Args:
            text: Text to convert to speech
            audio_prompt_path: Optional path to audio file for voice cloning (overrides default reference)
            
        Returns:
            Dictionary containing the generated speech tensor and metadata
        """
        try:
            # Use provided audio prompt or default reference
            prompt_path = audio_prompt_path or self.reference_audio
            
            # Generate speech with voice cloning
            wav = self.model.generate(
                text,
                audio_prompt_path=prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )
            
            return {
                'tts_speech': wav,
                'sample_rate': self.sample_rate,
                'text': text,
                'audio_prompt': prompt_path
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
    # Get the reference audio path
    reference_audio = Path("data/audio/noah_reference_audio.mp3")
    if not reference_audio.exists():
        print(f"Error: Reference audio file not found at {reference_audio}")
        sys.exit(1)
        
    print("Initializing Text-to-Speech system...")
    tts = TextToSpeech(
        exaggeration=0.8,
        cfg_weight=0.5,
        reference_audio=str(reference_audio)
    )
    print("System ready!")
    print(f"Using reference voice from: {reference_audio}")
    
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