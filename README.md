# Quickly run a local voice agent

## Run your OWN voice agent in a MINUTE!

Run a voice agent with thinking and logic on a small GPU like a NVIDIA 3060 RTX Laptop with less than 5Gb of GPU memory! In near real time also. (about ~1.0-1.5 s run response on small GPU)

## Setup 
 * Ensure you have `uv` installed. 
 * Clone this repo in the intended folder `git clone https://github.com/noahvandal/local_voice_ai_testing.git`
 * Enter the folder `cd local_voice_ai_testing`.
 * Give exe permissions (`chmod +x setup.sh`) and then run `bash setup.sh` from inside the root dir of the folder. 

## Running
 * To run the server, call `uv run python -m main`. 
 * To run just the conversational assistant in the terminal, run `uv run python -m src.conversation_manager`. 


## Use
 * You can make endoint calls at `http://localhost:8000/{endpoint}`, or use the docs webpage `http://localhost:8000/docs`. 
 * A gradio template for easier navigation and use is located at `http://localhost:8000/gradio`. 


 ## Models used
 ### ASR:
* ASR (automated speech recognition) is accomplished by a Silero VAD (voice activity detection) ML model detecting speech from input audio device, which is then batched and sent to NVIDIA parakeet 0.6B model (specifically the `nvidia/parakeet-tdt-0.6b-v2`). This then does transcription on the words (at low WER too I might add). 
### LLM: 
* The current LLM is using Unsloth's package for smaller model size, with a default use of `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`. However, any model accepted by `transformers`'s `AutoModelForCausualLM` will work, provided the GPU can support the size.
### TTS: 
* For low latency and good fidelity we are using the Kokoro 82B TTS. 

## Synchronization
* The thread synchronizatoin is taken care of in the `src/conversation_manager.py:ConversationManager` class. When running the server, it spins up an instance of this class and just calls methods for interacting with the various component variables of that class (e.g., ASR, TTS, LLM). 

## Personalization
Personalizing the manner of the response from the voice agent is a function of a few things; 
* (1) The model used
* (2) The system prompt
* (3) Hyperparameter settings (temperature, top_k, etc).

For our use case we are going to focus on the system prompt. This is where you can add instructions to make the personality fit what you desire. For example, if you want to make it sarcastic, say so: `you are a sarcastic assistant...`. More guides on prompting can be found online, but just know that any character traits or personality instinctives must be explicitly declared. 

* Pro tip: Instead of using regular text ("your name is TARS"), embed your logic in an XML framwork 
```
<personality>
    <name>TARS</name>
    <tone>Sarcastic, professional, funny</tone>
</personality>
```
For whatever reason, LLMs seem to work better in this format. Play around and test it out to see if you have the same response.