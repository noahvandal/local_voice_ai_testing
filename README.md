# Quickly run a local voice agent

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