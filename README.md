# Quick locally run voice agent

Run a voice agent with thinking and logic on a small GPU like a NVIDIA 3060 RTX Laptop with less than 5Gb of GPU memory! In near real time also. 

## Setup 
 * Ensure you have `uv` installed. 
 * Clone this repo in the intended folder `git clone `
 * Run `setup.sh` from inside the top level of the folder. 

## Running
 * To run the server, call `uv run python -m main`. 
 * To run just the conversational assistant in the terminal, run `uv run python -m src.conversation_manager`. 


## Use
 * You can make endoint calls at `http://localhost:8000/{endpoint}`, or use the docs webpage `http://localhost:8000/docs`. 
 * A gradio template for easier navigation and use is located at `http://localhost:8000/gradio`. 