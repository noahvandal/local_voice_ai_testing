#! /bin/bash

# Set up environment
uv init
uv sync 
source .venv/bin/activate

# Needed locally?
uv add pip
