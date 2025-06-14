"""
Huggingface LLM host.
"""
import unsloth
from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceLLMHost:
    """
    A host for Hugging Face language models.

    This class loads a specified model and tokenizer from Hugging Face and
    provides an interface to send queries to the model.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", use_unsloth: bool = False, always_use_thinking: bool = False, system_prompt: str = None):
        """
        Initializes the LLM host.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
            use_unsloth (bool): Whether to use unsloth for loading quantized models.
            always_use_thinking (bool): If True, forces the model to use thinking mode.
            system_prompt (str): An initial system prompt for the model.
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        self.use_unsloth = use_unsloth

        if self.use_unsloth:
            if FastLanguageModel is None:
                raise ImportError("Unsloth is not installed. Please install it with 'pip install unsloth'.")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        self.always_use_thinking = always_use_thinking
        self.system_prompt = system_prompt
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
        print("Model loaded successfully.")

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
        print("Conversation history cleared.")

    def query(self, prompt: str, enable_thinking: bool = True) -> str:
        """
        Sends a prompt to the LLM and returns the response.

        Args:
            prompt (str): The user prompt to send to the model.
            enable_thinking (bool): Whether to use the model's thinking mode.

        Returns:
            str: The model's response.
        """
        # The model handles /think and /no_think flags in the prompt to
        # switch between thinking and non-thinking modes.
        # We adjust generation parameters based on the recommendation.
        if "/no_think" in prompt or not self.always_use_thinking:
            generation_params = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.8,
            }
        else: # Default to thinking mode settings
            generation_params = {
                "max_new_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.95,
            }

        if self.always_use_thinking:
            prompt = f"/think {prompt}"
        else:
            prompt = f"/no_think {prompt}"

        messages = self.history + [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **generation_params
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            # The token id for </think> is 151668 for Qwen3
            think_token_id = 151668
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        if thinking_content and self.always_use_thinking:
            print("\n--- Thinking ---")
            print(thinking_content)
            print("----------------\n")
        
        # Per best practices, only add the final content to the history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": content})

        return content

if __name__ == "__main__":
    # --- Configuration ---
    # Set the model to use from Hugging Face
    model_to_use = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    # Use unsloth for loading quantized models.
    # For this to work, specify a 4-bit quantized model, like 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit'
    use_unsloth = True
    # Always use thinking mode for the model
    always_use_thinking = False
    # -------------------

    if use_unsloth and "unsloth" not in model_to_use:
        print(f"Warning: Using unsloth with model '{model_to_use}'. Consider specifying a 4-bit quantized model from unsloth for better performance, like 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit'.")

    system_prompt = "You are a helpful assistant. Your role is to be friendly and engaging. You have a playful demeanor \
        and can be a little sarcastic and silly at times. Be a good friend, and help the user with their questions in a \
            serious manner if the situation calls for it. Be brief and concise in your responses. Do NOT ramble"

    llm_host = HuggingFaceLLMHost(
        model_name=model_to_use,
        use_unsloth=use_unsloth,
        always_use_thinking=always_use_thinking,
        system_prompt=system_prompt
    )

    print("\nLLM Host is ready. Type your query and press Enter.")
    print("Type 'exit' or 'quit' to end the session. Type 'clear' to reset history.")
    
    while True:
        try:
            user_prompt = input("> ")
            if user_prompt.lower() in ["exit", "quit"]:
                break
            if user_prompt.lower() == "clear":
                llm_host.clear_history()
                continue
            
            response = llm_host.query(user_prompt)
            print(f"\nLLM: {response}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break