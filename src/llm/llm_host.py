"""
Huggingface LLM host.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceLLMHost:
    """
    A host for Hugging Face language models.

    This class loads a specified model and tokenizer from Hugging Face and
    provides an interface to send queries to the model.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", always_use_thinking: bool = False):
        """
        Initializes the LLM host.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.history = []
        self.always_use_thinking = always_use_thinking
        print("Model loaded successfully.")

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []
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
    # Default to Qwen/Qwen3-1.7B, but allow overriding from command line
    import sys
    model_to_use = "Qwen/Qwen3-1.7B"
    always_use_thinking = False
    if len(sys.argv) > 1:
        model_to_use = sys.argv[1]

    llm_host = HuggingFaceLLMHost(model_name=model_to_use, always_use_thinking=always_use_thinking)

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