"""
Simplified LLM module with improved error handling
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union

class SimpleLLM:
    """A simplified LLM wrapper that works with both CPU and GPU"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize a simplified LLM interface
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Import torch and transformers at initialization
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            print(f"Error importing necessary libraries: {e}")
            raise
    
    def load_model(self):
        """Load the model - deferred to allow flexibility"""
        try:
            # Set device
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate settings for the device
            if device == "cuda":
                # Use 8-bit quantization for efficient GPU usage
                self.model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                # Use CPU
                self.model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_name
                )
                self.model.to(device)
            
            print(f"Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on the prompt
        
        Args:
            prompt: Input text
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            success = self.load_model()
            if not success:
                return f"[Error: Model could not be loaded]"
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the newly generated text (remove the prompt)
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error during generation: {e}]"
    
    def get_completion(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Get a completion with optional system prompt
        
        Args:
            user_prompt: User's input
            system_prompt: Optional system instructions
            
        Returns:
            Generated completion
        """
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {user_prompt}\n\nAssistant:"
            
        return self.generate(full_prompt, max_length=500)
