import torch
from transformers import AutoTokenizer

def test_environment():
    # Check basic Python environment
    print("Python environment test")
    
    # Check CUDA - gracefully handle if not available
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("Running in CPU mode - this is expected on login nodes")
    
    # Test basic HuggingFace functionality
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
    
    print("Environment test complete!")

if __name__ == "__main__":
    test_environment()
