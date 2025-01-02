import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# Define constants
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Your base model
CHECKPOINT_DIR = "./finetunedmodel/checkpoint-500"  # Path to your checkpoint directory

# Login to Hugging Face (if necessary)
def try_login():
    print("Logging in to Hugging Face...")
    try:
        login(token="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")
    except ImportError:
        print("Hugging Face Hub is not installed. Please install to log in.")
    except Exception as e:
        print(f"Login failed: {e}")

try_login()

# Load the base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")

# Load the LoRA-adapted model from the checkpoint
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)

# Set the model to evaluation mode
model.eval()

# Now the model is loaded and ready for inference
