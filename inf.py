import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# Define constants
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Your base model
CHECKPOINT_DIR = "./finetunedmodel/checkpoint-400"  # Path to your checkpoint directory

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
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token to avoid warnings

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")

# Load the LoRA-adapted model from the checkpoint
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)

# Set the model to evaluation mode
model.eval()

# Define a function for inference
def generate_output(instruction, input_text):
    # Format the prompt
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    # Manually create attention mask if not present
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()
    
    # Ensure the input and attention mask are on the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    attention_mask = attention_mask.to(model.device)

    # Generate the output from the model
    with torch.no_grad():
        output = model.generate(input_ids=inputs['input_ids'], 
                                attention_mask=attention_mask,
                                max_length=150, 
                                num_beams=5, 
                                early_stopping=True)
    
    # Decode the generated output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the actual output (after the prompt)
    generated_output = output_text.split("Output:")[-1].strip()
    
    return generated_output

# Example inference (you can loop over your dataset)
instruction = "Generate highlights about this product"
input_text = "A Samsung smartphone"

generated_output = generate_output(instruction, input_text)
print(f"Generated Output: {generated_output}")
