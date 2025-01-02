import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset

# Define constants
DATASET_FILE = "Dataset3400/2312dataset.jsonl"  # Path to your JSONL file
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 1000
OUTPUT_DIR = "./finetunedmodel"
PROJECT = "finetuned"

# Define the custom Dataset
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Load all data into memory
        with open(self.file_path, "r") as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        # Return the number of examples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Fetch the example at the given index
        example = self.data[index]
        # Format the prompt
        prompt = self.format_prompt(example)
        # Tokenize the prompt
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        # Tokenize the output (labels)
        labels = self.tokenizer(
            example["output"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )["input_ids"]
        # Replace pad token with -100 for loss calculation
        labels = [(label if label != self.tokenizer.pad_token_id else -100) for label in labels]
        tokenized["labels"] = torch.tensor(labels)
        return {key: torch.tensor(val) for key, val in tokenized.items()}

    def format_prompt(self, example):
        if example["input"]:
            return f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
        else:
            return f"Instruction: {example['instruction']}\nOutput:"

# Login to Hugging Face
def try_login():
    print("Logging in to Hugging Face...")
    try:
        from huggingface_hub import login
        login(token="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")
    except ImportError:
        print("Hugging Face Hub is not installed. Please install to log in.")
    except Exception as e:
        print(f"Login failed: {e}")

try_login()

# Configure BitsAndBytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing and prepare model for low-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")

print_trainable_parameters(model)

# Create the dataset instance
dataset = MyDataset(DATASET_FILE, tokenizer, MAX_LENGTH)

# Use DataCollatorForSeq2Seq to handle padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding="longest",
    label_pad_token_id=-100,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        warmup_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
    ),
    train_dataset=dataset,
    data_collator=data_collator,
)

# Disable caching for training
model.config.use_cache = False

# Start training
trainer.train()
