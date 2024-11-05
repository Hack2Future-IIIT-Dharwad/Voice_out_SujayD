from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from unsloth_zoo.dataset_utils import train_on_responses_only

# Set up device and model parameters
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

torch.cuda.init()

# Define 4bit pre-quantized models
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Data preparation
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Train the model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=True,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(per_device_train_batch_size=2,gradient_accumulation_steps=4,warmup_steps=5,max_steps=60,learning_rate=2e-4,fp16=not torch.cuda.is_available(),optim="adamw_8bit",weight_decay=0.01,lr_scheduler_type="linear",seed=3407,output_dir="outputs",report_to="none"),)
trainer = train_on_responses_only(trainer,instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")

# Inference
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True,
                         temperature=1.5, min_p=0.1)
print(tokenizer.batch_decode(outputs))

# Save model
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
