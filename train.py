from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2

filtered_dataset = load_dataset("meta-math/MetaMathQA")["train"]
model = AutoModelForCausalLM.from_pretrained ("Qwen/Qwen2-Math-1.5B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Math-1.5B", padding="max length", truncation=True, max_length=2400)
tokenizer.pad_token = tokenizer.eos_token
model = torch.compile(model)
apply_liger_kernel_to_qwen2()
 
filtered_dataset = load_dataset("meta-math/MetaMathQA")["train"]
 
peft_config = LoraConfig(
    lora_dropout=0.1,
    r=128,
    bias="none",
    task_type="CAUSAL_LM"
)
 
def transform_metamath_mathshepherd(example):
    output_texts = []
    
    for i in range(len(example["query"])):
        final_str = f"{example['query'][i]} ### Answer:"
        steps = example["response"][i].split("\n")
        for i, step in enumerate(steps):
            if i == len(steps) - 1:
                final_str += f"{step} ки"
            elif i == len(steps) - 2:    
                final_str += f"Step {i+1}: {step} "
            else:
                final_str += f"Step {i+1}: {step} ки\n"
        
        output_texts.append(final_str)
        
    return {"text": output_texts}

filtered_dataset = filtered_dataset.map(transform_metamath_mathshepherd, batched=True)   
 
model = get_peft_model(model, peft_config)
 
# args = TrainingArguments(
#     per_device_train_batch_size=3,
#     gradient_accumulation_steps=4,
#     warmup_steps=10,
#     num_train_epochs=3,
#     learning_rate=1e-5,
#     bf16=True,
#     logging_steps=10,
#     remove_unused_columns=True,
#     output_dir="checkpoints/peft_rho_1",
# )
args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 250,
    learning_rate = 2e-4,
    bf16 = True,
    logging_steps = 1,
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    output_dir = "outputs",
)

response_template = " ### Answer:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
 
trainer = SFTTrainer(
    model,
    args=args,
    train_dataset=filtered_dataset,
    packing = False,
    max_seq_length=1600,
    # data_collator=collator,
    dataset_text_field = "text",
    tokenizer=tokenizer
    
)
trainer.train()
