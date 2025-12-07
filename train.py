# train.py
import os
import glob
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= 配置参数 =================
# 1. 模型路径 (根据你的 ls 结果)
MODEL_PATH = "/model/ModelScope/Qwen/Qwen3-1.7B"

# 2. 数据路径
DATA_DIR = "./train_data"

# 3. 输出目录
OUTPUT_DIR = "./output_qwen3_stardew"

# 4. 训练超参数
MAX_LENGTH = 2048        # 根据显存大小调整
BATCH_SIZE = 4           # 根据显存大小调整 (1.7B模型通常可以设置大一点)
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
LORA_RANK = 16           # LoRA 秩
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ================= 1. 加载数据 =================
def load_all_jsonl(data_dir):
    # 获取所有 .jsonl 文件
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    print(f"Found {len(files)} data files: {files}")
    
    if not files:
        raise ValueError(f"No jsonl files found in {data_dir}")

    # 加载所有文件并将它们合并
    dataset_list = []
    for file in files:
        # 假设 jsonl 格式包含 'text' 或者是对话格式 'messages'
        # 这里使用通用的 json 加载方式
        ds = load_dataset('json', data_files=file, split='train')
        dataset_list.append(ds)
    
    full_dataset = concatenate_datasets(dataset_list)
    print(f"Total training samples: {len(full_dataset)}")
    return full_dataset

# ================= 2. 数据预处理 =================
def process_func(example, tokenizer):
    # 这里假设你的数据格式是 Qwen 推荐的 ChatML 格式 (messages list)
    # 例如: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    # 如果你的 jsonl 只是纯文本 {"text": "..."}, 需要相应修改此处
    
    MAX_LENGTH = 2048 
    input_ids, attention_mask, labels = [], [], []
    
    # 处理 messages 字段
    if 'messages' in example:
        instruction = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        tokens = tokenizer(
            instruction, 
            add_special_tokens=False, 
            padding=False, 
            truncation=True, 
            max_length=MAX_LENGTH
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # 对于 Causal LM，labels 通常就是 input_ids (忽略 padding -100)
        # 简单的做法是直接复制 input_ids，Trainer 会自动处理 shift
        labels = input_ids + [tokenizer.eos_token_id]
        input_ids = input_ids + [tokenizer.eos_token_id]
        attention_mask = attention_mask + [1]
        
    # 处理纯 text 字段 (如果是纯文本续写)
    elif 'text' in example:
        tokens = tokenizer(
            example['text'] + tokenizer.eos_token, 
            truncation=True, 
            max_length=MAX_LENGTH,
            padding=False
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids

    # 截断以防万一
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ================= 主函数 =================
def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    
    # Qwen 的 pad_token 往往需要手动指定，如果未定义
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Dataset...")
    dataset = load_all_jsonl(DATA_DIR)
    
    # 移除原始列，只保留处理后的 tensor
    column_names = dataset.column_names
    tokenized_dataset = dataset.map(
        lambda x: process_func(x, tokenizer),
        batched=False,
        remove_columns=column_names
    )

    print("Loading Model...")
    # 根据显存情况，可以选择 load_in_8bit=True 或 load_in_4bit=True (需要 bitsandbytes)
    # 1.7B 模型很小，通常 fp16 或 bf16 直接加载即可
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 如果显卡不支持 BF16，改为 torch.float16
        trust_remote_code=True
    )
    
    # 启用梯度检查点以节省显存 (可选)
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()

    # ================= LoRA 配置 =================
    print("Configuring LoRA...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, 
        r=LORA_RANK, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # ================= 训练参数 =================
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_steps=100,
        learning_rate=LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none", # 防止尝试连接 wandb
        fp16=False,       # 如果显卡旧，设为 True，并将上面的 bfloat16 改为 float16
        bf16=True,        # Ampere 架构 (30系/40系/A100) 推荐 True
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("Starting Training...")
    trainer.train()
    
    print(f"Training finished. Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # 保存 tokenizer 以便推理使用
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()