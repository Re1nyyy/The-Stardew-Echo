"""
ROMA + QLoRA multi-role fine-tuning + inference template
Filename: roma_qlo_ra_multirole.py

What this script provides:
- Utilities to load characterbook.json and worldbook.json
- Simple Role Selection Network (R_phi) implemented as an MLP
- Creation of role embeddings (optionally from a sentence-transformer encoder)
- Dataset builder that turns dialogues + role/world context into instruction-response pairs
- QLoRA-style fine-tuning setup using HuggingFace Transformers + PEFT
- Multi-role dialogue inference function that: given conversation history, samples a role z_i and generates a role-consistent response

Notes:
- This is a template. You will need to install required packages (transformers, datasets, accelerate, peft, bitsandbytes, sentence-transformers (optional), torch).
- Paths expected: book/characterbook.json and book/worldbook.json and input_conversations/ (or adapt the paths below).
- The QLoRA training uses `bitsandbytes` and `peft`. Hyperparameters are placeholders.

Use this as a starting point and adapt to your project structure and compute.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb
except Exception:
    # If imports fail here, user should pip install the required packages
    raise


# -----------------------------
# Utilities: load books & conversations
# -----------------------------

def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_conversation_files(input_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(input_dir):
        if fn.endswith('.json') or fn.endswith('.jsonl') or fn.endswith('.txt'):
            files.append(os.path.join(input_dir, fn))
    return files


# -----------------------------
# Role Embedding Builder
# -----------------------------

class RoleEmbeddingBuilder:
    """
    Build initial role embeddings from character descriptions. Tries to use a sentence-transformer
    encoder to build fixed embeddings; if unavailable, initializes random embeddings (learnable).
    """
    def __init__(self, role_names: List[str], device='cpu', encoder_model: str = None, dim: int = 768):
        self.role_names = role_names
        self.device = device
        self.dim = dim
        self.encoder_model = encoder_model

        # store as torch.nn.Parameter when training / fine-tuning
        self.role_embeddings = nn.ParameterDict()

        if encoder_model is not None:
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(encoder_model)
                for rn in role_names:
                    emb = encoder.encode(rn, convert_to_tensor=True)
                    emb = emb.float().to(device)
                    p = nn.Parameter(emb)
                    self.role_embeddings[rn] = p
            except Exception:
                # fall back to random
                for rn in role_names:
                    p = nn.Parameter(torch.randn(dim, device=device) * 0.01)
                    self.role_embeddings[rn] = p
        else:
            for rn in role_names:
                p = nn.Parameter(torch.randn(dim, device=device) * 0.01)
                self.role_embeddings[rn] = p

    def get_embedding(self, role_name: str) -> torch.Tensor:
        return self.role_embeddings[role_name]

    def state_dict(self):
        return {k: v.detach().cpu().numpy().tolist() for k, v in self.role_embeddings.items()}


# -----------------------------
# Dataset builder for QLoRA fine-tuning
# -----------------------------

# -----------------------------
# Dataset builder for QLoRA fine-tuning (改：手动 padding，兼容无 pad_token 的 tokenizer)
# -----------------------------
class MultiRoleDataset(Dataset):
    """
    Builds examples and performs manual padding so we do NOT rely on tokenizer.add_special_tokens().
    Each example returns tensors: input_ids, attention_mask, labels (labels use -100 on padding).
    """
    def __init__(self, tokenizer, conversations: List[Dict], characterbook: Dict, worldbook: Dict, max_length=1024):
        self.tok = tokenizer
        self.conversations = conversations
        self.characterbook = characterbook
        self.worldbook = worldbook
        self.max_length = int(max_length)

        self.examples = []
        self._build_examples()

    def _make_context(self, role_name: str) -> str:
        char = self.characterbook.get(role_name, {})
        parts = []
        parts.append("World:")
        for k, v in list(self.worldbook.items())[:5]:
            parts.append(f"{k}: {v}")
        parts.append("Character:")
        for kk, vv in char.items():
            parts.append(f"{kk}: {vv}")
        return "\n".join(parts)

    def _build_examples(self):
        # choose a pad id fallback (we will still prefer tokenizer.pad_token_id if present)
        pad_id_fallback = None
        if getattr(self.tok, "pad_token_id", None) is not None:
            pad_id_fallback = int(self.tok.pad_token_id)
        elif getattr(self.tok, "eos_token_id", None) is not None:
            pad_id_fallback = int(self.tok.eos_token_id)
        elif getattr(self.tok, "unk_token_id", None) is not None:
            pad_id_fallback = int(self.tok.unk_token_id)
        else:
            pad_id_fallback = 0

        for conv in self.conversations:
            role = conv.get('role')
            history = conv.get('history', [])
            reply = conv.get('reply', '')

            context = self._make_context(role)
            history_text = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in history])

            prompt = f"#CONTEXT\n{context}\n#HISTORY\n{history_text}\n#ROLE\n{role}\n#REPLY\n"
            full_input = prompt + reply

            # Tokenize WITHOUT padding (we will pad manually)
            enc = self.tok(full_input, truncation=True, max_length=self.max_length, padding=False)
            input_ids = enc.get('input_ids')
            # ensure a Python list
            if isinstance(input_ids, (list, tuple)):
                input_ids = list(input_ids)
            elif hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            else:
                input_ids = [int(x) for x in input_ids]

            attn = [1] * len(input_ids)

            # If shorter than max_length -> pad manually
            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                pad_id = pad_id_fallback
                input_ids = input_ids + [pad_id] * pad_len
                attn = attn + [0] * pad_len
            else:
                # already truncated during tokenization
                input_ids = input_ids[:self.max_length]
                attn = attn[:self.max_length]

            labels = [lid if aid == 1 else -100 for lid, aid in zip(input_ids, attn)]

            self.examples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attn, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def build_conversations_from_input(input_dir: str, history_window: int = 6) -> List[Dict]:
    """
    读取 input_dir 中的 .json / .jsonl 文件（支持两种常见格式）并把每一次“说话/发言”转换为训练样本：
      sample = {
        "role": <角色名：turn['character'] 优先，否则用 turn['role']>,
        "history": [ { "speaker": <character_or_role>, "text": <content> }, ... ]  # 最近 history_window 条
        "reply": <当前 turn 的 content>
        "source_file": <来源文件名>  # 可选，便于溯源
      }

    生成策略：
      - 遍历文件内的 conversation 列表（如果文件为 jsonl，则每行可能包含一个对象）
      - 对每一条 turn 都生成一个样本（你也可以改为只对 assistant 生成样本）
      - history 包含当前发言之前的若干条（默认 6 条）
    """
    conversations_out = []

    def _load_file(path):
        # 支持 .jsonl（每行一个 json）和普通 .json（可能包含 {"conversation":[...]}）
        records = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # 每行可能就是 {"conversation": [...]} 或直接就是一个 turn。
                    if isinstance(rec, dict) and "conversation" in rec and isinstance(rec["conversation"], list):
                        records.extend(rec["conversation"])
                    elif isinstance(rec, list):
                        records.extend(rec)
                    else:
                        # 如果每行本身就是一个 turn（role/character/content）
                        records.append(rec)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "conversation" in data and isinstance(data["conversation"], list):
                    records = data["conversation"]
                elif isinstance(data, list):
                    records = data
                else:
                    # 如果文件就是单条 turn 的 dict
                    if isinstance(data, dict):
                        records = [data]
        return records

    file_list = list_conversation_files(input_dir)
    for p in file_list:
        try:
            turns = _load_file(p)
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
            continue
        # normalize: ensure turns is a list of dicts with keys role/character/content
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            content = (turn.get("content") or turn.get("text") or "").strip()
            if not content:
                continue

            # choose role/character name (优先 character)
            role_name = turn.get("character") or turn.get("role") or "Unknown"

            # build history: previous up to history_window turns (as speaker/text)
            start = max(0, i - history_window)
            history = []
            for prev in turns[start:i]:
                if not isinstance(prev, dict):
                    continue
                speaker = prev.get("character") or prev.get("role") or "Unknown"
                text = (prev.get("content") or prev.get("text") or "").strip()
                if text == "":
                    continue
                history.append({"speaker": speaker, "text": text})

            sample = {
                "role": role_name,
                "history": history,
                "reply": content,
                "source_file": os.path.basename(p)
            }
            conversations_out.append(sample)

    return conversations_out

# -----------------------------
# Training loop 
# -----------------------------
def train_qloa(model_path: str,
               conversations: List[Dict],
               characterbook: Dict,
               worldbook: Dict,
               output_dir: str = 'output_qloa',
               epochs: int = 3,
               batch_size: int = 1,
               max_length: int = 1024,
               lr: float = 5e-5,
               val_split: float = 0.1,
               grad_accum_steps: int = 4,
               weight_decay: float = 0.01,
               max_grad_norm: float = 1.0,
               eval_steps: int = 50,
               print_steps: int = 10,
               patience: int = 3,
               max_steps: int = None):
    import math
    from copy import deepcopy
    from transformers import get_linear_schedule_with_warmup

    os.makedirs(output_dir, exist_ok=True)

    # load tokenizer & model
    print("Loading tokenizer and model from", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    # ensure pad token handled without adding tokens (Qwen-style)
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif getattr(tokenizer, "bos_token", None) is not None:
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.pad_token_id = tokenizer.bos_token_id
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            # fallback (dataset already handles manual padding fallback id)
            pass

    # LoRA config: more conservative r to help generalization
    print("Applying LoRA (PEFT)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )
    model = get_peft_model(model, lora_config)

    # build dataset
    full_dataset = MultiRoleDataset(tokenizer, conversations, characterbook, worldbook, max_length=max_length)
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError("No training samples found (conversations empty). Check input files.")

    # train/val split
    indices = list(range(dataset_size))
    random.shuffle(indices)
    val_count = max(1, int(dataset_size * val_split)) if val_split > 0 else 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) if val_count > 0 else None

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # compute total training steps
    if max_steps is not None:
        total_steps = max_steps
    else:
        # steps per epoch = ceil(len(train_dataset)/(batch_size*grad_accum_steps))
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size / grad_accum_steps)
        total_steps = steps_per_epoch * epochs

    warmup_steps = max(1, int(0.03 * total_steps))  # small warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.to(device)
    model.train()

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_steps = 0
    global_step = 0
    completed_steps = 0

    print(f"Starting training on device={device}, epochs={epochs}, batch_size={batch_size}, total_steps~{total_steps}")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            loss = loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps  # accumulate original loss

            if (step + 1) % grad_accum_steps == 0:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                completed_steps += 1

                if global_step % print_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    print(f"  step {global_step} (epoch_step {step}) avg_loss {avg_loss:.4f}")

                # evaluation
                if (val_dataloader is not None) and (global_step % eval_steps == 0):
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_dataloader:
                            vb = {k: v.to(device) for k, v in vb.items()}
                            out = model(**vb)
                            vloss = out.loss if hasattr(out, "loss") else out["loss"]
                            val_losses.append(vloss.item())
                    avg_val_loss = sum(val_losses) / max(1, len(val_losses))
                    print(f"  >>> eval at global_step {global_step}: val_loss = {avg_val_loss:.4f}")
                    # early stopping & save best
                    if avg_val_loss < best_val_loss - 1e-6:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch
                        no_improve_steps = 0
                        print(f"  *** New best val_loss {best_val_loss:.4f} - saving checkpoint")
                        model.save_pretrained(os.path.join(output_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(output_dir, "best"))
                    else:
                        no_improve_steps += 1
                        print(f"  No improvement count: {no_improve_steps}/{patience}")
                        if no_improve_steps >= patience:
                            print("Early stopping triggered (no improvement).")
                            # load best (if any) and return
                            try:
                                best_path = os.path.join(output_dir, "best")
                                if os.path.isdir(best_path):
                                    print("Loading best checkpoint before exit.")
                                    # best model is PEFT-wrapped, reloading here might be optional; we just stop
                            except Exception:
                                pass
                            # final save current as last
                            model.save_pretrained(os.path.join(output_dir, "last"))
                            tokenizer.save_pretrained(os.path.join(output_dir, "last"))
                            return tokenizer, model
                    model.train()

                # hard stop if reached max_steps
                if max_steps is not None and global_step >= max_steps:
                    print(f"Reached max_steps {max_steps}. Stopping training.")
                    model.save_pretrained(os.path.join(output_dir, "last"))
                    tokenizer.save_pretrained(os.path.join(output_dir, "last"))
                    return tokenizer, model

        # epoch end - small eval
        if (val_dataloader is not None):
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_dataloader:
                    vb = {k: v.to(device) for k, v in vb.items()}
                    out = model(**vb)
                    vloss = out.loss if hasattr(out, "loss") else out["loss"]
                    val_losses.append(vloss.item())
            avg_val_loss = sum(val_losses) / max(1, len(val_losses))
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                no_improve_steps = 0
                print("  *** New best val_loss at epoch end - saving checkpoint")
                model.save_pretrained(os.path.join(output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(output_dir, "best"))
            else:
                no_improve_steps += 1
                print(f"  No improvement count: {no_improve_steps}/{patience}")
                if no_improve_steps >= patience:
                    print("Early stopping triggered (no improvement).")
                    model.save_pretrained(os.path.join(output_dir, "last"))
                    tokenizer.save_pretrained(os.path.join(output_dir, "last"))
                    return tokenizer, model

    # training finished
    print("Training complete. Saving final model to", output_dir)
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")
    return tokenizer, model




# -----------------------------
# Simple Role Selection Network R_phi
# -----------------------------

class RoleSelector(nn.Module):
    """
    A small MLP that takes a pooled embedding of the conversation state and outputs a distribution over roles.
    """
    def __init__(self, input_dim: int, n_roles: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_roles)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


# -----------------------------
# QLoRA fine-tuning helper
# -----------------------------

def detect_lora_target_modules(model) -> List[str]:
    """
    Try to autodetect a reasonable set of target module names for LoRA depending on model implementation.
    Returns a small list like ['q_proj','k_proj','v_proj','o_proj'] or ['c_attn','c_proj'] etc.
    """
    names = set(n for n, _ in model.named_modules())
    # common patterns:
    if any("q_proj" in n for n in names) and any("k_proj" in n for n in names):
        # transformer implementations that separate q/k/v/o
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if any("c_attn" in n for n in names) and any("c_proj" in n for n in names):
        # fused attention (eg. some GPT-style implementations)
        return ["c_attn", "c_proj"]
    # fallback: try q_proj/v_proj
    if any("q_proj" in n for n in names) or any("v_proj" in n for n in names):
        return ["q_proj", "v_proj"]
    # last resort common names
    return ["q_proj", "v_proj", "k_proj", "o_proj"]





# -----------------------------
# Multi-role inference: select role + generate
# -----------------------------

@torch.no_grad()
def infer_multi_role(model, tokenizer, role_selector: RoleSelector, role_emb_builder: RoleEmbeddingBuilder,
                     conversation_history: List[Dict], candidate_roles: List[str], device='cuda',
                     max_new_tokens=128, temperature=0.8, top_p=0.95):
    """
    conversation_history: list of messages {"speaker": "Bob", "text": "..."}
    candidate_roles: list of role names to choose from

    Steps:
    1. Pool conversation history with tokenizer to get a fixed embedding vector (simple approach: mean of token embeddings from LM's embed)
    2. Use role_selector to compute logits over roles
    3. Sample role z
    4. Build prompt combining world + role + history
    5. Generate response from model
    """
    # 1) compute simple pooled state embedding using tokenizer + model embeddings
    history_text = "\n".join([f"{m['speaker']}: {m['text']}" for m in conversation_history[-6:]])
    prompt = f"#HISTORY\n{history_text}\n"

    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = enc['input_ids'].to(device)

    # Get token embeddings from model's embedding layer if available
    if hasattr(model, 'get_input_embeddings'):
        emb_layer = model.get_input_embeddings()
        with torch.no_grad():
            token_embs = emb_layer(input_ids)  # (1, L, D)
            pooled = token_embs.mean(dim=1)
    else:
        pooled = torch.randn(1, role_selector.net[0].in_features, device=device)

    # 2) compute role logits
    logits = role_selector(pooled)  # (1, n_roles)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # 3) sample a role index
    role_idx = int(torch.multinomial(torch.tensor(probs), num_samples=1).item())
    role_choice = candidate_roles[role_idx]

    # 4) create full prompt with world and character info
    # We expect user to provide small world/char strings externally if needed; here's a quick placeholder
    role_context = []
    role_context.append(f"#ROLE\n{role_choice}\n")
    history_prompt = prompt
    full_prompt = "#CONTEXT\n" + history_prompt + "\n" + "#ROLE_CONTEXT\n" + role_context[0] + "\n#REPLY\n"

    input_ids_prompt = tokenizer(full_prompt, return_tensors='pt').input_ids.to(device)

    outputs = model.generate(
        input_ids=input_ids_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen = tokenizer.decode(outputs[0][input_ids_prompt.shape[-1]:], skip_special_tokens=True)
    return role_choice, gen


# -----------------------------
# Example: glue everything together
# -----------------------------




if __name__ == '__main__':
    # Paths
    CHAR_PATH = 'book/characterbook.json'
    WORLD_PATH = 'book/worldbook.json'
    INPUT_DIR = 'input_conversations'

    # Load
    characterbook = load_json(CHAR_PATH)
    worldbook = load_json(WORLD_PATH)
    conversations = build_conversations_from_input(INPUT_DIR)

    # Example role list
    roles = list(characterbook.keys()) if len(characterbook) > 0 else ['RoleA', 'RoleB']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize role embedding builder (optional sentence-transformer encoder)
    role_emb_builder = RoleEmbeddingBuilder(roles, device=device, encoder_model=None, dim=768)

    # Simple role selector
    role_selector = RoleSelector(input_dim=768, n_roles=len(roles)).to(device)

    # To train QLoRA: uncomment and set model_path
    model_path = r"D:\project\11111\Qwen-1_8B"  # example
    tokenizer, model = train_qloa(model_path, conversations, characterbook, worldbook, output_dir='roma_qloa_output')

    # Example inference (after loading a trained causal LM and tokenizer)
    # For demo we only show how to call infer_multi_role; user should load finetuned model & tokenizer
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('roma_qloa_output')
    # model = AutoModelForCausalLM.from_pretrained('roma_qloa_output', device_map='auto', load_in_4bit=False)

    # demo_history = [
    #     {"speaker": "Alice", "text": "Good morning, any news from the mine?"},
    #     {"speaker": "Bob", "text": "I saw strange lights last night near the old well."}
    # ]

    # role_choice, reply = infer_multi_role(model, tokenizer, role_selector, role_emb_builder, demo_history, roles, device=device)
    # print('Selected role:', role_choice)
    # print('Reply:', reply)

    print('Script loaded. Customize paths, model_path, and uncomment training / inference blocks to run.')
