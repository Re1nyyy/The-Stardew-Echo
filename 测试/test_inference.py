"""
Final test_inference.py
Purpose: iterative multi-role dialogue generation using a base causal LM + LoRA adapter (PEFT).
Generates N rounds of conversation in the same format as your training data:
  {"role": "assistant", "character": "<name>", "content": "(...) dialogue + action ..."}

Features:
- Loads base model + LoRA adapter (PEFT)
- Uses a rotating speaker order (configurable) to generate multi-role conversations
- Builds prompts using the same training-style format (few-shot included) to bias output style
- Heuristics to extract a single turn's content from model output robustly
- Saves generated conversation to output jsonl and prints to console

Usage example:
  python test_inference.py --base_model D:\project\11111\Qwen-1_8B \
                           --lora_model D:\project\11111\roma_qloa_output \
                           --rounds 12

"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------- Helper utilities --------------------

def print_jsonl(path: str, conv: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for turn in conv:
            f.write(json.dumps(turn, ensure_ascii=False) + '\n')


def safe_decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


# try to extract a single assistant turn (character + content) from generated text
# returns (character_name, content_text) or (None, None) on failure

def parse_generated_turn(gen_text: str, candidate_chars: List[str]) -> Optional[Dict[str, str]]:
    """
    Heuristics (in order):
      1) if JSON object present -> parse and return if fields exist
      2) find patterns like "character": "NAME" or "character": NAME
      3) find patterns like '角色：NAME' or 'NAME：' or 'NAME:' at line start
      4) fallback: treat whole generation as content and pick next candidate character (caller should choose)
    """
    txt = gen_text.strip()
    # 1) try to find a JSON object
    m = re.search(r"\{[\s\S]*?\}", txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            # accept common keys
            char = obj.get('character') or obj.get('角色') or obj.get('speaker')
            cont = obj.get('content') or obj.get('content_text') or obj.get('text') or obj.get('台词')
            if char and cont:
                return {'character': str(char).strip(), 'content': str(cont).strip()}
        except Exception:
            pass

    # 2) key:value style (e.g. character: "阿比盖尔" or character: 阿比盖尔)
    m = re.search(r"character\s*[:=]\s*[\"']?([^\n\"']+)[\"']?", txt, re.IGNORECASE)
    if m:
        char = m.group(1).strip()
        # content after 'content' key
        m2 = re.search(r"content\s*[:=]\s*[\"']([\s\S]*?)[\"']", txt, re.IGNORECASE)
        if m2:
            return {'character': char, 'content': m2.group(1).strip()}
        # fallback: everything after that line
        rest = txt[m.end():].strip()
        if rest:
            return {'character': char, 'content': rest.split('\n')[0].strip()}

    # 3) Chinese label style: 角色：NAME 或 NAME：
    # look for lines like '角色：阿比盖尔' or '阿比盖尔：......'
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # pattern '角色：NAME' -> next non-empty line is content
    for i, ln in enumerate(lines):
        m = re.match(r"(?:角色|character)\s*[：:\-\s]+(.+)$", ln)
        if m:
            char = m.group(1).strip()
            # next line as content if exists
            if i + 1 < len(lines):
                return {'character': char, 'content': lines[i + 1]}
    # pattern 'NAME：text' or 'NAME: text'
    for ln in lines:
        m = re.match(r"^(.{1,20}?)\s*[：:]\s*(.+)$", ln)
        if m:
            name = m.group(1).strip()
            cont = m.group(2).strip()
            # prefer if name in candidate_chars (or matches roughly)
            for c in candidate_chars:
                if name == c or name.startswith(c) or c.startswith(name):
                    return {'character': c, 'content': cont}
            # if we don't match candidate but it looks plausible, return it
            return {'character': name, 'content': cont}

    # 4) If generation contains parentheses/dialogue like '（...）' take that as start
    m = re.search(r"（[\s\S]{1,200}?）[\s\S]*", txt)
    if m:
        cont = txt
        # if any candidate character name appears in the preceding chunk, pick it
        for c in candidate_chars:
            if c in txt[:200]:
                return {'character': c, 'content': cont.strip()}
        # fallback: just return content (no character)
        return {'character': None, 'content': cont.strip()}

    # 5) fallback: take first sentence
    s = txt.split('\n')[0].strip()
    if s:
        return {'character': None, 'content': s}

    return None


# -------------------- Generation functions --------------------

def build_prompt_from_history(history: List[Dict], few_shot_examples: List[Dict], candidate_chars: List[str]) -> str:
    """Constructs a prompt using the same training-style structure and adds few-shot examples to bias style.
    We provide a clear instruction to produce one next turn in the training format.
    """
    # few-shot block
    fs = ''
    if few_shot_examples:
        fs_lines = ["# FEW-SHOT EXAMPLES (do not change)\n"]
        for ex in few_shot_examples:
            fs_lines.append(f"role: {ex.get('role','user')}")
            fs_lines.append(f"character: {ex.get('character','team')}")
            # ensure long content stays as one line to make parsing easier
            content_single = ex.get('content','').replace('\n',' ')[:1000]
            fs_lines.append(f"content: {content_single}\n")
        fs = '\n'.join(fs_lines) + "\n"

    # history block (use training format)
    hist_lines = ["# CONVERSATION HISTORY\n"]
    for turn in history:
        r = turn.get('role','user')
        c = turn.get('character','team')
        t = turn.get('content','').replace('\n',' ')
        hist_lines.append(f"role: {r}")
        hist_lines.append(f"character: {c}")
        hist_lines.append(f"content: {t}\n")
    hist_block = '\n'.join(hist_lines)

    # instruction block: produce one assistant turn ONLY, in training format
    inst = (
        "# INSTRUCTION: Generate exactly ONE NEXT TURN in the SAME training format.\n"
        "Output must be exactly three lines (role / character / content) and nothing else.\n"
        "The role should be 'assistant'. The character must be one of: " + ', '.join(candidate_chars) + ".\n"
        "Example required format: \nrole: assistant\ncharacter: 阿比盖尔\ncontent: （动作）台词\n\n"
    )

    prompt = fs + hist_block + "\n" + inst
    return prompt


def generate_next_turn(model, tokenizer, history: List[Dict], candidate_chars: List[str], device: torch.device,
                       max_new_tokens:int=160, temperature:float=0.7, top_p:float=0.9, few_shot_examples:List[Dict]=None):
    prompt = build_prompt_from_history(history, few_shot_examples or [], candidate_chars)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # generate
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen = safe_decode(tokenizer, outputs[0])
    # strip prompt prefix
    gen_after = gen[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):] if hasattr(tokenizer, 'decode') else gen
    gen_after = gen_after.strip()

    parsed = parse_generated_turn(gen_after, candidate_chars)

    return parsed, gen_after


# -------------------- Main script --------------------

def main(args):
    # load tokenizer & model
    print(f"Loading base model from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print('Device:', device)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map='auto' if device.type=='cuda' else None,
        low_cpu_mem_usage=True,
    )

    # load peft adapter
    print('Loading LoRA adapter from:', args.lora_model)
    model = PeftModel.from_pretrained(model, args.lora_model)
    model.to(device)
    model.eval()

    # candidate characters: prefer to read from book files if provided, otherwise use args.characters
    if args.characterbook and os.path.exists(args.characterbook):
        try:
            with open(args.characterbook, 'r', encoding='utf-8') as f:
                cb = json.load(f)
                chars = list(cb.keys()) if isinstance(cb, dict) else args.characters
        except Exception:
            chars = args.characters
    else:
        chars = args.characters

    if not chars:
        print('No characters provided. Use --characters or provide book file.')
        sys.exit(1)

    print('Candidate characters:', chars)

    # few-shot examples: small set to bias style (use training-like lines). We include a very short example.
    few_shot_examples = [
        {"role":"assistant","character":"阿比盖尔","content":"（靠在码头，凝视海面）这些水母好像在跳舞，真漂亮。"},
        {"role":"assistant","character":"威利","content":"（扶着栏杆）小心不要靠太近，潮水今天有点异常。"}
    ]

    # initial history: if input file given, load it (we expect same training format), otherwise build from CLI seed
    history: List[Dict] = []
    if args.seed_file and os.path.exists(args.seed_file):
        # load first conversation from seed file
        with open(args.seed_file, 'r', encoding='utf-8') as f:
            # try jsonl or json with conversation
            text = f.read().strip()
            try:
                data = json.loads(text)
            except Exception:
                # try jsonl -> take first line
                with open(args.seed_file, 'r', encoding='utf-8') as fr:
                    first = fr.readline().strip()
                    data = json.loads(first) if first else {}
            # if data contains 'conversation' list, use it
            convs = data.get('conversation') if isinstance(data, dict) else None
            if convs and isinstance(convs, list):
                for t in convs:
                    # convert training turns to our internal format
                    role = t.get('role','user')
                    char = t.get('character') or t.get('speaker') or 'team'
                    content = t.get('content') or t.get('text') or ''
                    history.append({'role': role, 'character': char, 'content': content})
    else:
        # use simple seed from args
        if args.seed_text:
            history.append({'role':'user','character':'team','content':args.seed_text})
        else:
            history.append({'role':'user','character':'team','content':'(场景描写)'})

    # If there are provided starter assistant turns, add them
    if args.starting_turns:
        for s in args.starting_turns:
            history.append({'role':'assistant','character':s[0],'content':s[1]})

    # Generation loop
    generated: List[Dict] = list(history)  # copy

    rounds = args.rounds

    print('\n================= Generating conversation =================')
    for i in range(rounds):
        parsed, raw = generate_next_turn(model, tokenizer, generated, chars, device,
                                         max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature,
                                         top_p=args.top_p,
                                         few_shot_examples=few_shot_examples)

        # choose speaker if parser returned None character
        if parsed is None:
            # fallback: rotate characters
            char = chars[len([t for t in generated if t['role']=='assistant']) % len(chars)]
            content = raw or '(模型未生成文本)'
            new_turn = {'role':'assistant','character':char,'content':content}
            print(f'[{i+1}] Parsed failed, falling back to char={char}')
        else:
            if parsed.get('character') is None:
                # char not found in generation -> choose rotate
                char = chars[len([t for t in generated if t['role']=='assistant']) % len(chars)]
                parsed['character'] = char
            new_turn = {'role':'assistant','character':parsed['character'],'content':parsed['content']}

        generated.append(new_turn)

        # print nicely
        print('\n--- Round', i+1, '->', new_turn['character'], '---')
        print(new_turn['content'])

    # save
    out_path = args.output or 'generated_conversation.jsonl'
    print('\nSaving generated conversation to', out_path)
    print_jsonl(out_path, generated)
    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Iterative multi-role inference script (test_inference.py)')
    p.add_argument('--base_model', type=str, required=True)
    p.add_argument('--lora_model', type=str, required=True)
    p.add_argument('--characterbook', type=str, default='book/characterbook.json')
    p.add_argument('--seed_file', type=str, default='')
    p.add_argument('--seed_text', type=str, default='场景：矿洞入口。昨晚有人听到矿洞深处传来巨大声响。三人正在讨论是否进入调查。')
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=160)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--output', type=str, default='generated_conversation.jsonl')
    p.add_argument('--force_cpu', action='store_true')
    # allow specifying characters directly
    p.add_argument('--characters', nargs='*', default=['阿比盖尔','塞巴斯蒂安','莎莉'])
    # allow adding starting assistant turns: pass as repeated pairs --starting_turns 阿比盖尔 "(动作)台词" ...
    p.add_argument('--starting_turns', nargs='*', default=None)

    args = p.parse_args()

    # parse starting_turns into pairs if provided
    if args.starting_turns:
        pairs = []
        arr = args.starting_turns
        if len(arr) % 2 != 0:
            print('Warning: --starting_turns should be pairs of character and content. Ignoring.')
            args.starting_turns = None
        else:
            for i in range(0, len(arr), 2):
                pairs.append((arr[i], arr[i+1]))
            args.starting_turns = pairs

    main(args)
