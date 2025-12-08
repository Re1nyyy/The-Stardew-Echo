#!/usr/bin/env python3
"""
convert_multichar_chat.py

将各种 ChatGPT/自定义格式的对话记录 批量转换为 roma_qlo_ra_multirole 所需的标准化 JSON 格式。
支持输入类型：JSONL（每行一个 JSON 对象）、完整 JSON（对象或数组）、ChatGPT messages 格式。
自动识别说话人 name/role/character，使用给定角色列表在文本中做角色识别填充（best-effort）。
清理无关 metadata 与常见时间戳/重复项。

输出：每个源文件 -> output_dir/<源名>.converted.json

"""

import os
import json
import re
import argparse
from glob import glob
from typing import List, Dict, Any, Optional

# -------------------------
# Default role list (you gave these)
# -------------------------
DEFAULT_ROLES = [
    "刘易斯","皮埃尔","阿比盖尔","塞巴斯蒂安","谢恩","海莉","莫里斯",
    "莱纳斯","法师","潘姆","罗宾","威利","科罗布斯","肯特","矮人",
    "玛妮","格斯","潘妮","哈维"
]

# -------------------------
# Utilities
# -------------------------
def load_roles(path: Optional[str]) -> List[str]:
    if path and os.path.exists(path):
        roles = []
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    roles.append(ln)
        return roles
    return DEFAULT_ROLES

# clean content: remove weird duplicated whitespace, control chars, timestamps, and some metadata tokens
_timestamp_patterns = [
    # e.g. December 7, 2025 5:04pm
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\s*\d{1,2}:\d{2}\s*(?:am|pm)?\b',
    # ISO-like
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',
    # short datetime like 2025-12-07T09:05:04.157Z or 2025-12-07 09:05:04
    r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?',
    # times like 09:05:04.157Z or 09:04:58.632Z
    r'\d{1,2}:\d{2}:\d{2}(?:\.\d+)?Z?',
    # common "gen_id" numeric tokens (best-effort remove long numeric ids)
    r'\bgen_id[:=]?\d+\b',
]

def clean_text(s: str) -> str:
    if s is None:
        return ""
    # Normalize whitespace and remove control characters
    text = re.sub(r'[\r\t\f\v]+', ' ', s)
    # remove timestamps and similar metadata patterns
    for pat in _timestamp_patterns:
        text = re.sub(pat, ' ', text)
    # remove "send_date":"December 7, 2025 5:04pm" inline patterns (key-like)
    text = re.sub(r'"?send_date"?\s*:\s*"[^"]+"', ' ', text)
    # replace multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    # remove leading/trailing separators like "### ..." if it's purely section header
    text = re.sub(r'^(#+\s*)+', '', text).strip()
    # Remove weird JSON-like keys embedded (best-effort)
    text = re.sub(r'\b(gen_started|gen_finished|time_to_first_token|original_avatar|force_avatar)\b[:=]?\S*', '', text)
    # Optionally remove repeated longer numeric sequences that look like ids
    text = re.sub(r'\b\d{6,}\b', '', text)
    # Trim again
    text = text.strip()
    return text

def detect_character_from_text(text: str, roles: List[str]) -> Optional[str]:
    # search for role names in text, return first match (longest-first to avoid substring issues)
    if not text:
        return None
    # check for "角色：" or "角色：" list style first
    m = re.search(r'角色[:：]\s*([^\n;，。]+)', text)
    if m:
        # pick first name in that listing
        listed = m.group(1)
        # split on common separators
        parts = re.split(r'[，,;；/、]', listed)
        if parts:
            cand = parts[0].strip()
            if cand in roles:
                return cand
            # try trim parentheses
            cand = re.sub(r'[()（）].*?$', '', cand).strip()
            if cand in roles:
                return cand

    # to avoid matching single-character common words, sort roles by length desc
    for r in sorted(roles, key=lambda x: -len(x)):
        if r and re.search(r'(?<!\w)'+re.escape(r)+r'(?!\w)', text):
            return r
    return None

def extract_content_from_obj(obj: Dict[str, Any], roles: List[str]) -> Optional[Dict[str, Any]]:
    """
    Turn a raw JSON object (one record) into normalized dict with keys: role, character, content.
    Best-effort extraction from various shapes.
    """
    # Primary content candidates
    content = None
    # common fields in your examples: 'mes', 'content', 'swipes', 'swipe_info' etc.
    if 'mes' in obj and obj['mes']:
        content = obj['mes']
    elif 'content' in obj and obj['content']:
        # content can be dict/list in some cases, convert to str
        if isinstance(obj['content'], (list, dict)):
            content = json.dumps(obj['content'], ensure_ascii=False)
        else:
            content = obj['content']
    elif 'swipes' in obj and obj['swipes']:
        # prefer first swipe text
        if isinstance(obj['swipes'], list) and len(obj['swipes']) > 0:
            content = obj['swipes'][0]
    elif 'message' in obj:
        content = obj['message']
    elif 'text' in obj:
        content = obj['text']

    content = clean_text(content or "")

    # determine role type
    role = None
    if 'is_user' in obj:
        role = 'user' if obj.get('is_user') else 'assistant'
    elif 'role' in obj:
        # could be 'user','assistant','system'
        role_raw = str(obj['role']).lower()
        if role_raw in ('user','assistant','system'):
            role = role_raw
        else:
            # sometimes role may be a name like '莫里斯' - treat as assistant
            role = 'assistant'
    else:
        # fallback: if name present and is one of roles => assistant, else user
        if 'name' in obj and obj['name']:
            if obj['name'] in roles:
                role = 'assistant'
            else:
                # sometimes system messages have "system" in name
                role = 'assistant'
        else:
            role = 'assistant'

    # determine character name
    character = None
    # priority 1: explicit 'name' or 'character' fields
    if 'character' in obj and obj.get('character'):
        character = obj.get('character')
    elif 'name' in obj and obj.get('name'):
        # exclude "system" or "user" literal occasionally used
        nm = str(obj.get('name')).strip()
        if nm.lower() not in ('system','user','assistant'):
            character = nm

    # if still None and role == assistant: try detect in content using role list
    if character is None and role == 'assistant':
        guessed = detect_character_from_text(content, roles)
        if guessed:
            character = guessed

    # if role == user and character appears in content as "扮演 X" or "现在请你扮演 X"
    if role == 'user' and not character:
        m = re.search(r'扮演\s*[:：]?\s*([^\s，。,；:：]+)', content)
        if m:
            cand = m.group(1).strip()
            if cand in roles:
                character = cand

    # Final clean: if content is empty after cleaning, and object contains other descriptive fields like 'title' or 'swipe'
    if not content:
        # attempt a few fallbacks
        if 'title' in obj and obj['title']:
            content = clean_text(obj['title'])
        elif 'swipe_info' in obj and isinstance(obj['swipe_info'], list) and len(obj['swipe_info'])>0:
            first = obj['swipe_info'][0]
            if isinstance(first, dict):
                content = clean_text(first.get('swipes', [''])[0] if first.get('swipes') else first.get('mes', ''))
        # else content remains empty string

    if content is None:
        content = ""

    return {"role": role, "character": character, "content": content}

# -------------------------
# Main conversion logic
# -------------------------
def parse_file_to_conversation(path: str, roles: List[str], verbose: bool=False) -> Dict[str, Any]:
    """
    Read various JSON/JSONL formats and produce a single conversation dict.
    """
    items = []
    text = None
    with open(path, 'r', encoding='utf-8') as f:
        try:
            text = f.read()
        except Exception as e:
            if verbose:
                print(f"[WARN] reading {path}: {e}")
            return {}

    # try parse as whole JSON first
    def try_load_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    parsed = try_load_json(text)
    if parsed is None:
        # try JSONL: parse line by line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        parsed_lines = []
        for ln in lines:
            p = try_load_json(ln)
            if p is not None:
                parsed_lines.append(p)
            else:
                # sometimes the file contains stray plain text lines; ignore or wrap
                # we skip non-json lines
                if verbose:
                    print(f"[DEBUG] skipping non-json line in {path}: {ln[:80]}...")
        # treat parsed_lines as sequence of objects
        # if only one object per line, we'll process those as messages
        if parsed_lines:
            raw_items = parsed_lines
        else:
            # last resort: treat whole file as text -> single system entry
            raw_items = [{"role":"system","content":text}]
    else:
        # parsed is valid JSON
        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict):
            # check for ChatGPT-like 'messages' key
            if 'messages' in parsed and isinstance(parsed['messages'], list):
                raw_items = parsed['messages']
            # sometimes conversation stored under 'conversation' key
            elif 'conversation' in parsed and isinstance(parsed['conversation'], list):
                raw_items = parsed['conversation']
            else:
                # If it's an object that represents one message (has name/mes fields) -> treat as single item
                # But sometimes file contains many top-level objects (we already tried JSONL); so just wrap
                raw_items = [parsed]
        else:
            raw_items = [{"role":"system","content":text}]

    # Convert raw_items to normalized messages
    conv = []
    prev_content = None
    for idx, obj in enumerate(raw_items):
        if not isinstance(obj, dict):
            # skip non-dict entries
            if verbose:
                print(f"[DEBUG] skipping non-dict item in {path} at idx {idx}")
            continue
        norm = extract_content_from_obj(obj, roles)
        # skip empty content blocks (unless we want to keep system prompts)
        if not norm['content']:
            # if system and content empty, skip
            if norm['role'] == 'system':
                continue
            # else keep as empty for context? skip to avoid noise
            continue

        # avoid duplicates (e.g., repeated swipe copy)
        if prev_content and norm['content'] == prev_content:
            if verbose:
                print(f"[DEBUG] dedup content in {path} at idx {idx}")
            # still advance but skip adding duplicate
            prev_content = norm['content']
            continue

        conv.append(norm)
        prev_content = norm['content']

    out = {
        "source": os.path.basename(path),
        "conversation": conv
    }
    return out

def batch_convert(input_dir: str, output_dir: str, roles: List[str], verbose: bool=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, "*")))
    if not files:
        print(f"[WARN] No files found in {input_dir}")
        return

    for fpath in files:
        if os.path.isdir(fpath):
            continue
        try:
            conv = parse_file_to_conversation(fpath, roles, verbose=verbose)
            if not conv or not conv.get('conversation'):
                if verbose:
                    print(f"[INFO] No conversation extracted from {fpath}, skipping.")
                continue
            base = os.path.splitext(os.path.basename(fpath))[0]
            out_path = os.path.join(output_dir, base + ".converted.json")
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(conv, out_f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"[OK] Converted {fpath} -> {out_path} ({len(conv['conversation'])} turns)")
        except Exception as e:
            print(f"[ERROR] Failed to convert {fpath}: {e}")

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Convert multi-character chat files into roma_qlo_ra_multirole 'input_conversations' format.")
    p.add_argument("--input_dir", required=True, help="Directory containing raw chat files (JSONL/JSON/ChatGPT-style).")
    p.add_argument("--output_dir", required=True, help="Directory to write converted files (will be created).")
    p.add_argument("--roles_file", required=False, default=None, help="Optional file with role names (one per line). If omitted uses built-in list.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    roles = load_roles(args.roles_file)
    if args.verbose:
        print(f"[INFO] Loaded {len(roles)} roles.")
    batch_convert(args.input_dir, args.output_dir, roles, verbose=args.verbose)
    if args.verbose:
        print("[DONE] batch conversion finished.")

if __name__ == "__main__":
    main()
