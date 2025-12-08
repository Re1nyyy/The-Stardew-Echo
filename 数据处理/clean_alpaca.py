import json
import os
import re
from typing import List, Dict, Any

# --- 1. 路径和配置 (与您之前设置的相同) ---
WORLD_BOOK_PATH = 'book\worldbook.json'
CHARACTER_BOOK_PATH = 'book\characterbook.json'
INPUT_DIR = 'input_conversations' 
OUTPUT_DIR = 'train_data_alpaca'

# 核心配置：玩家/用户的名称标识
PLAYER_NAME = "team" 

# 通用系统提示词 (用于所有角色，结合RAG知识注入)
SYSTEM_PROMPT_TEMPLATE = (
    "你是一个能够进行多角色扮演的AI助手。你的目标是扮演星露谷中的一个指定角色，根据给定的知识、对话历史，以高保真的角色口吻、情绪和立场进行回复。请严格遵守世界观中的交流准则。\n"
    "--- 角色档案和世界观已注入 SYSTEM 字段，请参阅下方 ---"
)

# --------------------------
# 2. 数据加载函数 (保持不变)
# --------------------------

def load_json_files(filepath: str) -> Dict[str, Any]:
    """安全地读取普通 JSON 文件内容 (用于 worldbook/characterbook)"""
    if not os.path.exists(filepath):
        print(f"警告: RAG 知识文件未找到 - {filepath}。将使用空数据。")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON 文件 - {filepath}。请检查文件格式。错误信息: {e}")
        return {}

def load_conversation_data(filepath: str) -> List[Dict[str, str]]:
    """读取 JSON Lines 文件内容，并提取 name 和 mes 字段"""
    messages = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                name = data.get('name')
                mes = data.get('mes')

                if not mes and data.get('swipes') and isinstance(data['swipes'], list) and data['swipes']:
                    mes = data['swipes'][0] # 使用第一个 swipe 作为 mes

                if data.get('is_system', False) or name in ["system", "System"]:
                    continue

                if name and mes:
                    messages.append({"name": name, "mes": mes})

    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON Lines 文件 - {filepath}。请检查文件格式。错误信息: {e}")
        return []
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return []
    
    return messages

# --------------------------
# 3. RAG Context 构建函数 (保持不变)
# --------------------------

def build_rag_context(worldbook_data: Dict, characterbook_data: Dict, active_names: List[str], target_char: str) -> str:
    """
    根据世界观和当前活跃角色，动态构建 RAG Context 模板。
    处理 SillyTavern Characterbook 结构。
    """
    
    context_parts = []
    all_char_entries = {}
    
    char_entries = characterbook_data.get('entries', {})
    if not char_entries:
        char_entries = worldbook_data.get('entries', {})

    if char_entries:
        for uid, entry in char_entries.items():
            if not isinstance(entry, dict):
                continue
            
            keys = entry.get('key', [])
            raw_content = entry.get('content', '').strip()
            
            if not keys or not raw_content:
                continue

            char_name = keys[0] 
            cleaned_content = raw_content.replace('```yaml', '').replace('```', '').strip()
            
            all_char_entries[char_name] = {
                'content': cleaned_content,
                'is_active': any(name in active_names for name in keys)
            }
        
    context_parts.append("### 核心角色档案 (长期记忆)")
    
    # a. 目标角色档案 (放在最前面)
    if target_char in all_char_entries:
        target_entry = all_char_entries[target_char]['content']
        context_parts.append(f"【你扮演的角色：{target_char}】\n{target_entry}\n")
        del all_char_entries[target_char] 
    else:
        context_parts.append(f"警告：找不到目标角色 [{target_char}] 的详细档案。")

    # b. 活跃角色的档案
    active_other_chars = sorted([name for name, entry in all_char_entries.items() if entry['is_active']])
    
    for name in active_other_chars:
        entry = all_char_entries[name]
        context_parts.append(f"--- 活跃角色：{name} ---\n{entry['content']}\n")
                 
    context_parts.append("\n") 
    
    return "\n".join(context_parts)

# --------------------------
# 4. 清洗和 Alpaca 格式化函数 (保持不变)
# --------------------------

def clean_message(text: str) -> str:
    """清理消息中的特殊符号和多余换行符"""
    text = re.sub(r'\s*\n\s*', '\n', text).strip()
    return text

def format_history_alpaca(buffer: List[Dict], end_index: int) -> List[List[str]]:
    """
    将缓冲中的对话转换为 Alpaca history 格式: [["指令", "回答"], ...]
    """
    history_list = []
    
    for i in range(0, end_index, 2):
        instruction_turn = buffer[i] if i < len(buffer) else None
        response_turn = buffer[i+1] if i+1 < len(buffer) else None
        
        if instruction_turn and response_turn:
            instruction = f"[{instruction_turn['name']}]: {clean_message(instruction_turn['mes'])}"
            response = f"[{response_turn['name']}]: {clean_message(response_turn['mes'])}"
            history_list.append([instruction, response])
            
    return history_list

def format_to_alpaca_jsonl(raw_messages: List[Dict], rag_context: str, target_char: str) -> List[Dict[str, Any]]:
    """将原始群聊数据格式化为 LlamaFactory Alpaca (SFT) 格式"""
    training_samples = []
    
    if len(raw_messages) < 2:
        return []
    
    final_system_content = f"{SYSTEM_PROMPT_TEMPLATE}\n\n### 检索到的长期记忆\n---\n{rag_context}"
    conversation_buffer = [] 
    
    for i, current_message in enumerate(raw_messages):
        speaker_name = current_message["name"]
        message_content = current_message["mes"]
        
        # 找到目标角色回复，即生成一个训练样本
        if speaker_name == target_char and len(conversation_buffer) > 0:
            
            # --- 1. 提取 Output ---
            output_content = clean_message(message_content)
            
            # --- 2. 提取 Instruction/Input ---
            # 目标回复的前一条消息是 prompt
            prompt_turn = conversation_buffer[-1] 
            prompt_name = prompt_turn['name']
            
            # Instruction: Call to Action，要求模型扮演目标角色回复
            instruction = f"请以 [{target_char}] 的身份回复以下消息。"
            # Input: 实际的 prompt 内容（说话人: 消息）
            input_data = f"[{prompt_name}]: {clean_message(prompt_turn['mes'])}"
            
            # --- 3. 提取 History ---
            # History 是 prompt_turn 之前的所有消息
            history_list = format_history_alpaca(conversation_buffer, len(conversation_buffer) - 1)
            
            # --- 4. 构建 Alpaca 训练样本 ---
            alpaca_sample = {
                "instruction": clean_message(instruction),
                "input": input_data,
                "output": output_content,
                "system": clean_message(final_system_content),
                "history": history_list
            }
            
            training_samples.append(alpaca_sample)
            
            # --- 5. 更新缓冲 ---
            # 将目标回复加入缓冲，作为下一轮历史的起点
            conversation_buffer.append(current_message)
        
        else:
            # 非目标角色发言，加入缓冲
            conversation_buffer.append(current_message)
            
        # 清洗：对话太长时进行截断（防止 history 过长）
        if len(conversation_buffer) > 20: 
             conversation_buffer = conversation_buffer[-10:] 
            
    return training_samples

# --------------------------
# 5. 单个文件处理函数
# --------------------------

def process_single_conversation_file(
    input_filepath: str, 
    output_filepath_template: str, # 接收模板字符串
    world_data: Dict, 
    char_data: Dict,
    target_char: str
):
    """处理单个对话文件，生成 Alpaca 格式数据并保存"""
    
    # 动态生成特定角色的输出文件名
    output_filepath = output_filepath_template.format(target_char=target_char)
    
    # 检查是否已存在该角色的数据，避免重复处理
    if os.path.exists(output_filepath):
        # 警告：这里简化处理，如果文件已存在则追加数据
        print(f"警告：文件 {output_filepath} 已存在，新样本将追加到文件末尾。")
        mode = 'a'
    else:
        mode = 'w'
    
    raw_messages = load_conversation_data(input_filepath)

    if not raw_messages:
        # print(f"❌ 文件 {os.path.basename(input_filepath)} 中未加载到有效的对话记录，跳过。")
        return
    
    active_names = list(set([msg["name"] for msg in raw_messages]))
    
    rag_context = build_rag_context(world_data, char_data, active_names, target_char)

    jsonl_data = format_to_alpaca_jsonl(raw_messages, rag_context, target_char)

    if not jsonl_data:
        # print(f"❌ 文件 {os.path.basename(input_filepath)} 无法生成 {target_char} 的训练样本，跳过。")
        return

    try:
        with open(output_filepath, mode, encoding='utf-8') as f:
            for sample in jsonl_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        return len(jsonl_data)
    except Exception as e:
        print(f"保存文件 {output_filepath} 时发生错误: {e}")
        return 0


# --------------------------
# 6. 主执行逻辑 (自动遍历所有角色)
# --------------------------

def main():
    print(f"启动 Alpaca SFT 数据处理工具 (多角色自动遍历模式)...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载 RAG 知识文件
    world_data = load_json_files(WORLD_BOOK_PATH)
    char_data = load_json_files(CHARACTER_BOOK_PATH)
    
    # 2. 从角色书解析所有需要训练的角色名称
    all_target_chars = []
    char_entries = char_data.get('entries', {})
    for entry in char_entries.values():
        if entry.get('key'):
            # 使用第一个 key 作为角色的标准名称
            all_target_chars.append(entry['key'][0])
            
    if not all_target_chars:
        print("❌ 错误: 无法从 characterbook.json 中解析出任何角色名称。请检查文件格式。")
        return

    print(f"成功识别 {len(all_target_chars)} 个目标训练角色: {', '.join(all_target_chars)}")
    
    # 3. 准备输入文件列表
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ 错误: 输入文件夹 '{INPUT_DIR}' 不存在。")
        return
        
    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.json', '.jsonl'))]
    
    if not file_list:
        print(f"警告: 文件夹 '{INPUT_DIR}' 中未找到任何 .json 或 .jsonl 文件。")
        return
        
    # 4. 循环：为每个目标角色处理所有对话文件
    total_samples_generated = 0
    
    for target_char in all_target_chars:
        print("\n" + "=" * 50)
        print(f"🚀 开始为角色：【{target_char}】生成数据...")
        
        char_sample_count = 0
        
        # 构造输出文件名模板 (使用 {target_char} 占位符)
        output_name = f"stardew_alpaca_{target_char}.jsonl"
        output_filepath_template = os.path.join(OUTPUT_DIR, output_name)
        
        # 确保输出文件是空的（或者移除，这里我们默认覆盖/创建新文件，以防历史记录混乱）
        if os.path.exists(output_filepath_template):
             os.remove(output_filepath_template)
             print(f"已清除旧文件：{output_name}")


        for filename in file_list:
            input_filepath = os.path.join(INPUT_DIR, filename)
            
            # 调用单个文件处理函数
            samples = process_single_conversation_file(
                input_filepath, 
                output_filepath_template, # 传递模板
                world_data, 
                char_data,
                target_char 
            )
            char_sample_count += samples

        print(f"✅ 角色 【{target_char}】 数据生成完成。共生成 {char_sample_count} 个样本。")
        total_samples_generated += char_sample_count


    print("\n" + "=" * 50)
    print(f"🎉 所有 {len(all_target_chars)} 个角色的数据处理完成。总共生成了 {total_samples_generated} 个训练样本。")
    print(f"数据文件位于 '{OUTPUT_DIR}' 文件夹中。")
    print("=" * 50)


if __name__ == "__main__":
    main()