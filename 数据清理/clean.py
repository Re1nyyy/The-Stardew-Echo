import json
import os
import re
from typing import List, Dict, Any

WORLD_BOOK_PATH = 'book/worldbook.json'
CHARACTER_BOOK_PATH = 'book/characterbook.json'
INPUT_DIR = 'input_conversations' 
OUTPUT_DIR = 'train_data' 

SYSTEM_PROMPT = "你是一个能够进行多角色扮演的AI助手。你的目标是扮演星露谷中的一个指定角色，根据给定的知识、对话历史，以高保真的角色口吻、情绪和立场进行回复。请严格遵守世界观中的交流准则。"


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
                    mes = data['swipes'][0]

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
# 3. RAG Context 构建函数 (与之前相同)
# --------------------------

def build_rag_context(worldbook_data: Dict, characterbook_data: Dict, active_names: List[str]) -> str:
    """根据世界观和当前活跃角色，动态构建 RAG Context 模板。"""
    
    context_parts = []
    
    # --- 1. 提取世界观和行为准则 (UID 0) ---
    world_entries = worldbook_data.get('entries', {})
    if '0' in world_entries:
        entry_0 = world_entries['0']
        if isinstance(entry_0, dict):
            world_content = entry_0.get('content', '').strip()
            # 为 ChatML 格式调整 header
            context_parts.append(f"### 世界观与交流准则\n{world_content}\n")
    
    # --- 2. 提取相关角色档案 ---
    char_entries = characterbook_data.get('entries', {})
    if char_entries:
        context_parts.append("### 核心角色档案")
        
        for uid, entry in char_entries.items():
            if not isinstance(entry, dict):
                continue
            
            # 使用 key 字段中的角色名来判断是否活跃
            is_relevant = any(name in active_names for name in entry.get('key', []))
            
            if is_relevant:
                char_content = entry.get('content', '').strip()
                # 尝试获取第一个 key 作为角色名，如果没有则用 UID 代替
                char_name = entry.get('key', [f'未知角色 (UID {uid})'])[0] 
                context_parts.append(f"* {char_name} (UID {uid}):\n{char_content}\n")
        
        context_parts.append("\n") 
        
    return "\n".join(context_parts)

# --------------------------
# 4. 格式化和清洗函数 (ChatML Format) (与之前相同)
# --------------------------

def clean_message(text: str) -> str:
    """清理消息中的特殊符号和多余换行符"""
    # 移除消息开头/结尾的空白，并将内部多个换行/空白压缩成一个换行
    text = re.sub(r'\s*\n\s*', '\n', text).strip()
    return text

def format_to_chatml_jsonl(raw_messages: List[Dict], rag_context: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """将原始群聊数据格式化为 ChatML (messages) 所需的 JSONL 格式"""
    training_samples = []
    history = ""
    
    if len(raw_messages) < 2:
        return []
    
    # 构造 SYSTEM 角色的内容 (RAG 知识注入到 System)
    system_content = f"{SYSTEM_PROMPT}\n\n### 检索到的长期记忆\n---\n{rag_context}"
    
    for i in range(1, len(raw_messages)):
        
        # 1. 将前一条消息加入历史
        previous_message = raw_messages[i-1]
        history += f"**{previous_message['name']}**: {clean_message(previous_message['mes'])}\n"
        
        # 2. 构造当前回合的 User/Assistant 内容
        current_message = raw_messages[i]
        assistant_name = current_message["name"]
        assistant_content = current_message["mes"]
        
        # USER 字段内容 (历史 + Call to Action)
        user_content = f"""
### 对话历史
---
{history}

### 你的回合
---
现在请你扮演 **{assistant_name}** 回复。
"""
        
        # 3. 构造 ChatML 消息数组
        chatml_sample = {
            "messages": [
                {"role": "system", "content": clean_message(system_content)},
                {"role": "user", "content": clean_message(user_content)},
                {"role": "assistant", "content": clean_message(assistant_content)}
            ]
        }
        
        training_samples.append(chatml_sample)
        
    return training_samples

# --------------------------
# 5. 单个文件处理函数
# --------------------------

def process_single_conversation_file(
    input_filepath: str, 
    output_filepath: str, 
    world_data: Dict, 
    char_data: Dict
):
    """处理单个对话文件，生成 ChatML 格式数据并保存"""
    
    print(f"\n--- 正在处理文件: {os.path.basename(input_filepath)} ---")
    
    raw_messages = load_conversation_data(input_filepath)

    if not raw_messages:
        print(f"❌ 文件 {os.path.basename(input_filepath)} 中未加载到有效的对话记录，跳过。")
        return
    
    active_names = list(set([msg["name"] for msg in raw_messages]))
    
    # 构建 RAG Context
    rag_context = build_rag_context(world_data, char_data, active_names)
    print(f"已识别 {len(active_names)} 个活跃角色。")

    # 执行 ChatML 格式化
    jsonl_data = format_to_chatml_jsonl(raw_messages, rag_context)

    if not jsonl_data:
        print(f"❌ 文件 {os.path.basename(input_filepath)} 无法生成训练样本，跳过。")
        return

    # 保存到 JSONL 文件
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for sample in jsonl_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ 成功将 {len(jsonl_data)} 个训练样本保存到 {output_filepath}")
    except Exception as e:
        print(f"保存文件 {output_filepath} 时发生错误: {e}")


# --------------------------
# 6. 主执行逻辑 (迭代处理文件夹)
# --------------------------

def main():
    print(f"启动 ChatML 数据处理工具...")
    
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出文件夹已就绪: {OUTPUT_DIR}")

    # 2. 检查输入目录
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ 错误: 输入文件夹 '{INPUT_DIR}' 不存在。请创建该文件夹并将对话文件放入其中。")
        return
    
    # 3. 加载 RAG 知识文件 (只需加载一次)
    print(f"正在加载 RAG 知识文件...")
    world_data = load_json_files(WORLD_BOOK_PATH)
    char_data = load_json_files(CHARACTER_BOOK_PATH)
    if not (world_data and char_data):
        print("RAG 知识文件加载不完整，后续训练样本的 Context 可能会缺失。")
    
    # 4. 遍历输入目录中的文件
    processed_count = 0
    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.json', '.jsonl'))]
    
    if not file_list:
        print(f"警告: 文件夹 '{INPUT_DIR}' 中未找到任何 .json 或 .jsonl 文件。")
        return

    print(f"找到 {len(file_list)} 个待处理的对话文件。")
    
    for filename in file_list:
        input_filepath = os.path.join(INPUT_DIR, filename)
        
        # 构造输出文件名 (使用相同的名字，确保后缀是 .jsonl)
        name_part, _ = os.path.splitext(filename)
        output_filepath = os.path.join(OUTPUT_DIR, f"{name_part}.jsonl")
        
        # 调用单个文件处理函数
        process_single_conversation_file(
            input_filepath, 
            output_filepath, 
            world_data, 
            char_data
        )
        processed_count += 1

    print("\n" + "=" * 50)
    print(f"所有文件处理完成。总共处理了 {processed_count} 个文件。")
    print(f"所有生成的 ChatML 数据都在 '{OUTPUT_DIR}' 文件夹中。")
    print("=" * 50)


if __name__ == "__main__":
    main()