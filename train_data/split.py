import os
import json
import re
import shutil
from datetime import datetime

def parse_dialogue_text(dialogue_text):
    """解析对话文本，提取每个角色的发言"""
    # 移除对话历史标记
    if "### 对话历史" in dialogue_text:
        dialogue_text = dialogue_text.split("### 对话历史")[1].strip()
    if "---" in dialogue_text:
        dialogue_text = dialogue_text.split("---")[1].strip()
    
    # 按行分割
    lines = dialogue_text.split('\n')
    
    messages = []
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 匹配角色发言格式：**角色名**: 或 角色名: （有时有括号描述）
        speaker_match = re.match(r'\*\*([^*]+)\*\*:\s*(.*)', line)
        if not speaker_match:
            speaker_match = re.match(r'([^:]+):\s*(.*)', line)
        
        if speaker_match:
            # 保存上一条消息
            if current_speaker and current_content:
                messages.append({
                    "role": current_speaker,
                    "content": '\n'.join(current_content).strip()
                })
            
            # 开始新消息
            current_speaker = speaker_match.group(1).strip()
            current_content = [speaker_match.group(2).strip()]
        else:
            # 继续当前消息的内容
            if current_speaker and line:
                current_content.append(line)
    
    # 添加最后一条消息
    if current_speaker and current_content:
        messages.append({
            "role": current_speaker,
            "content": '\n'.join(current_content).strip()
        })
    
    return messages

def process_jsonl_file(file_path):
    """处理单个JSONL文件，将对话拆分成多行"""
    print(f"处理文件: {os.path.basename(file_path)}")
    
    # 备份原文件
    backup_dir = "backup"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
    shutil.copy2(file_path, backup_path)
    print(f"  备份文件: {backup_path}")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except:
            print(f"  第{line_num}行: JSON解析失败")
            new_lines.append(line)
            continue
        
        # 检查是否有messages字段
        if 'messages' not in data:
            print(f"  第{line_num}行: 没有messages字段")
            new_lines.append(line)
            continue
        
        # 提取系统消息
        system_message = None
        user_message = None
        
        for msg in data['messages']:
            if msg.get('role') == 'system':
                system_message = msg
            elif msg.get('role') == 'user':
                user_message = msg
        
        if not user_message:
            print(f"  第{line_num}行: 没有用户消息")
            new_lines.append(line)
            continue
        
        # 解析对话文本
        dialogue_text = user_message.get('content', '')
        character_messages = parse_dialogue_text(dialogue_text)
        
        print(f"  解析出 {len(character_messages)} 条角色发言")
        
        # 为每条角色发言创建新的对话
        for i, char_msg in enumerate(character_messages):
            new_data = {
                "messages": []
            }
            
            # 添加系统消息（如果有）
            if system_message:
                new_data["messages"].append(system_message)
            
            # 添加当前角色发言作为用户消息
            new_data["messages"].append({
                "role": "user",
                "content": f"扮演{char_msg['role']}，请说：{char_msg['content']}"
            })
            
            # 添加assistant角色作为回复
            new_data["messages"].append({
                "role": "assistant", 
                "content": char_msg['content']
            })
            
            # 添加元信息
            new_data['original_index'] = line_num
            new_data['character_index'] = i
            new_data['character_name'] = char_msg['role']
            new_data['total_characters'] = len(character_messages)
            
            new_lines.append(json.dumps(new_data, ensure_ascii=False))
    
    # 写入新文件（覆盖原文件）
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')
    
    print(f"  完成: {len(lines)}行 -> {len(new_lines)}行")
    return True

def process_all_jsonl_files(directory="."):
    """处理当前目录下所有JSONL文件"""
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print("没有找到JSONL文件!")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件:")
    for file in jsonl_files:
        print(f"  - {file}")
    
    print("\n开始处理...")
    
    success_count = 0
    for file_name in jsonl_files:
        file_path = os.path.join(directory, file_name)
        try:
            success = process_jsonl_file(file_path)
            if success:
                success_count += 1
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(jsonl_files)} 个文件")

# 运行
if __name__ == "__main__":
    process_all_jsonl_files()