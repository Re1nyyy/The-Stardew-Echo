import json
import os
import glob
from typing import List, Dict, Any
import re

def is_topic_change(messages: List[Dict], current_idx: int) -> bool:
    """判断是否发生话题转换"""
    if current_idx < 2:
        return False
    
    # 获取当前消息和前两条消息的内容
    current_content = messages[current_idx]["content"] if current_idx < len(messages) else ""
    prev1_content = messages[current_idx-1]["content"] if current_idx-1 >= 0 else ""
    prev2_content = messages[current_idx-2]["content"] if current_idx-2 >= 0 else ""
    
    # 话题转换的迹象：
    # 1. 消息中包含明确的话题转换词汇
    topic_change_keywords = [
        "对了，", "顺便问一下，", "话说回来，", "换个话题，",
        "那么，", "接下来，", "另外，", "还有，",
        "By the way", "Anyway", "Speaking of", "On another note"
    ]
    
    for keyword in topic_change_keywords:
        if keyword in current_content:
            return True
    
    # 2. 消息长度突然变短（可能是一个新话题的开始）
    if len(current_content) < 30 and len(prev1_content) > 50:
        return True
    
    # 3. 检测时间/季节/地点的变化
    time_keywords = ["早上", "中午", "晚上", "明天", "后天", "下周", "春天", "夏天", "秋天", "冬天"]
    location_keywords = ["农场", "杂货店", "餐吧", "矿洞", "海滩", "森林", "山区"]
    
    current_has_time = any(keyword in current_content for keyword in time_keywords)
    prev_has_time = any(keyword in prev1_content for keyword in time_keywords)
    
    current_has_location = any(keyword in current_content for keyword in location_keywords)
    prev_has_location = any(keyword in prev1_content for keyword in location_keywords)
    
    # 如果时间或地点发生明显变化，可能话题也变了
    if (current_has_time and not prev_has_time) or (current_has_location and not prev_has_location):
        return True
    
    return False

def split_long_dialogue(messages: List[Dict], max_rounds: int = 8) -> List[List[Dict]]:
    """将长对话切割成多个短对话"""
    chunks = []
    current_chunk = []
    
    for i, message in enumerate(messages):
        current_chunk.append(message)
        
        # 每max_rounds轮切割，或在话题转换处切割
        if (len(current_chunk) >= max_rounds or is_topic_change(messages, i)):
            # 确保chunk有始有终（至少3轮）
            if len(current_chunk) >= 3:
                # 检查最后一个消息是否完整（不是以冒号或未完成句子结束）
                last_content = current_chunk[-1]["content"]
                if last_content and not last_content.strip().endswith(('.', '!', '?', '。', '！', '？')):
                    # 寻找下一个合适的结束点
                    for j in range(i+1, min(i+3, len(messages))):
                        next_content = messages[j]["content"]
                        if next_content and next_content.strip().endswith(('.', '!', '?', '。', '！', '？')):
                            current_chunk.append(messages[j])
                            i = j  # 更新索引
                            break
                
                # 检查当前chunk是否以用户消息开始（如果可能）
                if current_chunk and current_chunk[0]["role"] != "user":
                    # 尝试找到最近的用户消息作为开始
                    for k in range(1, len(current_chunk)):
                        if current_chunk[k]["role"] == "user":
                            chunks.append(current_chunk[k:])
                            current_chunk = current_chunk[:k]
                            break
                
                if len(current_chunk) >= 3:
                    chunks.append(current_chunk)
                current_chunk = []
    
    # 处理剩余的消息
    if len(current_chunk) >= 3:
        # 如果剩余消息较少，尝试合并到上一个chunk
        if chunks and len(current_chunk) < 5 and len(chunks[-1]) + len(current_chunk) <= max_rounds + 2:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)
    
    return chunks

def process_jsonl_file(file_path: str, max_rounds: int = 8) -> None:
    """处理单个JSONL文件"""
    print(f"处理文件: {file_path}")
    
    output_lines = []
    total_chunks = 0
    total_dialogues = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            if "messages" not in data:
                output_lines.append(line.strip())
                continue
            
            messages = data["messages"]
            
            # 跳过太短的对话
            if len(messages) <= max_rounds:
                output_lines.append(line.strip())
                total_dialogues += 1
                continue
            
            # 切割长对话
            chunks = split_long_dialogue(messages, max_rounds)
            
            for chunk in chunks:
                if len(chunk) >= 3:  # 只保留有意义的对话片段
                    new_data = data.copy()
                    new_data["messages"] = chunk
                    output_lines.append(json.dumps(new_data, ensure_ascii=False))
                    total_chunks += 1
            
            total_dialogues += 1
            print(f"  第{line_num}行: {len(messages)}轮 -> {len(chunks)}个片段")
            
        except json.JSONDecodeError as e:
            print(f"  警告: 第{line_num}行JSON解析错误: {e}")
            output_lines.append(line.strip())
        except Exception as e:
            print(f"  错误: 第{line_num}行处理异常: {e}")
            output_lines.append(line.strip())
    
    # 直接删除
    if os.path.exists(file_path):
        os.remove(file_path)  # 删除单个文件
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for output_line in output_lines:
            f.write(output_line + '\n')
    
    print(f"  完成: {total_dialogues}个对话 -> {total_chunks}个片段")
    print(f"  原文件备份为: {backup_path}")

def main():
    # 获取当前目录下所有jsonl文件
    jsonl_files = glob.glob("*.jsonl")
    
    # 排除Python脚本文件
    python_files = glob.glob("*.py")
    jsonl_files = [f for f in jsonl_files if f not in python_files]
    
    if not jsonl_files:
        print("未找到JSONL文件！")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件:")
    for file in jsonl_files:
        print(f"  - {file}")
    
    # 设置最大对话轮次（可以根据需要调整）
    max_rounds = 8
    
    print("\n开始处理...")
    for file_path in jsonl_files:
        try:
            process_jsonl_file(file_path, max_rounds)
            print("-" * 50)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main()