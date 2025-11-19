"""测试模型生成是否正常"""
import sys
import os

# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_generation():
    """测试模型生成"""
    
    # 模型路径（从配置文件中获取）
    model_path = "/home/jovyan2/query_rl/output/qwen3-8b-lora-sft/v3-20251031-111238/checkpoint-159-merged"
    
    print("=" * 80)
    print("测试模型生成")
    print("=" * 80)
    print(f"\n模型路径: {model_path}")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误：模型路径不存在！")
        print(f"请检查配置文件中的模型路径")
        return
    
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 测试prompt（从数据集中提取的真实prompt）
    test_prompt = """## 背景
你是教培行业的对话理解与用户洞察专家，具备多年公考教培销售经验，擅长用户画像提取、上下文理解与问题改写。

## 当前任务
请基于以下信息，依次完成三个任务：
1. 提取用户画像（user_profile）
2. 提炼历史上下文摘要（history_summary）
3. 对当前用户输入进行 query 改写（rewritten_query）

## 输入信息
- 历史对话内容：
"[客户][2025-10-19 20:28:13]: 我已经添加了你，现在我们可以开始聊天了。
[销售][2025-10-19 20:28:19]: 同学你好❤，我是你的专属课程助教老师-陈老师
直播课是10月21号到24号每晚19：00，这几天由我来负责你本次的学习~"

- 用户当前输入：
"13927332890 广东省考"

## 输出要求
**重要：请直接输出JSON，不要使用<think>标签，不要输出思考过程！**
你必须返回一个 JSON 格式的对象，包含以下三个字段：
{
  "user_profile": "用户画像描述",
  "history_summary": "历史摘要",
  "rewritten_query": "改写后的查询"
}
"""
    
    print("\n" + "=" * 80)
    print("测试Prompt:")
    print("=" * 80)
    print(test_prompt[:500] + "...")
    
    # 应用chat template
    # 关键：使用enable_thinking=False禁用thinking模式
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 禁用thinking模式！
    )
    
    print("\n" + "=" * 80)
    print("应用chat template后:")
    print("=" * 80)
    print(text[:500] + "...")
    
    # 生成
    print("\n" + "=" * 80)
    print("开始生成...")
    print("=" * 80)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 禁用thinking模式，直接生成JSON
    # 方法1：使用stop_strings（如果tokenizer支持）
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,  # 降低temperature
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 尝试添加stop_strings来阻止think标签
    if hasattr(tokenizer, 'stop_strings'):
        generation_config["stop_strings"] = ["</think>", "<think>"]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的部分（去掉prompt）
    response = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    print("\n" + "=" * 80)
    print("生成结果:")
    print("=" * 80)
    print(response)
    
    # 检查是否是JSON格式
    import json
    try:
        parsed = json.loads(response)
        print("\n✅ JSON解析成功！")
        print(f"  - user_profile: {parsed.get('user_profile', 'N/A')[:100]}")
        print(f"  - history_summary: {parsed.get('history_summary', 'N/A')[:100]}")
        print(f"  - rewritten_query: {parsed.get('rewritten_query', 'N/A')[:100]}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON解析失败: {e}")
        print("生成的内容不是有效的JSON格式")
        print("\n这说明模型可能：")
        print("1. 没有正确训练")
        print("2. 使用了错误的checkpoint")
        print("3. SFT训练数据有问题")

if __name__ == "__main__":
    test_model_generation()
