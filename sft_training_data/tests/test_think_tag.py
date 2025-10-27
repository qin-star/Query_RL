"""
测试<think>标记是否正确添加
"""

import json

def test_think_tag():
    """测试think标记格式"""
    
    print("=" * 80)
    print("测试 <think> 标记添加")
    print("=" * 80)
    
    # 模拟转换过程
    user_profile = "用户为备考公务员的应届生"
    history_summary = "用户正在了解课程信息"
    rewritten_query = "公务员考试培训课程的价格是多少？"
    
    # 构建JSON输出
    assistant_output = {
        "user_profile": user_profile,
        "history_summary": history_summary,
        "rewritten_query": rewritten_query
    }
    
    json_output = json.dumps(assistant_output, ensure_ascii=False, indent=2)
    
    # 添加<think>标记
    assistant_content = f"<think>\n\n</think>\n\n{json_output}"
    
    print("\n【生成的Assistant Content】")
    print("-" * 80)
    print(assistant_content)
    
    # 验证格式
    print("\n【格式验证】")
    print("-" * 80)
    
    # 1. 检查是否以<think>开头
    assert assistant_content.startswith('<think>\n\n</think>\n\n'), "❌ 未正确添加<think>标记"
    print("✅ <think>标记位置正确")
    
    # 2. 提取JSON部分并解析
    json_str = assistant_content.replace('<think>\n\n</think>\n\n', '', 1)
    
    try:
        parsed = json.loads(json_str)
        print("✅ JSON格式正确")
    except:
        print("❌ JSON格式错误")
        return False
    
    # 3. 验证字段完整性
    assert 'user_profile' in parsed, "❌ 缺少user_profile字段"
    assert 'history_summary' in parsed, "❌ 缺少history_summary字段"
    assert 'rewritten_query' in parsed, "❌ 缺少rewritten_query字段"
    print("✅ 所有字段完整")
    
    # 4. 验证内容正确性
    assert parsed['user_profile'] == user_profile
    assert parsed['history_summary'] == history_summary
    assert parsed['rewritten_query'] == rewritten_query
    print("✅ 内容正确")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)
    
    # 展示完整的messages结构
    print("\n【完整的Messages结构示例】")
    print("-" * 80)
    
    sample_message = {
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的助手..."
            },
            {
                "role": "user",
                "content": "请分析以下对话..."
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    
    print(json.dumps(sample_message, ensure_ascii=False, indent=2))
    
    return True


def test_parsing_with_think_tag():
    """测试带<think>标记的解析"""
    
    print("\n" + "=" * 80)
    print("测试解析带<think>标记的内容")
    print("=" * 80)
    
    # 模拟从训练数据中读取
    assistant_content_with_think = """<think>

</think>

{
  "user_profile": "用户为备考公务员的应届生",
  "history_summary": "用户正在了解课程信息",
  "rewritten_query": "公务员考试培训课程的价格是多少？"
}"""
    
    print("\n【待解析内容】")
    print("-" * 80)
    print(assistant_content_with_think)
    
    # 解析方法1：直接替换
    if assistant_content_with_think.startswith('<think>\n\n</think>\n\n'):
        json_str = assistant_content_with_think.replace('<think>\n\n</think>\n\n', '', 1)
    else:
        json_str = assistant_content_with_think
    
    print("\n【提取的JSON】")
    print("-" * 80)
    print(json_str)
    
    try:
        parsed = json.loads(json_str)
        print("\n【解析结果】")
        print("-" * 80)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("\n✅ 解析成功")
        return True
    except Exception as e:
        print(f"\n❌ 解析失败: {e}")
        return False


if __name__ == "__main__":
    success = test_think_tag()
    
    if success:
        test_parsing_with_think_tag()

