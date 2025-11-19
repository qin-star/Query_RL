#!/usr/bin/env python3
"""
测试 RAG API 连接和响应速度
独立脚本，不依赖 src 模块
"""

import sys
import time
import requests
from pathlib import Path

def test_rag_api():
    """测试 RAG API"""
    
    print("🔍 测试 RAG API 连接...")
    print("=" * 60)
    
    # 使用默认配置（不依赖配置文件）
    rag_url = 'http://localhost:7861'
    print(f"📍 RAG URL: {rag_url}")
    
    # 默认训练配置
    tenant_id = "chengla"
    contact_id = "Customer_knowledge_17"
    account_id = "Sale_knowledge_17"
    message_id = "chengla_query_rl_message_id"
    score_threshold = 0.9
    
    # 测试数据
    test_context = "这是一个测试上下文"
    test_thought_unit = "测试查询"
    
    # 测试 /rag/chat 端点 (32B 模型)
    print(f"\n📡 测试 RAG Chat (32B): {rag_url}/rag/chat")
    success_32b, time_32b = test_single_api(
        rag_url, 
        "/rag/chat",
        test_context,
        test_thought_unit,
        tenant_id,
        contact_id,
        account_id,
        message_id,
        score_threshold,
        "32B"
    )
    
    # 测试 /rag/chat_8b 端点 (8B 模型)
    print(f"\n📡 测试 RAG Chat 8B: {rag_url}/rag/chat_8b")
    success_8b, time_8b = test_single_api(
        rag_url,
        "/rag/chat_8b",
        test_context,
        test_thought_unit,
        tenant_id,
        contact_id,
        account_id,
        message_id,
        score_threshold,
        "8B"
    )
    
    # 总结
    print("\n" + "=" * 60)
    if success_8b and success_32b:
        print("✅ 所有 RAG API 都可用")
        print(f"📊 平均响应时间: {(time_8b + time_32b) / 2:.2f} 秒")
        print(f"💡 预计单个样本处理时间: {(time_8b + time_32b) * 5:.2f} 秒（5个候选）")
        return 0
    else:
        print("❌ 部分或全部 RAG API 不可用")
        print("\n建议:")
        print("1. 检查 RAG 服务是否启动")
        print(f"2. 检查配置文件中的 RAG_URL: {rag_url}")
        print("3. 检查网络连接和防火墙")
        return 1

def test_single_api(base_url, endpoint, context, thought_unit, 
                   tenant_id, contact_id, account_id, message_id,
                   score_threshold, model_name):
    """测试单个 API
    
    Args:
        base_url: RAG服务基础URL
        endpoint: API端点 (/rag/chat 或 /rag/chat_8b)
        context: 上下文内容
        thought_unit: 思考单元/查询内容
        tenant_id: 租户ID
        contact_id: 联系人ID
        account_id: 账户ID
        message_id: 消息ID
        score_threshold: 分数阈值
        model_name: 模型名称（用于显示）
    """
    
    try:
        # 构造请求体（匹配 rag_chater.py 的格式）
        payload = {
            "tenant_id": tenant_id,
            "contact_id": contact_id,
            "account_id": account_id,
            "message_id": message_id,
            "kb_name": tenant_id,  # kb_name 使用 tenant_id
            "thought_unit": thought_unit,
            "score_threshold": score_threshold,
            "context": context  # 使用单个context字符串
        }
        
        # 如果是8B端点，可以添加额外的可选字段
        if endpoint == "/rag/chat_8b":
            payload.update({
                "user_profile": "",
                "history_summary": "",
                "rewritten_query": ""
            })
        
        # 发送请求
        full_url = f"{base_url}{endpoint}"
        start_time = time.time()
        response = requests.post(
            full_url,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        # 检查响应
        if response.status_code == 200:
            print(f"  ✓ {model_name} 可用")
            print(f"  ⏱ 响应时间: {elapsed:.2f} 秒")
            
            # 显示响应内容
            try:
                data = response.json()
                # 根据实际响应格式显示信息
                if isinstance(data, list):
                    print(f"  📄 返回结果数: {len(data)}")
                elif isinstance(data, dict):
                    print(f"  📄 响应数据: {list(data.keys())}")
                else:
                    print(f"  📄 响应类型: {type(data).__name__}")
            except Exception as e:
                print(f"  ⚠ 解析响应失败: {e}")
            
            return True, elapsed
        else:
            print(f"  ✗ {model_name} 返回错误: {response.status_code}")
            print(f"  📄 错误信息: {response.text[:200]}")
            return False, 0
            
    except requests.exceptions.Timeout:
        print(f"  ✗ {model_name} 超时（>30秒）")
        return False, 0
    except requests.exceptions.ConnectionError:
        print(f"  ✗ {model_name} 连接失败（服务可能未启动）")
        return False, 0
    except Exception as e:
        print(f"  ✗ {model_name} 错误: {e}")
        return False, 0

if __name__ == "__main__":
    sys.exit(test_rag_api())
