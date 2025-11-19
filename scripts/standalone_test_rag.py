#!/usr/bin/env python3
"""
独立的RAG API测试脚本
不依赖任何项目模块，可以直接运行
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime

# ==================== 配置区域 ====================
RAG_BASE_URL = "http://localhost:7861"
TENANT_ID = "chengla"
CONTACT_ID = "Customer_knowledge_17"
ACCOUNT_ID = "Sale_knowledge_17"
SCORE_THRESHOLD = 0.9
TOP_K = 3

# 测试用的上下文
TEST_CONTEXT = """[客户][2025-10-10 00:01:39]: 我已经添加了你，现在我们可以开始聊天了。
[销售][2025-10-10 00:01:54]: 同学你好❤，我是你的专属课程助教老师-史老师
直播课是10月11号-14号每晚19：00，这几天就由我来负责你本次的学习安排以及资料发放~
📚课前必做
1.激活直播课👉https://cl2.cn/Buaib6u6（点击右下角免费领取--显示已报名去学习即为预约成功）
2.学习档案👉weixin://dl/business/?t=QpgfMaqYHDd（便于奇函老师了解同学目前学习情况）
2个链接完成之后请同学说一下【报名手机号+目标考试】，老师会根据档案信息单独给你制定专属学习规划~"""

# 测试用的增强字段
TEST_USER_PROFILE = "用户为应届毕业生，本科在读，计划以应届生身份参加2025年内蒙古省考，目标考试类型包括公务员和事业单位联考，当前处于备考初期阶段，关注考试规划、课程内容及备考方法，此前备考经验不足，成绩不理想。"
TEST_HISTORY_SUMMARY = "用户参加了2024年下半年事业单位联考，成绩不理想（48分），未参加国考，计划以应届生身份参加2025年内蒙古省考，并同步备考事业编考试，销售老师推荐了杨奇涵老师的课程，强调系统规划和高效备考的重要性。"
TEST_REWRITTEN_QUERY = "事业单位联考每年是否举行两次？"

# ==================== 工具函数 ====================

def print_header(title: str):
    """打印标题"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")


def print_result(success: bool, message: str, details: dict = None):
    """打印结果"""
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")
    if details:
        for key, value in details.items():
            print(f"  - {key}: {value}")


async def call_rag_endpoint(endpoint: str, payload: dict) -> tuple:
    """
    调用RAG端点
    
    Returns:
        (success, data, error_msg, cost_time)
    """
    url = f"{RAG_BASE_URL}{endpoint}"
    start_time = datetime.now()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
        cost_time = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            return True, response.json(), None, cost_time
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            return False, None, error_msg, cost_time
            
    except httpx.TimeoutException:
        cost_time = (datetime.now() - start_time).total_seconds()
        return False, None, "请求超时", cost_time
    except httpx.ConnectError:
        cost_time = (datetime.now() - start_time).total_seconds()
        return False, None, "连接失败（服务可能未启动）", cost_time
    except Exception as e:
        cost_time = (datetime.now() - start_time).total_seconds()
        return False, None, f"{type(e).__name__}: {str(e)}", cost_time


# ==================== 测试函数 ====================

async def test_32b_endpoint():
    """测试32B端点"""
    print_header("测试 /rag/chat (32B模型)")
    
    payload = {
        "tenant_id": TENANT_ID,
        "contact_id": CONTACT_ID,
        "account_id": ACCOUNT_ID,
        "thought_unit": "",
        "score_threshold": SCORE_THRESHOLD,
        "kb_name": TENANT_ID,
        "top_k": TOP_K,
        "context": TEST_CONTEXT
    }
    
    print(f"📦 Payload: tenant_id={TENANT_ID}, context长度={len(TEST_CONTEXT)}")
    print(f"🔄 调用中...")
    
    success, data, error, cost_time = await call_rag_endpoint("/rag/chat", payload)
    
    if success:
        result_count = len(data) if isinstance(data, list) else "N/A"
        print_result(True, "32B端点调用成功", {
            "耗时": f"{cost_time:.3f}s",
            "结果数量": result_count
        })
        if isinstance(data, list) and len(data) > 0:
            print(f"\n📄 第一条结果预览:")
            print(json.dumps(data[0], ensure_ascii=False, indent=2)[:300])
    else:
        print_result(False, "32B端点调用失败", {
            "错误": error,
            "耗时": f"{cost_time:.3f}s"
        })
    
    return success, cost_time


async def test_8b_endpoint():
    """测试8B端点"""
    print_header("测试 /rag/chat_8b (8B模型)")
    
    payload = {
        "tenant_id": TENANT_ID,
        "contact_id": CONTACT_ID,
        "account_id": ACCOUNT_ID,
        "thought_unit": "",
        "score_threshold": SCORE_THRESHOLD,
        "kb_name": TENANT_ID,
        "top_k": TOP_K,
        "context": TEST_CONTEXT,
        "user_profile": TEST_USER_PROFILE,
        "history_summary": TEST_HISTORY_SUMMARY,
        "rewritten_query": TEST_REWRITTEN_QUERY
    }
    
    print(f"📦 Payload:")
    print(f"  - tenant_id: {TENANT_ID}")
    print(f"  - context长度: {len(TEST_CONTEXT)}")
    print(f"  - user_profile长度: {len(TEST_USER_PROFILE)}")
    print(f"  - history_summary长度: {len(TEST_HISTORY_SUMMARY)}")
    print(f"  - rewritten_query: {TEST_REWRITTEN_QUERY}")
    print(f"🔄 调用中...")
    
    success, data, error, cost_time = await call_rag_endpoint("/rag/chat_8b", payload)
    
    if success:
        result_count = len(data) if isinstance(data, list) else "N/A"
        print_result(True, "8B端点调用成功", {
            "耗时": f"{cost_time:.3f}s",
            "结果数量": result_count
        })
        if isinstance(data, list) and len(data) > 0:
            print(f"\n📄 第一条结果预览:")
            print(json.dumps(data[0], ensure_ascii=False, indent=2)[:300])
    else:
        print_result(False, "8B端点调用失败", {
            "错误": error,
            "耗时": f"{cost_time:.3f}s"
        })
    
    return success, cost_time


async def test_connection():
    """测试基础连接"""
    print_header("连接诊断")
    
    print(f"🔍 检查RAG服务: {RAG_BASE_URL}")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(RAG_BASE_URL)
        print_result(True, "服务可访问", {"状态码": response.status_code})
        return True
    except httpx.ConnectError:
        print_result(False, "无法连接到服务", {
            "建议": "检查服务是否启动，端口是否正确"
        })
        return False
    except Exception as e:
        print_result(False, f"连接测试异常: {type(e).__name__}", {
            "错误": str(e)
        })
        return False


# ==================== 主函数 ====================

async def main():
    """主测试流程"""
    print_header("🚀 RAG API 独立测试")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 RAG服务: {RAG_BASE_URL}")
    
    # 1. 连接测试
    connection_ok = await test_connection()
    if not connection_ok:
        print("\n❌ 连接失败，无法继续测试")
        return 1
    
    # 2. 测试32B端点
    success_32b, time_32b = await test_32b_endpoint()
    
    # 3. 测试8B端点
    success_8b, time_8b = await test_8b_endpoint()
    
    # 4. 总结
    print_header("测试总结")
    
    if success_32b and success_8b:
        print("✅ 所有端点测试通过")
        print(f"\n⏱️  性能:")
        print(f"  - 32B: {time_32b:.3f}s")
        print(f"  - 8B: {time_8b:.3f}s")
        print(f"\n💡 可以开始训练流程")
        return 0
    else:
        print("❌ 部分端点测试失败")
        if not success_32b:
            print("  - 32B端点异常")
        if not success_8b:
            print("  - 8B端点异常")
        print(f"\n💡 建议:")
        print(f"  1. 检查RAG服务日志")
        print(f"  2. 确认端点路径是否正确")
        print(f"  3. 验证payload格式")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
