"""
模拟实际训练过程的RAG调用测试
直接使用HTTP调用，完全复现训练时的调用方式
"""
import asyncio
import sys
import os
import json
import httpx
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# RAG服务配置
RAG_BASE_URL = "http://localhost:7861"
RAG_8B_ENDPOINT = "/rag/chat_8b"
RAG_32B_ENDPOINT = "/rag/chat"

# 测试用的完整payload（你提供的实际数据）
TEST_PAYLOAD_8B = {
    "tenant_id": "chengla",
    "contact_id": "Customer_knowledge_17",
    "account_id": "Sale_knowledge_17",
    "thought_unit": "",
    "score_threshold": 0.9,
    "kb_name": "chengla",
    "top_k": 3,
    "context": """[客户][2025-10-10 00:01:39]: 我已经添加了你，现在我们可以开始聊天了。
[销售][2025-10-10 00:01:54]: 同学你好❤，我是你的专属课程助教老师-史老师
直播课是10月11号-14号每晚19：00，这几天就由我来负责你本次的学习安排以及资料发放~
📚课前必做
1.激活直播课👉https://cl2.cn/Buaib6u6（点击右下角免费领取--显示已报名去学习即为预约成功）
2.学习档案👉weixin://dl/business/?t=QpgfMaqYHDd（便于奇函老师了解同学目前学习情况）
2个链接完成之后请同学说一下【报名手机号+目标考试】，老师会根据档案信息单独给你制定专属学习规划~""",
    "user_profile": "用户为应届毕业生，本科在读，计划以应届生身份参加2025年内蒙古省考，目标考试类型包括公务员和事业单位联考，当前处于备考初期阶段，关注考试规划、课程内容及备考方法，此前备考经验不足，成绩不理想。",
    "history_summary": "用户参加了2024年下半年事业单位联考，成绩不理想（48分），未参加国考，计划以应届生身份参加2025年内蒙古省考，并同步备考事业编考试，销售老师推荐了杨奇涵老师的课程，强调系统规划和高效备考的重要性。",
    "rewritten_query": "事业单位联考每年是否举行两次？"
}


def print_section(title: str, symbol: str = "="):
    """打印分隔线"""
    print(f"\n{symbol * 80}")
    print(f"{title}")
    print(f"{symbol * 80}")


def print_payload_info(payload: dict):
    """打印payload信息"""
    print(f"📦 Payload信息:")
    print(f"  - tenant_id: {payload.get('tenant_id')}")
    print(f"  - contact_id: {payload.get('contact_id')}")
    print(f"  - account_id: {payload.get('account_id')}")
    print(f"  - kb_name: {payload.get('kb_name')}")
    print(f"  - score_threshold: {payload.get('score_threshold')}")
    print(f"  - top_k: {payload.get('top_k')}")
    print(f"  - context长度: {len(payload.get('context', ''))}")
    print(f"  - user_profile长度: {len(payload.get('user_profile', ''))}")
    print(f"  - history_summary长度: {len(payload.get('history_summary', ''))}")
    print(f"  - rewritten_query: {payload.get('rewritten_query')}")


async def call_rag_api(url: str, payload: dict, endpoint_name: str = "RAG") -> dict:
    """
    直接调用RAG API（模拟训练时的调用方式）
    
    Args:
        url: 完整的API URL
        payload: 请求payload
        endpoint_name: 端点名称（用于日志）
    
    Returns:
        响应数据字典，包含status, data, error等信息
    """
    result = {
        "success": False,
        "status_code": None,
        "data": None,
        "error": None,
        "cost_time": 0.0
    }
    
    start_time = datetime.now()
    
    try:
        print(f"\n🔄 正在调用 {endpoint_name}: {url}")
        print(f"⏱️  开始时间: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
            end_time = datetime.now()
            cost_time = (end_time - start_time).total_seconds()
            result["cost_time"] = cost_time
            result["status_code"] = response.status_code
            
            print(f"⏱️  结束时间: {end_time.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"⏱️  耗时: {cost_time:.3f}s")
            print(f"📊 状态码: {response.status_code}")
            
            if response.status_code == 200:
                result["success"] = True
                result["data"] = response.json()
                print(f"✅ 调用成功！")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"❌ 调用失败: {result['error']}")
                
    except httpx.TimeoutException as e:
        result["error"] = f"请求超时: {str(e)}"
        print(f"⏰ {result['error']}")
    except httpx.ConnectError as e:
        result["error"] = f"连接失败: {str(e)}"
        print(f"🔌 {result['error']}")
    except Exception as e:
        result["error"] = f"未知错误: {type(e).__name__}: {str(e)}"
        print(f"❌ {result['error']}")
        import traceback
        traceback.print_exc()
    
    return result


async def test_8b_endpoint():
    """测试8B端点（使用你提供的payload）"""
    print_section("测试 /rag/chat_8b 端点（8B模型）", "=")
    
    print_payload_info(TEST_PAYLOAD_8B)
    
    url = f"{RAG_BASE_URL}{RAG_8B_ENDPOINT}"
    result = await call_rag_api(url, TEST_PAYLOAD_8B, "8B端点")
    
    if result["success"]:
        data = result["data"]
        print(f"\n📋 响应数据分析:")
        print(f"  - 数据类型: {type(data)}")
        
        if isinstance(data, list):
            print(f"  - 结果数量: {len(data)}")
            if len(data) > 0:
                print(f"\n📄 第一条结果:")
                print(json.dumps(data[0], ensure_ascii=False, indent=2)[:500])
                if len(data) > 1:
                    print(f"\n... 还有 {len(data) - 1} 条结果")
        elif isinstance(data, dict):
            print(f"  - 字典键: {list(data.keys())}")
            print(f"\n📄 完整响应:")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:500])
        else:
            print(f"  - 原始数据: {str(data)[:200]}")
    
    return result


async def test_32b_endpoint():
    """测试32B端点（基准对比）"""
    print_section("测试 /rag/chat 端点（32B模型 - 基准对比）", "=")
    
    # 32B不需要user_profile等字段
    payload_32b = {
        "tenant_id": TEST_PAYLOAD_8B["tenant_id"],
        "contact_id": TEST_PAYLOAD_8B["contact_id"],
        "account_id": TEST_PAYLOAD_8B["account_id"],
        "thought_unit": TEST_PAYLOAD_8B["thought_unit"],
        "score_threshold": TEST_PAYLOAD_8B["score_threshold"],
        "kb_name": TEST_PAYLOAD_8B["kb_name"],
        "top_k": TEST_PAYLOAD_8B["top_k"],
        "context": TEST_PAYLOAD_8B["context"]
    }
    
    print_payload_info(payload_32b)
    
    url = f"{RAG_BASE_URL}{RAG_32B_ENDPOINT}"
    result = await call_rag_api(url, payload_32b, "32B端点")
    
    if result["success"]:
        data = result["data"]
        print(f"\n📋 响应数据分析:")
        print(f"  - 数据类型: {type(data)}")
        
        if isinstance(data, list):
            print(f"  - 结果数量: {len(data)}")
            if len(data) > 0:
                print(f"\n📄 第一条结果:")
                print(json.dumps(data[0], ensure_ascii=False, indent=2)[:500])
        elif isinstance(data, dict):
            print(f"  - 字典键: {list(data.keys())}")
            print(f"\n📄 完整响应:")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:500])
    
    return result


async def compare_results(result_8b: dict, result_32b: dict):
    """对比8B和32B的结果"""
    print_section("结果对比分析", "=")
    
    print(f"\n⏱️  性能对比:")
    print(f"  - 8B耗时: {result_8b['cost_time']:.3f}s")
    print(f"  - 32B耗时: {result_32b['cost_time']:.3f}s")
    print(f"  - 差异: {abs(result_8b['cost_time'] - result_32b['cost_time']):.3f}s")
    
    print(f"\n✅ 成功率:")
    print(f"  - 8B: {'成功' if result_8b['success'] else '失败'}")
    print(f"  - 32B: {'成功' if result_32b['success'] else '失败'}")
    
    if result_8b["success"] and result_32b["success"]:
        data_8b = result_8b["data"]
        data_32b = result_32b["data"]
        
        if isinstance(data_8b, list) and isinstance(data_32b, list):
            print(f"\n📊 结果数量对比:")
            print(f"  - 8B返回: {len(data_8b)} 条")
            print(f"  - 32B返回: {len(data_32b)} 条")
            
            # 检查结果是否相同
            if len(data_8b) == len(data_32b):
                print(f"  ✅ 结果数量一致")
            else:
                print(f"  ⚠️  结果数量不同")


async def test_with_variations():
    """测试不同的payload变体"""
    print_section("测试Payload变体", "=")
    
    variations = [
        {
            "name": "空user_profile",
            "payload": {**TEST_PAYLOAD_8B, "user_profile": ""}
        },
        {
            "name": "空history_summary",
            "payload": {**TEST_PAYLOAD_8B, "history_summary": ""}
        },
        {
            "name": "空rewritten_query",
            "payload": {**TEST_PAYLOAD_8B, "rewritten_query": ""}
        },
        {
            "name": "全部增强字段为空",
            "payload": {
                **TEST_PAYLOAD_8B,
                "user_profile": "",
                "history_summary": "",
                "rewritten_query": ""
            }
        }
    ]
    
    results = []
    for var in variations:
        print(f"\n🧪 测试变体: {var['name']}")
        url = f"{RAG_BASE_URL}{RAG_8B_ENDPOINT}"
        result = await call_rag_api(url, var['payload'], f"8B-{var['name']}")
        results.append({
            "name": var['name'],
            "result": result
        })
    
    # 汇总结果
    print_section("变体测试汇总", "-")
    for item in results:
        status = "✅" if item['result']['success'] else "❌"
        print(f"{status} {item['name']}: {item['result']['cost_time']:.3f}s")


async def diagnose_connection():
    """诊断连接问题"""
    print_section("连接诊断", "=")
    
    print(f"🔍 检查RAG服务连接...")
    print(f"  - 目标URL: {RAG_BASE_URL}")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 尝试连接根路径
            response = await client.get(RAG_BASE_URL)
            print(f"  ✅ 服务可访问 (状态码: {response.status_code})")
    except httpx.ConnectError:
        print(f"  ❌ 无法连接到服务")
        print(f"  💡 请检查:")
        print(f"     1. RAG服务是否启动？")
        print(f"     2. 端口7861是否正确？")
        print(f"     3. 防火墙是否阻止连接？")
    except Exception as e:
        print(f"  ⚠️  连接测试异常: {e}")


async def main():
    """主测试流程"""
    print_section("🚀 RAG训练场景模拟测试", "=")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 RAG服务: {RAG_BASE_URL}")
    
    # 1. 连接诊断
    await diagnose_connection()
    
    # 2. 测试8B端点（主要测试）
    result_8b = await test_8b_endpoint()
    
    # 3. 测试32B端点（对比）
    result_32b = await test_32b_endpoint()
    
    # 4. 对比结果
    await compare_results(result_8b, result_32b)
    
    # 5. 测试变体（可选）
    print(f"\n❓ 是否测试payload变体？(会额外调用4次API)")
    # 自动跳过，避免过多调用
    # await test_with_variations()
    
    # 最终总结
    print_section("测试总结", "=")
    if result_8b["success"]:
        print(f"✅ 8B端点测试通过")
        print(f"   - 可以用于训练流程")
        print(f"   - 响应时间: {result_8b['cost_time']:.3f}s")
    else:
        print(f"❌ 8B端点测试失败")
        print(f"   - 错误: {result_8b['error']}")
        print(f"   - 需要修复后才能开始训练")
    
    if result_32b["success"]:
        print(f"✅ 32B端点测试通过（基准）")
    else:
        print(f"⚠️  32B端点测试失败")
    
    print(f"\n💡 建议:")
    if result_8b["success"] and result_32b["success"]:
        print(f"  ✅ 两个端点都正常，可以开始训练")
        print(f"  📝 建议在训练前再次确认:")
        print(f"     - 模型生成的JSON格式是否正确")
        print(f"     - reward_score函数是否能正确解析")
    elif result_8b["success"]:
        print(f"  ⚠️  8B端点正常，但32B端点异常")
        print(f"     - 如果只用8B训练，可以继续")
        print(f"     - 如果需要对比，需要修复32B")
    else:
        print(f"  ❌ 8B端点异常，无法进行训练")
        print(f"     - 请检查RAG服务配置")
        print(f"     - 确认端点路径是否正确")


if __name__ == "__main__":
    asyncio.run(main())
