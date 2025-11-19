"""测试RAG调用是否正常"""
import asyncio
import sys
import os
import json

# 添加src路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.pipeline import get_rag_rl_result
from src.core.rag_chater import RagChater

async def test_rag_call():
    """测试RAG调用"""
    
    # 测试数据
    context = """
[客户][2025-10-10 00:01:39]: 我已经添加了你，现在我们可以开始聊天了。
[销售][2025-10-10 00:01:54]: 同学你好❤，我是你的专属课程助教老师-史老师
直播课是10月11号-14号每晚19：00，这几天就由我来负责你本次的学习安排以及资料发放~
📚课前必做
1.激活直播课👉https://cl2.cn/Buaib6u6（点击右下角免费领取--显示已报名去学习即为预约成功）
2.学习档案👉weixin://dl/business/?t=QpgfMaqYHDd（便于奇函老师了解同学目前学习情况）
2个链接完成之后请同学说一下【报名手机号+目标考试】，老师会根据档案信息单独给你制定专属学习规划~
"""
    
    user_profile = "用户为应届毕业生，本科在读，计划以应届生身份参加2025年内蒙古省考"
    history_summary = "用户参加了2024年下半年事业单位联考，成绩不理想（48分），未参加国考"
    rewritten_query = "事业单位联考每年是否举行两次？"
    
    print("=" * 80)
    print("测试RAG调用")
    print("=" * 80)
    print(f"\n输入参数:")
    print(f"  - context长度: {len(context)}")
    print(f"  - user_profile: {user_profile}")
    print(f"  - history_summary: {history_summary}")
    print(f"  - rewritten_query: {rewritten_query}")
    
    try:
        print("\n🔍 开始调用RAG...")
        chat_resp, chat_8b_resp = await get_rag_rl_result(
            context=context,
            user_profile=user_profile,
            history_summary=history_summary,
            rewritten_query=rewritten_query
        )
        
        print("\n✅ RAG调用成功！")
        print(f"\n🔷 32B响应:")
        print(f"  - 类型: {type(chat_resp)}")
        print(f"  - 长度: {len(chat_resp) if chat_resp else 0}")
        if chat_resp:
            print(f"  - 内容预览: {json.dumps(chat_resp[:2] if isinstance(chat_resp, list) else chat_resp, ensure_ascii=False, indent=2)}")
        
        print(f"\n🔶 8B响应:")
        print(f"  - 类型: {type(chat_8b_resp)}")
        print(f"  - 长度: {len(chat_8b_resp) if chat_8b_resp else 0}")
        if chat_8b_resp:
            print(f"  - 内容预览: {json.dumps(chat_8b_resp[:2] if isinstance(chat_8b_resp, list) else chat_8b_resp, ensure_ascii=False, indent=2)}")
            
    except Exception as e:
        print(f"\n❌ RAG调用失败: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_rag_call():
    """直接测试RAG接口"""
    print("\n" + "=" * 80)
    print("直接测试RAG接口（验证参数格式）")
    print("=" * 80)
    
    rag = RagChater(
        tenant_id="chengla",
        contact_id="Customer_knowledge_17",
        account_id="Sale_knowledge_17",
        message_id="chengla_query_rl_message_id"
    )
    
    context = "[客户][2025-10-10 00:01:39]: 我已经添加了你，现在我们可以开始聊天了。"
    
    print("\n测试 /rag/chat (32B)...")
    try:
        response_data, status, request_body, cost_time = await rag.chat(
            context=context,
            score_threshold=0.9,
            top_k=3
        )
        print(f"✅ 32B调用成功")
        print(f"  - 状态: {status}")
        print(f"  - 请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}")
        print(f"  - 响应长度: {len(response_data) if response_data else 0}")
    except Exception as e:
        print(f"❌ 32B调用失败: {e}")
    
    print("\n测试 /rag/chat_8b (8B)...")
    try:
        response_data, status, request_body, cost_time = await rag.chat_8b(
            context=context,
            user_profile="测试用户画像",
            history_summary="测试历史摘要",
            rewritten_query="测试查询",
            score_threshold=0.9,
            top_k=3
        )
        print(f"✅ 8B调用成功")
        print(f"  - 状态: {status}")
        print(f"  - 请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}")
        print(f"  - 响应长度: {len(response_data) if response_data else 0}")
    except Exception as e:
        print(f"❌ 8B调用失败: {e}")

async def test_training_scenario():
    """模拟训练场景的RAG调用"""
    print("\n" + "=" * 80)
    print("模拟训练场景的RAG调用（使用实际训练数据）")
    print("=" * 80)
    
    # 使用你提供的实际payload
    cleaned_context = """[客户][2025-10-10 00:01:39]: 我已经添加了你，现在我们可以开始聊天了。
[销售][2025-10-10 00:01:54]: 同学你好❤，我是你的专属课程助教老师-史老师
直播课是10月11号-14号每晚19：00，这几天就由我来负责你本次的学习安排以及资料发放~
📚课前必做
1.激活直播课👉https://cl2.cn/Buaib6u6（点击右下角免费领取--显示已报名去学习即为预约成功）
2.学习档案👉weixin://dl/business/?t=QpgfMaqYHDd（便于奇函老师了解同学目前学习情况）
2个链接完成之后请同学说一下【报名手机号+目标考试】，老师会根据档案信息单独给你制定专属学习规划~"""
    
    rag = RagChater(
        tenant_id="chengla",
        contact_id="Customer_knowledge_17",
        account_id="Sale_knowledge_17",
        message_id="chengla_query_rl_message_id"
    )
    
    print("\n🔷 测试32B调用（训练场景）...")
    try:
        response_data_32b, status_32b, request_body_32b, cost_time_32b = await rag.chat(
            context=cleaned_context,
            score_threshold=0.9,
            top_k=3
        )
        print(f"✅ 32B调用成功")
        print(f"  - 状态: {status_32b}")
        print(f"  - 耗时: {cost_time_32b:.2f}s")
        print(f"  - 响应长度: {len(response_data_32b) if response_data_32b else 0}")
        if response_data_32b:
            print(f"  - 第一条结果: {json.dumps(response_data_32b[0], ensure_ascii=False, indent=2)[:200]}...")
    except Exception as e:
        print(f"❌ 32B调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔶 测试8B调用（训练场景）...")
    try:
        response_data_8b, status_8b, request_body_8b, cost_time_8b = await rag.chat_8b(
            context=cleaned_context,
            user_profile='用户为应届毕业生，本科在读，计划以应届生身份参加2025年内蒙古省考，目标考试类型包括公务员和事业单位联考，当前处于备考初期阶段，关注考试规划、课程内容及备考方法，此前备考经验不足，成绩不理想。',
            history_summary='用户参加了2024年下半年事业单位联考，成绩不理想（48分），未参加国考，计划以应届生身份参加2025年内蒙古省考，并同步备考事业编考试，销售老师推荐了杨奇涵老师的课程，强调系统规划和高效备考的重要性。',
            rewritten_query='事业单位联考每年是否举行两次？',
            score_threshold=0.9,
            top_k=3
        )
        print(f"✅ 8B调用成功")
        print(f"  - 状态: {status_8b}")
        print(f"  - 耗时: {cost_time_8b:.2f}s")
        print(f"  - 响应长度: {len(response_data_8b) if response_data_8b else 0}")
        if response_data_8b:
            print(f"  - 第一条结果: {json.dumps(response_data_8b[0], ensure_ascii=False, indent=2)[:200]}...")
    except Exception as e:
        print(f"❌ 8B调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("训练场景测试完成")
    print("=" * 80)

if __name__ == "__main__":
    print("开始测试RAG调用配置...\n")
    
    # 测试1：基础接口测试
    asyncio.run(test_direct_rag_call())
    
    # 测试2：使用pipeline的完整测试
    asyncio.run(test_rag_call())
    
    # 测试3：模拟训练场景（最重要！）
    asyncio.run(test_training_scenario())
