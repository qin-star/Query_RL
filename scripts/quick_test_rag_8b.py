"""
快速测试RAG 8B端点 - 使用你提供的实际payload
"""
import asyncio
import httpx
import json

# 你提供的完整payload
payload = {
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

async def test():
    url = "http://localhost:7861/rag/chat_8b"
    
    print("=" * 80)
    print("快速测试 RAG 8B 端点")
    print("=" * 80)
    print(f"\n🌐 URL: {url}")
    print(f"\n📦 Payload:")
    print(f"  - tenant_id: {payload['tenant_id']}")
    print(f"  - rewritten_query: {payload['rewritten_query']}")
    print(f"  - user_profile长度: {len(payload['user_profile'])}")
    print(f"  - history_summary长度: {len(payload['history_summary'])}")
    print(f"  - context长度: {len(payload['context'])}")
    
    try:
        print(f"\n🔄 发送请求...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
        print(f"\n📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 调用成功！")
            print(f"\n📋 响应数据:")
            print(f"  - 类型: {type(data)}")
            
            if isinstance(data, list):
                print(f"  - 结果数量: {len(data)}")
                if len(data) > 0:
                    print(f"\n📄 第一条结果:")
                    print(json.dumps(data[0], ensure_ascii=False, indent=2))
            else:
                print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 调用失败")
            print(f"响应内容: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
