#!/usr/bin/env python3
"""
Qwen3-8B vLLM 服务压测脚本
目标：最大化 H100 GPU 利用率
"""

import asyncio
import aiohttp
import random
import argparse
import signal
import sys

# 默认配置
DEFAULT_ENDPOINT = "http://10.72.1.16:36784/v1/chat/completions"
DEFAULT_API_KEY = "sk-xxxx"
DEFAULT_MODEL_NAME = "Qwen3-8B-SFT"
DEFAULT_CONCURRENCY = 4  # 总并发数（建议 16~64）
DEFAULT_MAX_TOKENS = 512

# 长 prompt 池（提升计算负载）
PROMPTS = [
    "请详细解释量子力学的基本原理，包括波函数、叠加态和测量问题，不少于500字。",
    "写一篇关于人工智能对未来社会影响的议论文，要求结构清晰、论据充分，不少于600字。",
    "用通俗易懂的语言解释Transformer模型的工作原理，包括自注意力机制、位置编码和前馈网络。",
    "假设你是历史学家，请分析工业革命对全球经济格局的长期影响，并举例说明。",
    "生成一段高质量的 Python 代码，实现一个支持异步、代理和重试机制的 Web 爬虫，并附带详细注释。",
    "请总结深度学习在过去十年中的三大突破，并分析其对计算机视觉领域的影响。",
    "如果你是一位经济学家，请解释通货膨胀的成因及其对普通家庭的影响。",
    "描述宇宙大爆炸理论的主要证据，并讨论暗物质在宇宙演化中的作用。"
] * 50  # 扩充到 400+ 条，增加多样性


class StressTester:
    def __init__(self, endpoint, api_key, model_name, concurrency, max_tokens):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.session = None
        self.running = True

    async def send_request(self):
        """发送单个请求"""
        prompt = random.choice(PROMPTS)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个专业、准确、有帮助的 AI 助手。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            async with self.session.post(self.endpoint, json=payload, headers=headers) as resp:
                await resp.read()  # 不解析响应，只消耗数据
        except Exception:
            pass  # 忽略错误，持续压测

    async def worker(self):
        """工作协程：持续发请求"""
        while self.running:
            await self.send_request()

    async def run(self):
        # 设置信号处理（优雅退出）
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._stop)

        # 创建 aiohttp session
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=100)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        print(f"🚀 开始压测 | endpoint: {self.endpoint}")
        print(f"   并发数: {self.concurrency} | max_tokens: {self.max_tokens}")
        print("   按 Ctrl+C 停止压测\n")

        try:
            tasks = [self.worker() for _ in range(self.concurrency)]
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._stop()
        finally:
            await self.session.close()

    def _stop(self):
        print("\n🛑 正在停止压测...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B vLLM GPU 压测工具")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="API 地址")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="模型名称")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="总并发数 (默认: 32)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="生成最大 token 数 (默认: 512)")

    args = parser.parse_args()

    tester = StressTester(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model_name=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens
    )

    try:
        asyncio.run(tester.run())
    except KeyboardInterrupt:
        print("\n已退出。")
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()