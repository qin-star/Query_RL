from datetime import datetime
import time
from contextlib import contextmanager
from src.utils.log import logger

# 获取当前时间戳字符串
def get_now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_yyyy_mm_dd_hh_mm_ss_time_str(ymd_str: str) -> datetime:
    return datetime.strptime(ymd_str, '%Y-%m-%d %H:%M:%S')

# 计时上下文管理器类
@contextmanager
def TimingContext(name: str = ""):
    """计时上下文管理器，用于测量代码执行时间"""
    start_time = time.time()
    yield start_time
    end_time = time.time()
    elapsed_time = end_time - start_time
    if name:
        logger.info(f"[{name}] 执行时间: {elapsed_time:.4f}秒")
    return elapsed_time