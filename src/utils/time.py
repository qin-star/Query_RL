from datetime import datetime
import time
from contextlib import contextmanager
from src.utils.log import logger


# 获取当前时间戳字符串
def get_now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_yyyy_mm_dd_hh_mm_ss_time_str(ymd_str: str) -> datetime:
    return datetime.strptime(ymd_str, '%Y-%m-%d %H:%M:%S')


class _TimingContextManager:
    """计时上下文管理器辅助类"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.cost_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.cost_time = end_time - self.start_time
        if self.name:
            logger.info(f"[{self.name}] 执行时间: {self.cost_time:.4f}秒")
        return False


# 计时上下文管理器类
@contextmanager
def TimingContext(name: str = ""):
    """计时上下文管理器，用于测量代码执行时间
    
    使用方法:
        with TimingContext("my_task") as timing:
            # do something
            pass
        print(f"耗时: {timing.cost_time}秒")
    """
    timer = _TimingContextManager(name)
    with timer:
        yield timer