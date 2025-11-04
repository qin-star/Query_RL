from datetime import datetime

# 获取当前时间戳字符串
def get_now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_yyyy_mm_dd_hh_mm_ss_time_str(ymd_str: str) -> datetime:
    return datetime.strptime(ymd_str, '%Y-%m-%d %H:%M:%S')