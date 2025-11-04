import re
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

from src.utils.time import parse_yyyy_mm_dd_hh_mm_ss_time_str
from src.utils.contexts import rsplit_context
from src.utils.contexts import Message

class Message(BaseModel):

    """
    图片类型消息示例
    {
        "content": "xxxxxxxxxxxxxxx",
        "msgType": "image",
        "role": "customer",
        "url": "/xxx/xxx.jpeg",
        "msgTime": 1750148399466
    }
    文字类型消息示例
    {
        "content": "xxxxxxxxxxxxxxx",
        "msgType": "text",
        "role": "customer",
        "url": ""
        "msgTime": 1750148399466,
    }
    """
    role: Literal["customer", "salesman", "销售", "客户"] | str = ""
    dt_str: str = ""
    content: str = ""
    dt: Optional[datetime] = None
    url: str = ""
    msgTime: int = 100
    msgType: str = "text"

    def model_post_init(self, __context) -> None:
        if self.dt_str:
            self.dt = parse_yyyy_mm_dd_hh_mm_ss_time_str(self.dt_str)
            self.msgTime = int(self.dt.timestamp())
    
    def is_need_message(self):
        """判断是否为需要的类型消息"""
        return self.msgType in ["text", "image", "voice"]

    def is_salesman(self):
        return self.role in ["salesman", "销售"]

    def is_customer(self):
        return self.role in ["customer", "客户"]

    def is_image(self):
        """判断是否为图片类型消息"""
        return self.msgType == "image"

    def display_with_time(self):
        return f"[{'客户' if self.is_customer() else '销售'}][{self.dt_str}]：{self.content}"

    def display(self):
        return f"[{'客户' if self.is_customer() else '销售'}]：{self.content}"

    def is_multi_line(self):
        return "\n" in self.content

    def is_quote(self):
        return "这是一条引用" in self.content

    def is_empty(self):
        return not self.content.strip()

    def is_equal(self, msg):
        return not self.is_empty() and \
            self.dt_str == msg.dt_str and \
            self.get_role_name() == msg.get_role_name() and self.content == msg.content

    def get_role_name(self, lan: Literal["cn", "en"] = "cn"):
        if self.is_customer():
            return "客户" if lan == "cn" else "customer"
        if self.is_salesman():
            return "销售" if lan == "cn" else "salesman"
        return ""

    def to_json(self, lan: Literal["cn", "en"] = "en") -> dict:
        return {
            "role": self.get_role_name(lan=lan),
            # "dt_str": self.dt_str,
            "content": self.content,
            "msgType": self.msgType,
            "msgTime": self.msgTime * 1000,
            "url": self.url,
        }

    @staticmethod
    def judge_is_image_from_display(content: str):
        return "图片概要" in content and "信息总结" in content


def rsplit_context(context):
    """
    销售和客户分开
    :param context:
    :return:
    """
    if "[客户]" in context or "[销售]" in context:
        pat = r'(\[客户\]|\[销售\])'
    else:
        pat = r'(客户|销售)'
    # messages = re.split(r'(客户|销售)', context, flags=re.DOTALL)
    messages = re.split(pat, context, flags=re.DOTALL)
    return [''.join(pair).strip() for pair in zip(messages[1::2], messages[2::2]) if ''.join(pair).strip()]

def build_from_display(cls, raw_str: str):
        """从字符串上下文格式解析"""
        context_list = rsplit_context(raw_str.strip())
        format_context_list = list()
        for msg in context_list:
            # todo
            if "[客户]" in raw_str or "[销售]" in raw_str:
                pattern = r'\[(客户|销售)\]\s*\[(\d+-\d+-\d+\s*\d+:\d+:\d+)\][:：\s]*(.*)'
            else:
                pattern = r'(客户|销售)[:：\s]*(.*)'
            for parts in re.findall(pattern, msg, flags=re.DOTALL):
                if len(parts) == 3:
                    role, dt_str, content = parts
                else:
                    role, content = parts
                    dt_str = ""
                if Message.judge_is_image_from_display(content):
                    msg_type = "image"
                else:
                    msg_type = "text"
                msg = Message(role=role, dt_str=dt_str, msgType=msg_type, content=content)

                if not msg.is_empty():
                    format_context_list.append(msg)
        return cls(messages=format_context_list)
