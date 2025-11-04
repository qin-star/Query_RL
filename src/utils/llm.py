import ast
import json
import sys
from typing import Union

from langchain_openai import ChatOpenAI

from src.utils.log import logger
from src.utils.settings import SETTINGS

DEFAULT_LLM_MODEL = SETTINGS.MODEL_SETTINGS.DEFAULT_LLM_MODEL
DEFAULT_MODEL_CONFIG = SETTINGS.MODEL_SETTINGS.DEFAULT_MODEL_CONFIG
PLATFORM_MODEL_CONFIG = SETTINGS.MODEL_SETTINGS.PLATFORM_MODEL_CONFIG


def _get_llm_class(platform_type):
    """获取LLM类"""
    if hasattr(sys.modules[__name__], platform_type):
        return getattr(sys.modules[__name__], platform_type)
    logger.warning(f"class {platform_type} not exists, use ChatOpenAI")
    return ChatOpenAI


def _initialize_single_model(model_name: str, model_config):
    """初始化单个模型"""
    try:
        llm_class = _get_llm_class(model_config.PLATFORM_TYPE)
        model_instance = llm_class(**model_config.MODEL_CONFIG)
        logger.info(f"模型 {model_name} 初始化成功, base_url: {model_config.MODEL_CONFIG['base_url']}")
        return model_instance
    except Exception as e:
        logger.error(f"初始化模型 {model_name} 失败: {e}")
        return None


# 使用字典存储已初始化的模型（懒加载）
INITIALIZED_MODELS = {}

def get_chat_llm(model_name: Union[str, None] = DEFAULT_LLM_MODEL):
    """
    获取聊天模型实例（懒加载）
    
    Args:
        model_name: 模型名称，如果为None则使用默认模型
    
    Returns:
        模型实例
    """
    # 如果没有指定模型名称，使用默认模型
    if not model_name:
        model_name = DEFAULT_LLM_MODEL
    
    # 检查是否已经初始化过该模型
    if model_name in INITIALIZED_MODELS:
        logger.debug(f"使用已初始化的模型: {model_name}")
        return INITIALIZED_MODELS[model_name]
    
    # 懒加载：初始化指定的模型
    if model_name == DEFAULT_LLM_MODEL:
        # 初始化默认模型
        model_instance = _initialize_single_model(model_name, DEFAULT_MODEL_CONFIG)
        if model_instance:
            INITIALIZED_MODELS[model_name] = model_instance
            return model_instance
    else:
        # 初始化平台模型
        if model_name in PLATFORM_MODEL_CONFIG:
            model_instance = _initialize_single_model(model_name, PLATFORM_MODEL_CONFIG[model_name])
            if model_instance:
                INITIALIZED_MODELS[model_name] = model_instance
                return model_instance
    
    # 如果指定的模型初始化失败，尝试使用默认模型
    if model_name != DEFAULT_LLM_MODEL:
        logger.warning(f"模型 {model_name} 初始化失败，尝试使用默认模型 {DEFAULT_LLM_MODEL}")
        if DEFAULT_LLM_MODEL not in INITIALIZED_MODELS:
            default_model = _initialize_single_model(DEFAULT_LLM_MODEL, DEFAULT_MODEL_CONFIG)
            if default_model:
                INITIALIZED_MODELS[DEFAULT_LLM_MODEL] = default_model
                return default_model
        else:
            return INITIALIZED_MODELS[DEFAULT_LLM_MODEL]
    
    # 如果所有模型都初始化失败，创建一个基础的ChatOpenAI实例作为兜底
    logger.error(f"所有模型初始化失败，创建基础ChatOpenAI实例")
    return ChatOpenAI()


def quick_request(query, model_name=DEFAULT_LLM_MODEL):
    llm = get_chat_llm(model_name=model_name)
    response = llm.invoke(query)
    return response.content


async def quick_request_async(query, model_name=DEFAULT_LLM_MODEL):
    llm = get_chat_llm(model_name=model_name)
    response = await llm.ainvoke(query)
    return response.content


class SafeParser:

    @staticmethod
    def parse_json_to_dict(content: str) -> dict:
        result = {}

        try:
            result = json.loads(content)
        except Exception as e:
            """
            无法用 json 解析，尝试着使用字符串匹配的方式再次解析一次
            """
            logger.debug(f"parse by string, error={e}, content={content}")
            lines: list[str] = content.splitlines()
            tmp_result, start, end = "", 0, len(lines) - 1

            for i in range(len(lines)):
                if lines[i].strip().startswith("{"):
                    start = i
                    break
            for j in range(len(lines) - 1, 0, -1):
                if lines[j].strip().endswith("}"):
                    end = j
                    break
            for k in range(start, end + 1):
                tmp_result += lines[k].strip()

            response = tmp_result.replace(",}", "}")
            try:
                result = json.loads(response)
            except Exception as e:
                logger.debug(f"json.loads(response) error, {e}")
                if "{" in response and "}" in response:
                    try:
                        result = ast.literal_eval(response)
                        if not isinstance(result, dict):
                            result = {}
                    except Exception as e:
                        logger.debug(f"ast.literal_eval(response) error, {e}")
                        result = {}

        logger.debug(f'after parse, content={result}')

        return result

    @staticmethod
    def parse_json_to_list(content: str) -> dict:
        result = []

        try:
            result = json.loads(content)
        except Exception as e:
            """
            无法用 json 解析，尝试着使用字符串匹配的方式再次解析一次
            """
            logger.debug(f"parse by string, error={e}, content={content}")
            lines: list[str] = content.splitlines()
            tmp_result, start, end = "", 0, len(lines) - 1

            for i in range(len(lines)):
                if lines[i].strip().startswith("["):
                    start = i
                    break
            for j in range(len(lines) - 1, 0, -1):
                if lines[j].strip().endswith("]"):
                    end = j
                    break
            for k in range(start, end + 1):
                tmp_result += lines[k].strip()

            response = tmp_result.replace(",]", "]")
            try:
                result = json.loads(response)
            except Exception as e:
                logger.debug(e)
                if "[" in response and "]" in response:
                    try:
                        result = ast.literal_eval(response)
                        if not isinstance(result, list):
                            result = []
                    except Exception as e:
                        logger.debug(e)
                        result = []

        logger.warning(f'after parse, content={result}')

        return result


if __name__ == "__main__":
    llm = get_chat_llm()
    print(llm)
