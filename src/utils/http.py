from urllib.parse import urlparse, urlunparse

import httpx

from src.constant.http_constants import (
    HTTP_HEADERS, RequestMethod, ResponseCode
)
from src.utils.log import logger


def parse_microservice_url(url):
    """ 解析url，如果是微服务间调用，解析服务名 """
    parsed_url = urlparse(url)
    host = parsed_url.hostname

    # 获取解析后的ip和端口
    service_address = host
    new_netloc = service_address

    new_url = urlunparse((
        parsed_url.scheme,
        new_netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment
    ))

    return new_url


class HttpUtil(object):

    @classmethod
    def _handle_response_and_log_error(cls, response, url, method, request_data=None):
        """处理响应并记录错误日志"""
        if response.status_code == ResponseCode.SUCCESS:
            return response.json()
        
        logger.error(
            "HTTP request failed",
            url=url,
            method=method,
            request=request_data,
            status_code=response.status_code,
            response_text=response.text,
            format=False
        )
        return None

    @classmethod
    def _log_exception_error(cls, url, method, request_data, exception):
        """记录异常错误日志"""
        logger.error(
            "HTTP request exception",
            url=url,
            method=method,
            request=request_data,
            exception=repr(exception),
            exc_info=True,
            format=False
        )

    @classmethod
    def post(cls, url: str, data: dict, headers=HTTP_HEADERS, **kwargs):
        """
        定义类方法post, 用于发送POST请求
        args:
        :param url: 要请求的URL
        :param data: 请求参数，类型为dict
        :param headers: 可选参数，设置HTTP头部信息
        :return: 如果请求成功返回解析后的json数据；请求失败返回None
        """
        try:
            with httpx.Client(timeout=None) as client:
                response = client.post(url=parse_microservice_url(url), json=data, headers=headers, **kwargs)
                return cls._handle_response_and_log_error(response, url, RequestMethod.POST, data)
        except Exception as e:
            cls._log_exception_error(url, RequestMethod.POST, data, e)
            return None
    
    @classmethod
    async def apost(cls, url: str, data: dict, headers=HTTP_HEADERS, **kwargs):
        """
        定义类方法apost, 用于发送异步POST请求
        args:
        :param url: 要请求的URL
        :param data: 请求参数，类型为dict 
        :param headers: 可选参数，设置HTTP头部信息
        :return: 如果请求成功返回解析后的json数据；请求失败返回None
        """
        try:
            parsed_url = parse_microservice_url(url)
            logger.debug(f"[HTTP] Sending POST to {parsed_url}")
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url=parsed_url, json=data, headers=headers, **kwargs)
                logger.debug(f"[HTTP] Response status: {response.status_code}")
                return cls._handle_response_and_log_error(response, url, RequestMethod.POST, data)
        except Exception as e:
            logger.error(f"[HTTP] Exception type: {type(e).__name__}, message: {str(e)}")
            cls._log_exception_error(url, RequestMethod.POST, data, e)
            return None

    @classmethod
    def get(cls, url: str, params: dict, headers=HTTP_HEADERS, **kwargs):
        """
        定义类方法get, 用于发送GET请求
        args:
        :param url: 要请求的URL
        :param params: 请求参数，类型为dict
        :param headers: 可选参数，设置HTTP头部信息
        :return: 如果请求成功返回解析后的json数据；请求失败返回None
        """
        try:
            with httpx.Client(timeout=None) as client:
                response = client.get(url=parse_microservice_url(url), params=params, headers=headers, **kwargs)
                return cls._handle_response_and_log_error(response, url, RequestMethod.GET, params)
        except Exception as e:
            cls._log_exception_error(url, RequestMethod.GET, params, e)
            return None
    
