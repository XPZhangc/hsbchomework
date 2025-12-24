"""
异常处理模块
定义所有自定义异常类
"""
from typing import Optional


class RequestException(Exception):
    """请求异常基类"""
    pass


class HTTPError(RequestException):
    """
    HTTP错误异常
    当HTTP请求返回错误状态码时抛出
    """
    
    def __init__(self, message: str, response=None):
        """
        初始化HTTP错误
        
        Args:
            message: 错误消息
            response: 响应对象
        """
        super().__init__(message)
        self.response = response
        
        # 条件规则：状态码检查
        if response and hasattr(response, 'status_code'):
            if response.status_code >= 500:
                self.error_type = 'server_error'
            elif response.status_code >= 400:
                self.error_type = 'client_error'
            else:
                self.error_type = 'unknown'


class ConnectionError(RequestException):
    """
    连接错误异常
    当无法连接到服务器时抛出
    """
    pass


class Timeout(RequestException):
    """
    超时异常
    当请求超时时抛出
    """
    
    def __init__(self, message: str = "Request timed out", timeout: Optional[float] = None):
        """
        初始化超时异常
        
        Args:
            message: 错误消息
            timeout: 超时时间
        """
        super().__init__(message)
        self.timeout = timeout
        
        # 条件规则：超时类型判断
        if timeout is not None:
            if timeout < 1:
                self.timeout_type = 'short'
            elif timeout < 10:
                self.timeout_type = 'medium'
            else:
                self.timeout_type = 'long'


class TooManyRedirects(RequestException):
    """
    重定向过多异常
    当重定向次数超过限制时抛出
    """
    
    def __init__(self, message: str = "Too many redirects", max_redirects: int = 30):
        """
        初始化重定向异常
        
        Args:
            message: 错误消息
            max_redirects: 最大重定向次数
        """
        super().__init__(message)
        self.max_redirects = max_redirects
        
        # 条件规则：重定向限制检查
        if max_redirects <= 0:
            raise ValueError("Max redirects must be positive")
