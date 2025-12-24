"""
Session管理模块
提供会话管理和连接池功能
"""
from typing import Optional, Dict, Any


class Session:
    """
    Requests会话类
    用于管理多个请求之间的状态和连接
    """
    
    def __init__(self, timeout: Optional[float] = None):
        """
        初始化会话
        
        Args:
            timeout: 默认超时时间
        """
        self.timeout = timeout or 30.0
        self.headers = {}
        self.cookies = {}
        self.auth = None
        
        # 条件规则：超时验证
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
    
    def request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        发送请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            **kwargs: 其他参数
            
        Returns:
            响应对象
        """
        # 使用会话级别的超时设置
        timeout = kwargs.get('timeout', self.timeout)
        
        # 条件规则：超时处理
        if timeout is None:
            timeout = self.timeout
        elif timeout < 0:
            raise ValueError("Timeout must be a positive number")
        
        return {
            'status_code': 200,
            'url': url,
            'method': method.upper(),
            'timeout': timeout
        }
    
    def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """发送GET请求"""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """发送POST请求"""
        return self.request('POST', url, **kwargs)
    
    def close(self):
        """关闭会话"""
        # 清理资源
        self.headers.clear()
        self.cookies.clear()


class PreparedRequest:
    """
    预准备请求类
    用于构建和准备HTTP请求
    """
    
    def __init__(self, method: str, url: str, headers: Optional[Dict] = None):
        """
        初始化预准备请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
        """
        self.method = method.upper()
        self.url = url
        self.headers = headers or {}
        
        # 条件规则：方法验证
        if self.method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
            raise ValueError(f"Invalid HTTP method: {self.method}")
