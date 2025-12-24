"""
数据模型模块
定义请求和响应的数据模型
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from .exceptions import HTTPError


class Response:
    """
    HTTP响应类
    封装HTTP响应信息
    """
    
    def __init__(self, status_code: int, url: str, headers: Optional[Dict] = None):
        """
        初始化响应对象
        
        Args:
            status_code: 状态码
            url: 请求URL
            headers: 响应头
        """
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}
        self.text = ""
        self.content = b""
        self.elapsed = None
        
        # 条件规则：状态码分类
        if 200 <= status_code < 300:
            self.status_type = 'success'
        elif 300 <= status_code < 400:
            self.status_type = 'redirect'
        elif 400 <= status_code < 500:
            self.status_type = 'client_error'
        elif 500 <= status_code < 600:
            self.status_type = 'server_error'
        else:
            self.status_type = 'unknown'
    
    def json(self) -> Dict[str, Any]:
        """
        解析JSON响应
        
        Returns:
            JSON对象
        """
        import json
        # 条件规则：内容验证
        if not self.text:
            raise ValueError("Response content is empty")
        
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            raise ValueError("Response is not valid JSON")
    
    def raise_for_status(self):
        """
        如果状态码表示错误，抛出异常
        """
        # 条件规则：错误状态码检查
        if 400 <= self.status_code < 500:
            raise HTTPError(f"Client error: {self.status_code}", self)
        elif 500 <= self.status_code < 600:
            raise HTTPError(f"Server error: {self.status_code}", self)


class Request:
    """
    HTTP请求类
    封装HTTP请求信息
    """
    
    def __init__(self, method: str, url: str, headers: Optional[Dict] = None):
        """
        初始化请求对象
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
        """
        self.method = method.upper()
        self.url = url
        self.headers = headers or {}
        self.data = None
        self.params = None
        
        # 条件规则：方法验证
        if self.method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
            raise ValueError(f"Invalid HTTP method: {self.method}")
