"""
认证模块
提供HTTP认证功能
"""
from typing import Optional, Tuple


class HTTPBasicAuth:
    """
    HTTP基本认证类
    用于用户名密码认证
    """
    
    def __init__(self, username: str, password: str):
        """
        初始化基本认证
        
        Args:
            username: 用户名
            password: 密码
        """
        # 条件规则：参数验证
        if not username:
            raise ValueError("Username cannot be empty")
        if not password:
            raise ValueError("Password cannot be empty")
        
        self.username = username
        self.password = password
    
    def __call__(self, request):
        """将认证信息添加到请求中"""
        # 编码用户名和密码
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        request.headers['Authorization'] = f'Basic {encoded}'
        return request


class HTTPDigestAuth:
    """
    HTTP摘要认证类
    提供更安全的认证方式
    """
    
    def __init__(self, username: str, password: str):
        """
        初始化摘要认证
        
        Args:
            username: 用户名
            password: 密码
        """
        # 条件规则：参数验证
        if not username or not password:
            raise ValueError("Username and password are required")
        
        self.username = username
        self.password = password
    
    def __call__(self, request):
        """处理摘要认证"""
        # 实现摘要认证逻辑
        return request
