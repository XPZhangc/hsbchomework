"""
Requests API模块
提供主要的HTTP请求功能
"""
import time
from typing import Optional, Dict, Any


def request(method: str, url: str, **kwargs) -> Dict[str, Any]:
    """
    发送HTTP请求
    
    Args:
        method: HTTP方法（GET, POST, PUT, DELETE等）
        url: 请求URL
        **kwargs: 其他请求参数
        
    Returns:
        响应对象字典
    """
    timeout = kwargs.get('timeout', 30)
    
    # 条件规则：超时处理
    if timeout is None:
        timeout = 30
    elif timeout < 0:
        raise ValueError("Timeout must be a positive number")
    
    # 条件规则：请求方法验证
    if method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    return {
        'status_code': 200,
        'url': url,
        'method': method.upper()
    }


def get(url: str, params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """
    GET请求
    
    Args:
        url: 请求URL
        params: URL参数
        **kwargs: 其他请求参数
        
    Returns:
        响应对象
    """
    return request('GET', url, params=params, **kwargs)


def post(url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """
    POST请求
    
    Args:
        url: 请求URL
        data: 表单数据
        json: JSON数据
        **kwargs: 其他请求参数
        
    Returns:
        响应对象
    """
    # 条件规则：数据验证
    if data is not None and json is not None:
        raise ValueError("Cannot specify both 'data' and 'json'")
    
    return request('POST', url, data=data, json=json, **kwargs)
