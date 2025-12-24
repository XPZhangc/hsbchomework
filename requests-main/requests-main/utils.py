"""
工具函数模块
提供各种辅助功能
"""
from typing import Dict, Any, Optional
import re


def parse_url(url: str) -> Dict[str, str]:
    """
    解析URL
    
    Args:
        url: 要解析的URL
        
    Returns:
        包含协议、主机、路径等信息的字典
    """
    # 条件规则：URL格式验证
    if not url:
        raise ValueError("URL cannot be empty")
    
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    # 简单的URL解析
    parts = url.split('://', 1)
    protocol = parts[0]
    rest = parts[1] if len(parts) > 1 else ''
    
    # 条件规则：主机提取
    if '/' in rest:
        host, path = rest.split('/', 1)
        path = '/' + path
    else:
        host = rest
        path = '/'
    
    return {
        'protocol': protocol,
        'host': host,
        'path': path,
        'url': url
    }


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        
    Returns:
        合并后的字典
    """
    result = dict1.copy()
    
    # 条件规则：合并策略
    for key, value in dict2.items():
        if key in result:
            # 如果键已存在，使用新值覆盖
            result[key] = value
        else:
            # 如果键不存在，添加新键值对
            result[key] = value
    
    return result


def validate_header_name(name: str) -> bool:
    """
    验证HTTP头名称
    
    Args:
        name: 头名称
        
    Returns:
        是否有效
    """
    # 条件规则：头名称验证
    if not name:
        return False
    
    # HTTP头名称不能包含某些字符
    if re.search(r'[^\w\-]', name):
        return False
    
    return True
