"""
设计方案生成器：生成场景2的设计方案数据
"""
import random
from typing import List, Dict, Any
import re


class DesignGenerator:
    """生成设计方案训练数据"""
    
    def __init__(self):
        self.design_demands = [
            "Design a new authentication module based on the repository architecture.",
            "Propose a design for adding rate-limiting to the architecture.",
            "Design a logging system that integrates with the existing codebase.",
            "Propose a caching mechanism based on the current architecture.",
            "Design an error handling module following the repository patterns.",
            "Propose a design for adding monitoring capabilities.",
            "Design a configuration management system.",
            "Propose a design for adding retry logic.",
            "Design a plugin system based on the existing architecture.",
            "Propose a design for adding validation layer."
        ]
    
    def generate(self, repo_data: Dict[str, Any], num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        生成设计方案样本
        
        Args:
            repo_data: 仓库解析结果
            num_samples: 生成样本数量
            
        Returns:
            设计方案样本列表
        """
        samples = []
        structures = repo_data.get('structures', [])
        code_snippets = repo_data.get('code_snippets', [])
        
        if not structures and not code_snippets:
            return samples
        
        # 分析仓库架构
        architecture = self._analyze_architecture(structures, code_snippets)
        
        # 生成设计方案
        for i in range(min(num_samples, len(self.design_demands))):
            demand = self.design_demands[i]
            sample = self._generate_design_sample(demand, architecture, repo_data)
            if sample:
                samples.append(sample)
        
        return samples
    
    def _analyze_architecture(self, structures: List[Dict[str, Any]], 
                             code_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析仓库架构"""
        architecture = {
            'patterns': [],
            'base_classes': [],
            'common_interfaces': [],
            'extension_points': []
        }
        
        # 分析类结构
        for structure in structures:
            classes = structure.get('classes', [])
            for cls in classes:
                bases = cls.get('bases', [])
                if bases:
                    architecture['base_classes'].extend(bases)
                    architecture['patterns'].append('inheritance')
        
        # 分析函数模式
        for structure in structures:
            functions = structure.get('functions', [])
            if functions:
                architecture['patterns'].append('modular_functions')
        
        # 识别扩展点（如适配器模式）
        for snippet in code_snippets:
            code = snippet.get('code', '')
            if 'adapter' in code.lower() or 'handler' in code.lower():
                architecture['extension_points'].append('adapter_pattern')
            if 'base' in code.lower() and 'class' in code.lower():
                architecture['extension_points'].append('base_class')
        
        return architecture
    
    def _generate_design_sample(self, demand: str, architecture: Dict[str, Any], 
                               repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个设计方案样本"""
        # 根据需求类型生成方案
        if 'authentication' in demand.lower():
            scheme = self._generate_auth_design(architecture, repo_data)
        elif 'rate-limiting' in demand.lower() or 'rate limiting' in demand.lower():
            scheme = self._generate_rate_limit_design(architecture, repo_data)
        elif 'logging' in demand.lower():
            scheme = self._generate_logging_design(architecture, repo_data)
        elif 'caching' in demand.lower():
            scheme = self._generate_cache_design(architecture, repo_data)
        elif 'error' in demand.lower():
            scheme = self._generate_error_handling_design(architecture, repo_data)
        else:
            scheme = self._generate_generic_design(demand, architecture, repo_data)
        
        # 查找相关代码片段
        code_snippets = self._find_relevant_snippets(demand, repo_data)
        
        # 生成推理过程
        reasoning = self._generate_reasoning(demand, architecture, code_snippets)
        
        # 提取存储位置建议
        storage_location = self._suggest_storage_location(demand, code_snippets)
        
        return {
            'type': 'design',
            'demand': demand,
            'scheme': scheme,
            'code_snippets': code_snippets,
            'reasoning_trace': reasoning,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rules_extracted': architecture.get('patterns', []),
                'language': self._detect_primary_language(repo_data),
                'storage_location': storage_location or 'To be determined based on project structure'
            }
        }
    
    def _generate_auth_design(self, architecture: Dict[str, Any], 
                              repo_data: Dict[str, Any]) -> str:
        """生成认证模块设计方案"""
        # 查找现有的auth相关文件
        code_snippets = repo_data.get('code_snippets', [])
        auth_files = [s.get('file_path', '') for s in code_snippets if 'auth' in s.get('file_path', '').lower()]
        
        if 'adapter_pattern' in architecture.get('extension_points', []):
            location_info = f"Location: Extend existing {auth_files[0] if auth_files else 'auth.py'} or create auth_handler.py"
            return f"Integrate an AuthHandler class similar to existing adapters, with methods for token refresh and header injection. Extend Session class to include auth parameter. Follow the adapter pattern used in the codebase for consistency. {location_info}. The AuthHandler should implement methods like authenticate(), refresh_token(), and inject_headers(). It should integrate seamlessly with the existing request/response cycle and maintain compatibility with current authentication mechanisms."
        else:
            location_info = f"Location: Create auth_handler.py in the same directory as {auth_files[0] if auth_files else 'existing auth modules'}"
            return f"Create an AuthHandler class that inherits from a base handler class. {location_info}. Implement methods for token management and authentication. Integrate with the existing session management system. The handler should support multiple authentication schemes including Basic Auth, Bearer tokens, and custom authentication methods. It should handle token expiration and automatic refresh, ensuring secure and seamless authentication throughout the application lifecycle."
    
    def _generate_rate_limit_design(self, architecture: Dict[str, Any], 
                                   repo_data: Dict[str, Any]) -> str:
        """生成限流设计方案"""
        code_snippets = repo_data.get('code_snippets', [])
        adapter_files = [s.get('file_path', '') for s in code_snippets if 'adapter' in s.get('file_path', '').lower() or 'handler' in s.get('file_path', '').lower()]
        
        if 'adapter_pattern' in architecture.get('extension_points', []):
            location_info = f"Location: Create rate_limiter.py in adapters/ directory or extend {adapter_files[0] if adapter_files else 'existing adapter file'}"
            return f"Create a RateLimiter adapter that inherits from HTTPAdapter, with a queue and sleep mechanism before send(). {location_info}. Integrate into Session.mount() following the existing adapter pattern. The rate limiter should support configurable limits per endpoint, sliding window algorithms, and graceful handling of rate limit exceeded scenarios. It should provide metrics and logging capabilities for monitoring rate limit effectiveness."
        else:
            location_info = "Location: Create rate_limiter.py in utils/ or middleware/ directory"
            return f"Design a RateLimiter class with a token bucket algorithm. {location_info}. Integrate it into the request pipeline before sending requests. Maintain compatibility with existing request flow. The rate limiter should support multiple rate limiting strategies including fixed window, sliding window, and token bucket. It should handle concurrent requests efficiently and provide clear error messages when limits are exceeded."
    
    def _generate_logging_design(self, architecture: Dict[str, Any], 
                                repo_data: Dict[str, Any]) -> str:
        """生成日志系统设计方案"""
        location_info = "Location: Create logging_handler.py in utils/ directory or handlers/logging.py"
        return f"Create a LoggingHandler that integrates with the existing handler system. {location_info}. Add logging hooks at key points: request preparation, response handling, and error cases. Use a configurable logging level system. The logging handler should support structured logging with context information, log rotation and retention policies, and integration with external logging services. It should provide detailed request/response logging while maintaining performance and privacy considerations."
    
    def _generate_cache_design(self, architecture: Dict[str, Any], 
                              repo_data: Dict[str, Any]) -> str:
        """生成缓存设计方案"""
        location_info = "Location: Create cache_manager.py in utils/ directory or cache/cache_manager.py"
        return f"Design a CacheManager that can be integrated into the request/response cycle. {location_info}. Implement cache key generation based on request parameters. Add cache invalidation mechanisms and TTL support. The cache manager should support multiple storage backends including in-memory, Redis, and file-based caching. It should implement cache strategies like LRU, LFU, and time-based expiration. The system should handle cache stampede prevention and provide cache statistics for monitoring."
    
    def _generate_error_handling_design(self, architecture: Dict[str, Any], 
                                       repo_data: Dict[str, Any]) -> str:
        """生成错误处理设计方案"""
        code_snippets = repo_data.get('code_snippets', [])
        error_files = [s.get('file_path', '') for s in code_snippets if 'exception' in s.get('file_path', '').lower() or 'error' in s.get('file_path', '').lower()]
        
        if error_files:
            location_info = f"Location: Extend existing {error_files[0]} or create error_handler.py in the same directory"
        else:
            location_info = "Location: Create error_handler.py or extend exceptions.py"
        
        return f"Create an ErrorHandler class following the existing error handling patterns. {location_info}. Implement custom exception types and error recovery mechanisms. Integrate with the existing request/response flow. The error handler should support retry strategies with exponential backoff, error classification and categorization, and comprehensive error logging. It should provide user-friendly error messages while maintaining detailed diagnostic information for developers."
    
    def _generate_generic_design(self, demand: str, architecture: Dict[str, Any], 
                                 repo_data: Dict[str, Any]) -> str:
        """生成通用设计方案"""
        patterns = architecture.get('patterns', [])
        code_snippets = repo_data.get('code_snippets', [])
        
        # 提取目录结构建议
        if code_snippets:
            first_file = code_snippets[0].get('file_path', '')
            if first_file:
                dir_path = '/'.join(first_file.split('/')[:-1]) if '/' in first_file else '.'
                location_info = f"Location: Create new module in {dir_path}/ or extend existing files"
            else:
                location_info = "Location: Create new module file based on project structure"
        else:
            location_info = "Location: To be determined based on project structure"
        
        if 'inheritance' in patterns:
            return f"Design a new module following the inheritance pattern used in the codebase. {location_info}. Create base classes and extend them for specific implementations. Maintain consistency with existing architecture. The module should follow the established design patterns, provide clear interfaces, and ensure backward compatibility. It should integrate seamlessly with existing components while providing extensibility for future enhancements."
        else:
            return f"Design a modular component that integrates with the existing codebase. {location_info}. Follow the established patterns and maintain separation of concerns. Ensure compatibility with existing interfaces. The component should be well-documented, testable, and follow the single responsibility principle. It should provide clear APIs and handle edge cases gracefully while maintaining performance and reliability."
    
    def _find_relevant_snippets(self, demand: str, repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相关代码片段"""
        code_snippets = repo_data.get('code_snippets', [])
        relevant = []
        
        # 根据需求关键词查找相关代码
        keywords = []
        if 'auth' in demand.lower():
            keywords = ['auth', 'authentication', 'token', 'credential']
        elif 'rate' in demand.lower():
            keywords = ['adapter', 'handler', 'send', 'request']
        elif 'log' in demand.lower():
            keywords = ['log', 'logger', 'debug']
        elif 'cache' in demand.lower():
            keywords = ['cache', 'store', 'memory']
        else:
            keywords = ['base', 'class', 'handler']
        
        for snippet in code_snippets[:5]:  # 限制数量
            code = snippet.get('code', '').lower()
            if any(keyword in code for keyword in keywords):
                file_path = snippet.get('file_path', '')
                code_lines = snippet.get('code', '').split('\n')
                relevant.append({
                    'file_path': file_path,
                    'line_start': 1,
                    'line_end': min(20, len(code_lines)),
                    'code': '\n'.join(code_lines[:20])[:500]
                })
        
        # 如果没有找到相关代码，使用第一个代码片段
        if not relevant and code_snippets:
            snippet = code_snippets[0]
            file_path = snippet.get('file_path', '')
            code_lines = snippet.get('code', '').split('\n')
            relevant.append({
                'file_path': file_path,
                'line_start': 1,
                'line_end': min(20, len(code_lines)),
                'code': '\n'.join(code_lines[:20])[:500]
            })
        
        return relevant
    
    def _generate_reasoning(self, demand: str, architecture: Dict[str, Any], 
                           code_snippets: List[Dict[str, Any]]) -> str:
        """生成推理过程"""
        patterns = architecture.get('patterns', [])
        extension_points = architecture.get('extension_points', [])
        base_classes = architecture.get('base_classes', [])
        
        # 提取相关文件路径
        relevant_files = []
        for snippet in code_snippets[:3]:
            file_path = snippet.get('file_path', '')
            if file_path:
                relevant_files.append(file_path)
        
        reasoning = f"Step 1: Analyze existing architecture patterns: {', '.join(set(patterns)) if patterns else 'modular design'}. "
        
        if base_classes:
            reasoning += f"Step 2: Identify base classes for extension: {', '.join(set(base_classes[:3]))}. "
        elif extension_points:
            reasoning += f"Step 2: Identify extension points: {', '.join(set(extension_points))}. "
        else:
            reasoning += "Step 2: Review codebase structure for integration points. "
        
        if relevant_files:
            file_list = ', '.join([f.split('/')[-1] for f in relevant_files[:2]])
            reasoning += f"Step 3: Reference existing files: {file_list}. "
        
        reasoning += "Step 4: Propose extension following existing patterns. "
        reasoning += "Step 5: Ensure compatibility with current architecture. "
        
        # 添加存储位置建议
        storage_location = self._suggest_storage_location(demand, code_snippets)
        if storage_location:
            reasoning += f"Step 6: Recommended storage location: {storage_location}. "
        
        reasoning += "Conclusion: Modular design maintains warehouse architecture consistency."
        
        return reasoning
    
    def _suggest_storage_location(self, demand: str, code_snippets: List[Dict[str, Any]]) -> str:
        """建议方案存储位置"""
        if not code_snippets:
            return None
        
        # 分析需求类型，建议存储位置
        demand_lower = demand.lower()
        
        if 'auth' in demand_lower or 'authentication' in demand_lower:
            # 查找现有的auth相关文件
            for snippet in code_snippets:
                file_path = snippet.get('file_path', '')
                if 'auth' in file_path.lower():
                    return f"Create new file: {file_path.replace('.py', '_extended.py')} or extend existing {file_path}"
            return "Create new file: auth_handler.py or extend existing auth.py"
        
        elif 'rate' in demand_lower or 'limit' in demand_lower:
            for snippet in code_snippets:
                file_path = snippet.get('file_path', '')
                if 'adapter' in file_path.lower() or 'handler' in file_path.lower():
                    return f"Create new file: rate_limiter.py or integrate into {file_path}"
            return "Create new file: rate_limiter.py or adapters/rate_limiter.py"
        
        elif 'log' in demand_lower:
            return "Create new file: logging_handler.py or utils/logging.py"
        
        elif 'cache' in demand_lower:
            return "Create new file: cache_manager.py or utils/cache.py"
        
        elif 'error' in demand_lower or 'exception' in demand_lower:
            for snippet in code_snippets:
                file_path = snippet.get('file_path', '')
                if 'exception' in file_path.lower() or 'error' in file_path.lower():
                    return f"Extend existing {file_path} or create error_handler.py"
            return "Create new file: error_handler.py or extend exceptions.py"
        
        else:
            # 使用第一个代码片段的目录结构
            if code_snippets:
                first_file = code_snippets[0].get('file_path', '')
                if first_file:
                    dir_path = '/'.join(first_file.split('/')[:-1]) if '/' in first_file else '.'
                    file_name = first_file.split('/')[-1] if '/' in first_file else first_file
                    return f"Create new file in {dir_path}/ or extend {file_name}"
        
        return None
    
    def _detect_primary_language(self, repo_data: Dict[str, Any]) -> str:
        """检测主要编程语言"""
        files = repo_data.get('files', [])
        if not files:
            return 'python'
        
        lang_count = {}
        for file_info in files:
            lang = file_info.get('language', 'unknown')
            lang_count[lang] = lang_count.get(lang, 0) + 1
        
        if lang_count:
            return max(lang_count.items(), key=lambda x: x[1])[0]
        return 'python'

