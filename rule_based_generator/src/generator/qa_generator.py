"""
问答对生成器：生成场景1的问答对数据
"""
import random
from typing import List, Dict, Any
import re


class QAGenerator:
    """生成问答对训练数据"""
    
    def __init__(self):
        self.question_templates = [
            "What is the business rule for {rule_name} in {file_path}?",
            "How does {component} handle {aspect}?",
            "What is the rule for {rule_name}?",
            "How does the library handle {scenario}?",
            "What happens when {condition} in {file_path}?",
            "What is the default behavior for {feature}?",
            "How does {function_name} work?",
            "What is the rule for {rule_name} in function {function_name}?",
            "Why does {component} {behavior}?",
            "What is the logic behind {rule_name}?"
        ]
        
        self.answer_templates = [
            "The {component} {behavior} by {mechanism}.",
            "The rule states that {description}.",
            "When {condition}, the system {action}.",
            "The default behavior is {behavior}, which can be changed via {parameter}.",
            "The function {function_name} {description}."
        ]
    
    def generate(self, repo_data: Dict[str, Any], num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        生成问答对样本
        
        Args:
            repo_data: 仓库解析结果
            num_samples: 生成样本数量
            
        Returns:
            问答对样本列表
        """
        samples = []
        rules = repo_data.get('rules', [])
        code_snippets = repo_data.get('code_snippets', [])
        
        if not rules and not code_snippets:
            return samples
        
        # 从规则生成问答对
        rule_samples = min(num_samples // 2, len(rules))
        for rule in random.sample(rules, min(rule_samples, len(rules))):
            sample = self._generate_from_rule(rule, repo_data)
            if sample:
                samples.append(sample)
        
        # 从代码片段生成问答对
        remaining = num_samples - len(samples)
        if remaining > 0 and code_snippets:
            snippet_samples = min(remaining, len(code_snippets))
            for snippet in random.sample(code_snippets, min(snippet_samples, len(code_snippets))):
                sample = self._generate_from_snippet(snippet, repo_data)
                if sample:
                    samples.append(sample)
                    if len(samples) >= num_samples:
                        break
        
        return samples[:num_samples]
    
    def _generate_from_rule(self, rule: Dict[str, Any], repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """从规则生成问答对"""
        rule_type = rule.get('type', '')
        file_path = rule.get('file_path', '')
        code = rule.get('code', '')
        
        # 生成问题
        if rule_type == 'conditional_rule':
            condition = rule.get('condition', 'this condition')
            question = f"What is the business rule for handling {condition} in {file_path}?"
            answer = self._generate_answer_for_conditional(rule)
        
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'this function')
            question = f"How does {func_name} work in {file_path}?"
            answer = self._generate_answer_for_function(rule)
        
        elif rule_type == 'class_rule':
            class_name = rule.get('name', 'this class')
            question = f"What is the design of {class_name} in {file_path}?"
            answer = self._generate_answer_for_class(rule)
        
        else:
            description = rule.get('description', 'this rule')
            question = f"What is the rule for {description}?"
            answer = f"The rule states: {description}."
        
        # 生成推理过程
        reasoning = self._generate_reasoning(rule, 'qa')
        
        # 提取代码片段
        code_snippets = [{
            'file_path': file_path,
            'line_start': rule.get('line_start', 1),
            'line_end': rule.get('line_end', 1),
            'code': code[:500]  # 限制代码长度
        }]
        
        return {
            'type': 'qa',
            'question': question,
            'answer': answer,
            'code_snippets': code_snippets,
            'reasoning_trace': reasoning,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rules_extracted': [rule.get('description', '')],
                'language': self._detect_language_from_path(file_path)
            }
        }
    
    def _generate_from_snippet(self, snippet: Dict[str, Any], repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """从代码片段生成问答对"""
        file_path = snippet.get('file_path', '')
        code = snippet.get('code', '')
        language = snippet.get('language', 'python')
        
        # 分析代码片段，提取关键信息
        if language == 'python':
            # 提取函数名、类名等
            func_match = re.search(r'def\s+(\w+)', code)
            class_match = re.search(r'class\s+(\w+)', code)
            
            if func_match:
                func_name = func_match.group(1)
                question = f"What does the function {func_name} do in {file_path}?"
                answer = f"The function {func_name} performs operations as defined in {file_path}. " \
                        f"It processes the input according to the implementation logic."
            
            elif class_match:
                class_name = class_match.group(1)
                question = f"What is the purpose of class {class_name} in {file_path}?"
                answer = f"The class {class_name} provides functionality related to the module's purpose."
            
            else:
                # 通用问题
                question = f"What is the purpose of the code in {file_path}?"
                answer = f"The code in {file_path} implements specific functionality for the module."
        
        else:
            question = f"What does the code in {file_path} do?"
            answer = f"The code implements functionality in {file_path}."
        
        # 生成推理过程
        reasoning = f"Step 1: Analyze code structure in {file_path}. " \
                   f"Step 2: Identify key components and logic. " \
                   f"Step 3: Extract business rules and patterns. " \
                   f"Conclusion: Code implements specific functionality."
        
        # 提取代码片段（限制长度）
        code_lines = code.split('\n')
        line_start = 1
        line_end = min(20, len(code_lines))
        code_snippet = '\n'.join(code_lines[:line_end])
        
        return {
            'type': 'qa',
            'question': question,
            'answer': answer,
            'code_snippets': [{
                'file_path': file_path,
                'line_start': line_start,
                'line_end': line_end,
                'code': code_snippet[:500]
            }],
            'reasoning_trace': reasoning,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rules_extracted': [],
                'language': language
            }
        }
    
    def _generate_answer_for_conditional(self, rule: Dict[str, Any]) -> str:
        """为条件规则生成答案"""
        condition = rule.get('condition', 'condition')
        code = rule.get('code', '')
        
        # 分析代码中的关键信息
        if 'redirect' in code.lower():
            return f"The library handles redirects based on {condition}. " \
                  f"The default behavior can be configured via parameters."
        elif 'timeout' in code.lower():
            return f"Timeout handling follows the rule: {condition}. " \
                  f"The system prevents indefinite waits by enforcing timeouts."
        elif 'error' in code.lower() or 'exception' in code.lower():
            return f"Error handling is governed by {condition}. " \
                  f"The system raises exceptions or handles errors accordingly."
        else:
            return f"The rule states that when {condition}, the system performs specific actions " \
                  f"as defined in the implementation."
    
    def _generate_answer_for_function(self, rule: Dict[str, Any]) -> str:
        """为函数规则生成答案"""
        func_name = rule.get('name', 'function')
        params = rule.get('parameters', [])
        docstring = rule.get('docstring', '')
        
        if docstring:
            return f"The function {func_name} accepts parameters {', '.join(params)}. " \
                  f"{docstring[:200]}"
        else:
            return f"The function {func_name} accepts parameters {', '.join(params)}. " \
                  f"It performs operations based on the input parameters."
    
    def _generate_answer_for_class(self, rule: Dict[str, Any]) -> str:
        """为类规则生成答案"""
        class_name = rule.get('name', 'class')
        bases = rule.get('bases', [])
        docstring = rule.get('docstring', '')
        
        base_info = f"inherits from {', '.join(bases)}" if bases else "is a standalone class"
        
        if docstring:
            return f"The class {class_name} {base_info}. {docstring[:200]}"
        else:
            return f"The class {class_name} {base_info}. " \
                  f"It provides functionality following the object-oriented design pattern."
    
    def _generate_reasoning(self, rule: Dict[str, Any], sample_type: str) -> str:
        """生成推理过程"""
        file_path = rule.get('file_path', 'file')
        rule_type = rule.get('type', 'rule')
        
        if rule_type == 'conditional_rule':
            return f"Step 1: Identify conditional logic in {file_path}. " \
                  f"Step 2: Analyze the condition and its branches. " \
                  f"Step 3: Extract the business rule from the implementation. " \
                  f"Conclusion: Rule defines behavior based on specific conditions."
        
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'function')
            return f"Step 1: Locate function {func_name} in {file_path}. " \
                  f"Step 2: Analyze function parameters and logic. " \
                  f"Step 3: Extract business rules from function implementation. " \
                  f"Conclusion: Function implements specific business logic."
        
        else:
            return f"Step 1: Analyze code structure in {file_path}. " \
                  f"Step 2: Extract key components and patterns. " \
                  f"Step 3: Identify business rules and logic. " \
                  f"Conclusion: Code follows specific design patterns and rules."
    
    def _detect_language_from_path(self, file_path: str) -> str:
        """从文件路径检测语言"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.java'):
            return 'java'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.ts'):
            return 'typescript'
        else:
            return 'unknown'

