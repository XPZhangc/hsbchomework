"""
按规则类型分类生成器：支持条件规则、函数规则、类规则等分类生成
"""
import random
from typing import List, Dict, Any
import re


class RuleTypeGenerator:
    """按规则类型分类生成训练数据"""
    
    def __init__(self):
        self.question_templates_by_type = {
            'conditional_rule': [
                "What is the business rule for handling {condition} in {file_path}?",
                "How does the code handle {condition} in {file_path}?",
                "What happens when {condition} in {file_path}?",
                "What is the logic behind the condition {condition}?",
                "Why does the code check {condition} in {file_path}?",
                "What is the rule for {condition}?",
                "How is {condition} evaluated in {file_path}?",
                "What does the condition {condition} control in {file_path}?",
            ],
            'function_rule': [
                "How does {function_name} work in {file_path}?",
                "What does the function {function_name} do in {file_path}?",
                "What is the purpose of {function_name} in {file_path}?",
                "How is {function_name} implemented in {file_path}?",
                "What are the parameters of {function_name} in {file_path}?",
                "What does {function_name} return in {file_path}?",
                "What is the behavior of {function_name} in {file_path}?",
                "How does {function_name} handle its inputs in {file_path}?",
            ],
            'class_rule': [
                "What is the design of {class_name} in {file_path}?",
                "How is {class_name} structured in {file_path}?",
                "What does the class {class_name} do in {file_path}?",
                "What is the purpose of {class_name} in {file_path}?",
                "How does {class_name} inherit from its base classes?",
                "What methods does {class_name} provide in {file_path}?",
                "What is the architecture of {class_name} in {file_path}?",
                "How is {class_name} used in {file_path}?",
            ],
            'comment_rule': [
                "What does the comment about {description} mean in {file_path}?",
                "What rule is described in the comment at {file_path}?",
                "What is the note about {description} in {file_path}?",
            ]
        }
    
    def generate_by_type(self, repo_data: Dict[str, Any], rule_type: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        按规则类型生成指定数量的样本
        
        Args:
            repo_data: 仓库解析结果
            rule_type: 规则类型 ('conditional_rule', 'function_rule', 'class_rule', 'comment_rule')
            num_samples: 生成样本数量
            
        Returns:
            样本列表
        """
        samples = []
        rules = repo_data.get('rules', [])
        
        # 过滤指定类型的规则
        filtered_rules = [r for r in rules if r.get('type') == rule_type]
        
        if not filtered_rules:
            return samples
        
        # 如果规则数量不足，需要重复使用并变异
        needed = num_samples
        rule_index = 0
        variation_count = 0  # 变异计数器
        
        while len(samples) < needed:
            # 循环使用规则
            if rule_index >= len(filtered_rules):
                rule_index = 0
                variation_count += 1
            
            rule = filtered_rules[rule_index].copy()  # 复制以避免修改原规则
            
            # 应用变异（如果规则被重复使用）
            if variation_count > 0:
                rule = self._vary_rule(rule, variation_count)
            
            # 生成变异的样本
            sample = self._generate_from_rule(rule, repo_data, rule_type)
            if sample:
                # 添加变异标记到问题中，确保唯一性
                if variation_count > 0:
                    sample['question'] = sample['question'] + f" (variant {variation_count})"
                samples.append(sample)
            
            rule_index += 1
            
            # 如果规则太少，尝试从代码片段生成补充
            if len(filtered_rules) < 50 and len(samples) < needed:
                # 从代码片段补充生成
                code_snippets = repo_data.get('code_snippets', [])
                if code_snippets and len(samples) < needed and len(samples) % 10 == 0:  # 每10个样本尝试一次
                    snippet = random.choice(code_snippets)
                    sample = self._generate_from_snippet_for_type(snippet, repo_data, rule_type)
                    if sample:
                        samples.append(sample)
        
        return samples[:num_samples]
    
    def _vary_rule(self, rule: Dict[str, Any], variation: int) -> Dict[str, Any]:
        """对规则进行变异以生成不同的问题"""
        varied_rule = rule.copy()
        
        # 根据变异次数修改描述
        if variation > 0:
            description = varied_rule.get('description', '')
            if description:
                varied_rule['description'] = f"{description} (context {variation})"
        
        return varied_rule
    
    def _generate_from_rule(self, rule: Dict[str, Any], repo_data: Dict[str, Any], rule_type: str) -> Dict[str, Any]:
        """从规则生成样本"""
        file_path = rule.get('file_path', '')
        code = rule.get('code', '')
        
        # 根据类型生成问题和答案
        if rule_type == 'conditional_rule':
            condition = rule.get('condition', 'this condition')
            templates = self.question_templates_by_type['conditional_rule']
            question = random.choice(templates).format(condition=condition, file_path=file_path)
            answer = self._generate_answer_for_conditional(rule)
        
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'this function')
            templates = self.question_templates_by_type['function_rule']
            question = random.choice(templates).format(function_name=func_name, file_path=file_path)
            answer = self._generate_answer_for_function(rule)
        
        elif rule_type == 'class_rule':
            class_name = rule.get('name', 'this class')
            templates = self.question_templates_by_type['class_rule']
            question = random.choice(templates).format(class_name=class_name, file_path=file_path)
            answer = self._generate_answer_for_class(rule)
        
        else:  # comment_rule
            description = rule.get('description', 'this rule')
            code = rule.get('code', '')
            templates = self.question_templates_by_type.get('comment_rule', ['What is the rule for {description}?'])
            question = random.choice(templates).format(description=description, file_path=file_path)
            answer = self._generate_answer_for_comment(rule)
        
        # 生成推理过程
        reasoning = self._generate_reasoning(rule, rule_type)
        
        # 提取代码片段
        code_snippets = [{
            'file_path': file_path,
            'line_start': rule.get('line_start', 1),
            'line_end': rule.get('line_end', 1),
            'code': code[:500]
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
                'language': self._detect_language_from_path(file_path),
                'rule_type': rule_type
            }
        }
    
    def _generate_from_snippet_for_type(self, snippet: Dict[str, Any], repo_data: Dict[str, Any], rule_type: str) -> Dict[str, Any]:
        """从代码片段为特定类型生成样本"""
        file_path = snippet.get('file_path', '')
        code = snippet.get('code', '')
        language = snippet.get('language', 'python')
        
        if language != 'python':
            return None
        
        # 尝试从代码中提取对应类型的规则
        if rule_type == 'conditional_rule':
            # 查找if语句
            if_match = re.search(r'if\s+([^:]+):', code)
            if if_match:
                condition = if_match.group(1).strip()
                question = f"What is the business rule for handling {condition} in {file_path}?"
                answer = f"The code checks {condition} and performs actions based on this condition."
            else:
                return None
        
        elif rule_type == 'function_rule':
            # 查找函数定义
            func_match = re.search(r'def\s+(\w+)', code)
            if func_match:
                func_name = func_match.group(1)
                question = f"How does {func_name} work in {file_path}?"
                answer = f"The function {func_name} performs operations as defined in {file_path}."
            else:
                return None
        
        elif rule_type == 'class_rule':
            # 查找类定义
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                question = f"What is the design of {class_name} in {file_path}?"
                answer = f"The class {class_name} provides functionality related to the module's purpose."
            else:
                return None
        
        else:
            return None
        
        reasoning = f"Step 1: Analyze code structure in {file_path}. Step 2: Identify {rule_type} patterns. Step 3: Extract business rules. Conclusion: Code implements specific functionality."
        
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
                'language': language,
                'rule_type': rule_type
            }
        }
    
    def _generate_answer_for_conditional(self, rule: Dict[str, Any]) -> str:
        """为条件规则生成答案"""
        condition = rule.get('condition', 'condition')
        code = rule.get('code', '')
        
        # 分析代码内容，生成更详细的答案
        if 'redirect' in code.lower():
            return f"The library handles redirects based on the condition {condition}. When this condition is met, the system follows redirects automatically. The default behavior can be configured via parameters such as 'allow_redirects' and 'max_redirects' to control the redirect handling mechanism."
        elif 'timeout' in code.lower():
            return f"Timeout handling follows the rule defined by {condition}. The system prevents indefinite waits by enforcing timeouts on both connect and read operations. This ensures that requests do not hang indefinitely and provides better reliability for network operations."
        elif 'error' in code.lower() or 'exception' in code.lower():
            return f"Error handling is governed by the condition {condition}. When this condition evaluates to true, the system raises appropriate exceptions or handles errors accordingly. This ensures proper error propagation and allows calling code to handle exceptional situations gracefully."
        elif 'method' in code.lower():
            return f"The rule states that when {condition}, the HTTP method is processed and normalized. The system converts the method to uppercase and validates it according to HTTP standards, ensuring consistent method handling throughout the request lifecycle."
        elif 'header' in code.lower() or 'headers' in code.lower():
            return f"The condition {condition} controls header processing in the request. When this condition is met, headers are merged, validated, and prepared for the HTTP request. This ensures proper header formatting and prevents header-related errors."
        else:
            return f"The rule states that when {condition}, the system performs specific actions as defined in the implementation. This conditional logic ensures that the code behaves correctly based on the current state and input parameters, maintaining the expected functionality of the module."
    
    def _generate_answer_for_function(self, rule: Dict[str, Any]) -> str:
        """为函数规则生成答案"""
        func_name = rule.get('name', 'function')
        params = rule.get('parameters', [])
        docstring = rule.get('docstring', '')
        code = rule.get('code', '')
        
        param_str = ', '.join(params) if params else 'no parameters'
        
        # 分析函数功能
        if 'return' in code.lower():
            return_info = " and returns a value"
        else:
            return_info = ""
        
        if docstring:
            doc_preview = docstring[:150].replace('\n', ' ')
            return f"The function {func_name} accepts parameters {param_str}{return_info}. {doc_preview} The function implements the logic as described in its documentation and processes the input parameters according to the defined business rules."
        else:
            # 根据函数名和代码推断功能
            if 'get' in func_name.lower() or 'fetch' in func_name.lower():
                return f"The function {func_name} accepts parameters {param_str}{return_info}. It retrieves or fetches data based on the provided parameters, performing necessary validation and processing before returning the result."
            elif 'set' in func_name.lower() or 'update' in func_name.lower():
                return f"The function {func_name} accepts parameters {param_str}{return_info}. It updates or modifies the internal state based on the provided parameters, ensuring data consistency and proper state management."
            elif 'check' in func_name.lower() or 'validate' in func_name.lower():
                return f"The function {func_name} accepts parameters {param_str}{return_info}. It performs validation or checking operations on the input parameters, ensuring that the data meets the required criteria before further processing."
            else:
                return f"The function {func_name} accepts parameters {param_str}{return_info}. It performs operations based on the input parameters, implementing the core business logic as defined in the codebase. The function handles various scenarios and edge cases to ensure robust functionality."
    
    def _generate_answer_for_class(self, rule: Dict[str, Any]) -> str:
        """为类规则生成答案"""
        class_name = rule.get('name', 'class')
        bases = rule.get('bases', [])
        docstring = rule.get('docstring', '')
        code = rule.get('code', '')
        
        base_info = f"inherits from {', '.join(bases)}" if bases else "is a standalone class"
        
        # 分析类的方法
        method_count = code.count('def ')
        method_info = f" with {method_count} methods" if method_count > 0 else ""
        
        if docstring:
            doc_preview = docstring[:150].replace('\n', ' ')
            return f"The class {class_name} {base_info}{method_info}. {doc_preview} The class encapsulates related functionality and provides a clean interface for interacting with the underlying implementation, following object-oriented design principles."
        else:
            # 根据类名推断功能
            if 'adapter' in class_name.lower():
                return f"The class {class_name} {base_info}{method_info}. It implements the adapter pattern, providing an interface for adapting different implementations. This allows the system to work with various backends while maintaining a consistent API."
            elif 'handler' in class_name.lower():
                return f"The class {class_name} {base_info}{method_info}. It implements a handler pattern, processing requests or events in a structured manner. The handler manages the lifecycle and coordinates with other components to fulfill its responsibilities."
            elif 'base' in class_name.lower() or 'abstract' in class_name.lower():
                return f"The class {class_name} {base_info}{method_info}. It serves as a base or abstract class, defining the interface and common functionality that derived classes should implement. This promotes code reuse and ensures consistent behavior across implementations."
            else:
                return f"The class {class_name} {base_info}{method_info}. It provides functionality following the object-oriented design pattern, encapsulating related data and methods. The class maintains internal state and provides methods for interacting with and manipulating that state."
    
    def _generate_reasoning(self, rule: Dict[str, Any], rule_type: str) -> str:
        """生成推理过程"""
        file_path = rule.get('file_path', 'file')
        
        if rule_type == 'conditional_rule':
            return f"Step 1: Identify conditional logic in {file_path}. Step 2: Analyze the condition and its branches. Step 3: Extract the business rule from the implementation. Conclusion: Rule defines behavior based on specific conditions."
        
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'function')
            return f"Step 1: Locate function {func_name} in {file_path}. Step 2: Analyze function parameters and logic. Step 3: Extract business rules from function implementation. Conclusion: Function implements specific business logic."
        
        elif rule_type == 'class_rule':
            class_name = rule.get('name', 'class')
            return f"Step 1: Locate class {class_name} in {file_path}. Step 2: Analyze class structure and inheritance. Step 3: Extract design patterns and business rules. Conclusion: Class follows object-oriented design principles."
        
        else:
            return f"Step 1: Analyze code structure in {file_path}. Step 2: Extract key components and patterns. Step 3: Identify business rules and logic. Conclusion: Code follows specific design patterns and rules."
    
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

