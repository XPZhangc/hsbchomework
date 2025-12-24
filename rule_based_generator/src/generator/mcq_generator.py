"""
选择题生成器：生成四选一的选择题测试集
"""
import random
from typing import List, Dict, Any
import re


class MCQGenerator:
    """生成多选题（四选一）测试集"""
    
    def __init__(self):
        self.question_templates = [
            "What is the business rule for {condition} in {file_path}?",
            "How does {function_name} handle {aspect} in {file_path}?",
            "What is the purpose of {class_name} in {file_path}?",
            "What happens when {condition} in {file_path}?",
            "What does the function {function_name} return in {file_path}?",
            "What is the default behavior for {feature} in {file_path}?",
            "How is {component} implemented in {file_path}?",
            "What is the logic behind {rule_name} in {file_path}?",
            "What parameters does {function_name} accept in {file_path}?",
            "What design pattern does {class_name} use in {file_path}?",
        ]
    
    def generate(self, repo_data: Dict[str, Any], num_questions: int = 100) -> List[Dict[str, Any]]:
        """
        生成选择题测试集
        
        Args:
            repo_data: 仓库解析结果
            num_questions: 题目数量
            
        Returns:
            选择题列表
        """
        questions = []
        rules = repo_data.get('rules', [])
        code_snippets = repo_data.get('code_snippets', [])
        
        if not rules and not code_snippets:
            return questions
        
        # 打乱规则顺序
        shuffled_rules = rules.copy()
        random.shuffle(shuffled_rules)
        
        # 从规则生成选择题（循环使用规则以确保生成足够数量）
        rule_index = 0
        variation = 0
        seen_questions = set()
        
        while len(questions) < num_questions:
            if rule_index >= len(shuffled_rules):
                rule_index = 0
                variation += 1
                if variation > 10:  # 最多变异10次
                    break
            
            rule = shuffled_rules[rule_index].copy()
            if variation > 0:
                # 添加变异标记到描述中
                original_desc = rule.get('description', '')
                rule['description'] = f"{original_desc} (variant {variation})"
            
            question = self._generate_from_rule(rule, repo_data, rules, code_snippets)
            if question:
                # 确保问题唯一
                question_text = question['question'].lower().strip()
                if question_text not in seen_questions:
                    seen_questions.add(question_text)
                    questions.append(question)
            
            rule_index += 1
        
        # 从代码片段补充生成
        remaining = num_questions - len(questions)
        if remaining > 0 and code_snippets:
            snippet_variation = 0
            snippet_index = 0
            while len(questions) < num_questions and snippet_variation < 5:
                if snippet_index >= len(code_snippets):
                    snippet_index = 0
                    snippet_variation += 1
                
                snippet = code_snippets[snippet_index].copy()
                question = self._generate_from_snippet(snippet, repo_data, rules, code_snippets)
                if question:
                    question_text = question['question'].lower().strip()
                    if question_text not in seen_questions:
                        seen_questions.add(question_text)
                        questions.append(question)
                
                snippet_index += 1
        
        return questions[:num_questions]
    
    def _generate_from_rule(self, rule: Dict[str, Any], repo_data: Dict[str, Any], 
                            all_rules: List[Dict[str, Any]], 
                            code_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从规则生成选择题"""
        rule_type = rule.get('type', '')
        file_path = rule.get('file_path', '')
        
        # 生成问题
        if rule_type == 'conditional_rule':
            condition = rule.get('condition', 'this condition')
            question = f"What is the business rule for handling {condition} in {file_path}?"
            correct_answer = self._generate_answer_for_conditional(rule)
            wrong_answers = self._generate_wrong_answers_conditional(rule, all_rules, code_snippets)
        
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'this function')
            question = f"How does {func_name} work in {file_path}?"
            correct_answer = self._generate_answer_for_function(rule)
            wrong_answers = self._generate_wrong_answers_function(rule, all_rules, code_snippets)
        
        elif rule_type == 'class_rule':
            class_name = rule.get('name', 'this class')
            question = f"What is the design of {class_name} in {file_path}?"
            correct_answer = self._generate_answer_for_class(rule)
            wrong_answers = self._generate_wrong_answers_class(rule, all_rules, code_snippets)
        
        else:
            description = rule.get('description', 'this rule')
            question = f"What is the rule for {description}?"
            correct_answer = f"The rule states: {description}."
            wrong_answers = self._generate_generic_wrong_answers()
        
        # 生成选项
        options = [correct_answer] + wrong_answers[:3]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_index]
        
        # 生成解释
        explanation = self._generate_explanation(rule, correct_answer)
        
        # 提取代码片段
        code_snippets_list = [{
            'file_path': file_path,
            'line_start': rule.get('line_start', 1),
            'line_end': rule.get('line_end', 1),
            'code': rule.get('code', '')[:500]
        }]
        
        return {
            'type': 'mcq',
            'question': question,
            'options': {
                'A': options[0],
                'B': options[1],
                'C': options[2],
                'D': options[3]
            },
            'correct_answer': correct_letter,
            'explanation': explanation,
            'code_snippets': code_snippets_list,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rule_type': rule_type,
                'language': self._detect_language_from_path(file_path)
            }
        }
    
    def _generate_from_snippet(self, snippet: Dict[str, Any], repo_data: Dict[str, Any],
                               all_rules: List[Dict[str, Any]], 
                               code_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从代码片段生成选择题"""
        file_path = snippet.get('file_path', '')
        code = snippet.get('code', '')
        language = snippet.get('language', 'python')
        
        if language != 'python':
            return None
        
        # 尝试提取函数或类
        func_match = re.search(r'def\s+(\w+)', code)
        class_match = re.search(r'class\s+(\w+)', code)
        
        if func_match:
            func_name = func_match.group(1)
            question = f"What does the function {func_name} do in {file_path}?"
            correct_answer = f"The function {func_name} performs operations as defined in {file_path}."
            wrong_answers = [
                f"The function {func_name} is used for error handling only.",
                f"The function {func_name} manages database connections.",
                f"The function {func_name} handles user authentication."
            ]
        
        elif class_match:
            class_name = class_match.group(1)
            question = f"What is the purpose of class {class_name} in {file_path}?"
            correct_answer = f"The class {class_name} provides functionality related to the module's purpose."
            wrong_answers = [
                f"The class {class_name} is only used for testing purposes.",
                f"The class {class_name} handles network communication exclusively.",
                f"The class {class_name} manages file system operations."
            ]
        else:
            return None
        
        # 生成选项
        options = [correct_answer] + wrong_answers
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_index]
        
        explanation = f"This question tests understanding of the code structure in {file_path}."
        
        code_lines = code.split('\n')
        code_snippets_list = [{
            'file_path': file_path,
            'line_start': 1,
            'line_end': min(20, len(code_lines)),
            'code': '\n'.join(code_lines[:20])[:500]
        }]
        
        return {
            'type': 'mcq',
            'question': question,
            'options': {
                'A': options[0],
                'B': options[1],
                'C': options[2],
                'D': options[3]
            },
            'correct_answer': correct_letter,
            'explanation': explanation,
            'code_snippets': code_snippets_list,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rule_type': 'code_snippet',
                'language': language
            }
        }
    
    def _generate_answer_for_conditional(self, rule: Dict[str, Any]) -> str:
        """为条件规则生成正确答案"""
        condition = rule.get('condition', 'condition')
        code = rule.get('code', '')
        
        if 'redirect' in code.lower():
            return "The library handles redirects automatically, with configurable limits via parameters."
        elif 'timeout' in code.lower():
            return "The system enforces timeouts on connect and read operations to prevent indefinite waits."
        elif 'error' in code.lower() or 'exception' in code.lower():
            return "The system raises appropriate exceptions when the condition is met, ensuring proper error handling."
        else:
            return f"When {condition}, the system performs specific actions as defined in the implementation."
    
    def _generate_answer_for_function(self, rule: Dict[str, Any]) -> str:
        """为函数规则生成正确答案"""
        func_name = rule.get('name', 'function')
        params = rule.get('parameters', [])
        docstring = rule.get('docstring', '')
        
        param_str = ', '.join(params) if params else 'no parameters'
        
        if docstring:
            return f"The function {func_name} accepts {param_str} and implements the logic described in its documentation."
        else:
            return f"The function {func_name} accepts {param_str} and performs operations based on the input parameters."
    
    def _generate_answer_for_class(self, rule: Dict[str, Any]) -> str:
        """为类规则生成正确答案"""
        class_name = rule.get('name', 'class')
        bases = rule.get('bases', [])
        
        base_info = f"inherits from {', '.join(bases)}" if bases else "is a standalone class"
        return f"The class {class_name} {base_info} and provides functionality following object-oriented design principles."
    
    def _generate_wrong_answers_conditional(self, rule: Dict[str, Any], 
                                           all_rules: List[Dict[str, Any]],
                                           code_snippets: List[Dict[str, Any]]) -> List[str]:
        """为条件规则生成错误答案"""
        wrong_answers = [
            "The condition is always false and never executes.",
            "The system ignores the condition and always takes the else branch.",
            "The condition is only checked during initialization, not at runtime.",
            "The system raises an error regardless of the condition value.",
            "The condition is used for logging purposes only, not for control flow.",
        ]
        
        # 从其他规则中提取一些干扰项
        other_conditions = [r.get('condition', '') for r in all_rules 
                          if r.get('type') == 'conditional_rule' and r != rule]
        if other_conditions:
            wrong_answers.append(f"When {random.choice(other_conditions)}, the system behaves differently.")
        
        return wrong_answers
    
    def _generate_wrong_answers_function(self, rule: Dict[str, Any],
                                        all_rules: List[Dict[str, Any]],
                                        code_snippets: List[Dict[str, Any]]) -> List[str]:
        """为函数规则生成错误答案"""
        func_name = rule.get('name', 'function')
        wrong_answers = [
            f"The function {func_name} does not accept any parameters.",
            f"The function {func_name} always returns None regardless of input.",
            f"The function {func_name} is deprecated and should not be used.",
            f"The function {func_name} only works in debug mode.",
            f"The function {func_name} requires administrator privileges to execute.",
        ]
        
        # 从其他函数中提取干扰项
        other_functions = [r.get('name', '') for r in all_rules 
                          if r.get('type') == 'function_rule' and r != rule and r.get('name')]
        if other_functions:
            other_func = random.choice(other_functions)
            wrong_answers.append(f"The function {func_name} is an alias for {other_func}.")
        
        return wrong_answers
    
    def _generate_wrong_answers_class(self, rule: Dict[str, Any],
                                     all_rules: List[Dict[str, Any]],
                                     code_snippets: List[Dict[str, Any]]) -> List[str]:
        """为类规则生成错误答案"""
        class_name = rule.get('name', 'class')
        wrong_answers = [
            f"The class {class_name} is an abstract class with no concrete implementation.",
            f"The class {class_name} is only used for testing and not in production code.",
            f"The class {class_name} inherits from multiple base classes using multiple inheritance.",
            f"The class {class_name} is a singleton class that can only be instantiated once.",
            f"The class {class_name} is deprecated and replaced by a newer implementation.",
        ]
        
        # 从其他类中提取干扰项
        other_classes = [r.get('name', '') for r in all_rules 
                        if r.get('type') == 'class_rule' and r != rule and r.get('name')]
        if other_classes:
            other_class = random.choice(other_classes)
            wrong_answers.append(f"The class {class_name} is a subclass of {other_class}.")
        
        return wrong_answers
    
    def _generate_generic_wrong_answers(self) -> List[str]:
        """生成通用错误答案"""
        return [
            "The rule is not enforced and can be bypassed.",
            "The rule only applies in development environment, not in production.",
            "The rule is optional and can be disabled via configuration.",
            "The rule is deprecated and no longer used in the current version.",
            "The rule is only checked when debugging is enabled.",
        ]
    
    def _generate_explanation(self, rule: Dict[str, Any], correct_answer: str) -> str:
        """生成解释"""
        rule_type = rule.get('type', '')
        file_path = rule.get('file_path', '')
        
        if rule_type == 'conditional_rule':
            return f"The correct answer explains the conditional logic in {file_path}. The condition controls the program flow based on specific criteria, ensuring proper behavior of the system."
        elif rule_type == 'function_rule':
            func_name = rule.get('name', 'function')
            return f"The correct answer describes how {func_name} works in {file_path}. Understanding function behavior is crucial for using the API correctly."
        elif rule_type == 'class_rule':
            class_name = rule.get('name', 'class')
            return f"The correct answer explains the design of {class_name} in {file_path}. The class structure follows object-oriented principles to provide the required functionality."
        else:
            return f"The correct answer describes the rule in {file_path}. Understanding these rules helps in correctly using and extending the codebase."
    
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

