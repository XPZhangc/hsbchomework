"""
AST提取器：从Python代码中提取业务规则和代码结构
"""
import ast
import re
from typing import List, Dict, Any, Tuple


class ASTExtractor:
    """使用AST解析提取代码规则和结构"""
    
    def __init__(self):
        self.rules = []
        self.functions = []
        self.classes = []
    
    def extract_rules(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """
        从代码中提取业务规则
        
        Args:
            code: 源代码字符串
            file_path: 文件路径
            
        Returns:
            规则列表，每个规则包含类型、位置、代码等信息
        """
        rules = []
        try:
            tree = ast.parse(code)
            
            # 提取条件语句（if-else）作为业务规则
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    rule = self._extract_if_rule(node, code, file_path)
                    if rule:
                        rules.append(rule)
                
                # 提取函数定义
                elif isinstance(node, ast.FunctionDef):
                    func_rule = self._extract_function_rule(node, code, file_path)
                    if func_rule:
                        rules.append(func_rule)
                
                # 提取类定义
                elif isinstance(node, ast.ClassDef):
                    class_rule = self._extract_class_rule(node, code, file_path)
                    if class_rule:
                        rules.append(class_rule)
        
        except SyntaxError:
            # 如果代码无法解析，尝试提取注释中的规则
            rules.extend(self._extract_rules_from_comments(code, file_path))
        
        return rules
    
    def _extract_if_rule(self, node: ast.If, code: str, file_path: str) -> Dict[str, Any]:
        """提取if语句作为业务规则"""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 5
        
        # 获取条件表达式
        try:
            condition = ast.unparse(node.test)
        except AttributeError:
            condition = self._get_condition_str(node.test, code)
        
        # 获取代码片段
        code_lines = code.split('\n')
        code_snippet = '\n'.join(code_lines[start_line-1:end_line])
        
        return {
            'type': 'conditional_rule',
            'condition': condition,
            'file_path': file_path,
            'line_start': start_line,
            'line_end': end_line,
            'code': code_snippet,
            'description': f"Conditional rule: {condition}"
        }
    
    def _extract_function_rule(self, node: ast.FunctionDef, code: str, file_path: str) -> Dict[str, Any]:
        """提取函数定义作为规则"""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        
        code_lines = code.split('\n')
        code_snippet = '\n'.join(code_lines[start_line-1:min(end_line, len(code_lines))])
        
        # 提取参数
        params = [arg.arg for arg in node.args.args]
        
        # 提取文档字符串
        docstring = ast.get_docstring(node) or ""
        
        return {
            'type': 'function_rule',
            'name': node.name,
            'parameters': params,
            'file_path': file_path,
            'line_start': start_line,
            'line_end': end_line,
            'code': code_snippet,
            'docstring': docstring,
            'description': f"Function: {node.name}({', '.join(params)})"
        }
    
    def _extract_class_rule(self, node: ast.ClassDef, code: str, file_path: str) -> Dict[str, Any]:
        """提取类定义作为规则"""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
        
        code_lines = code.split('\n')
        code_snippet = '\n'.join(code_lines[start_line-1:min(end_line, len(code_lines))])
        
        # 提取基类
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                try:
                    bases.append(ast.unparse(base))
                except AttributeError:
                    bases.append(str(type(base).__name__))
        
        docstring = ast.get_docstring(node) or ""
        
        return {
            'type': 'class_rule',
            'name': node.name,
            'bases': bases,
            'file_path': file_path,
            'line_start': start_line,
            'line_end': end_line,
            'code': code_snippet,
            'docstring': docstring,
            'description': f"Class: {node.name}"
        }
    
    def _extract_rules_from_comments(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """从注释中提取规则（当AST解析失败时）"""
        rules = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 查找包含"rule", "TODO", "NOTE"等关键词的注释
            if re.search(r'(rule|TODO|NOTE|FIXME|HACK)', line, re.IGNORECASE):
                rules.append({
                    'type': 'comment_rule',
                    'file_path': file_path,
                    'line_start': i,
                    'line_end': i,
                    'code': line,
                    'description': f"Comment rule at line {i}"
                })
        
        return rules
    
    def _get_condition_str(self, node: ast.AST, code: str) -> str:
        """获取条件表达式的字符串表示"""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Compare):
                left = self._get_condition_str(node.left, code)
                ops = [op.__class__.__name__ for op in node.comparators]
                return f"{left} {ops[0]} ..."
            else:
                return str(type(node).__name__)
        except:
            return "condition"
    
    def extract_code_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """提取代码结构信息"""
        structure = {
            'file_path': file_path,
            'functions': [],
            'classes': [],
            'imports': []
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'params': [arg.arg for arg in node.args.args]
                    })
                
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [base.id for base in node.bases if isinstance(base, ast.Name)]
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        structure['imports'].extend([alias.name for alias in node.names])
                    else:
                        module = node.module or ""
                        structure['imports'].append(module)
        
        except SyntaxError:
            pass
        
        return structure
