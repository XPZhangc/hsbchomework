"""
数据验证器：检查数据质量，过滤低质量样本
"""
from typing import List, Dict, Any, Set
import re


class DataValidator:
    """验证和过滤训练数据"""
    
    def __init__(self):
        self.min_question_length = 10
        self.min_answer_length = 20
        self.min_reasoning_length = 30
    
    def validate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证样本列表
        
        Args:
            samples: 原始样本列表
            
        Returns:
            验证后的样本列表
        """
        validated = []
        seen_questions = set()
        seen_demands = set()
        
        for sample in samples:
            # 基本验证
            if not self._validate_basic(sample):
                continue
            
            # 检查重复（允许变异版本）
            if sample['type'] == 'qa':
                question = sample.get('question', '').lower().strip()
                # 移除变异标记后检查核心问题
                core_question = question.split('(variant')[0].strip()
                if core_question in seen_questions and '(variant' not in question:
                    continue
                seen_questions.add(core_question)
            
            elif sample['type'] == 'design':
                demand = sample.get('demand', '').lower().strip()
                if demand in seen_demands:
                    continue
                seen_demands.add(demand)
            
            # 验证逻辑正确性
            if not self._validate_logic(sample):
                continue
            
            # 验证代码片段
            if not self._validate_code_snippets(sample):
                continue
            
            validated.append(sample)
        
        return validated
    
    def _validate_basic(self, sample: Dict[str, Any]) -> bool:
        """基本验证：检查必需字段和长度"""
        # 检查类型
        if sample.get('type') not in ['qa', 'design']:
            return False
        
        # 检查问题/需求
        if sample['type'] == 'qa':
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            if len(question) < self.min_question_length:
                return False
            if len(answer) < self.min_answer_length:
                return False
        
        elif sample['type'] == 'design':
            demand = sample.get('demand', '')
            scheme = sample.get('scheme', '')
            
            if len(demand) < self.min_question_length:
                return False
            if len(scheme) < self.min_answer_length:
                return False
        
        # 检查推理过程
        reasoning = sample.get('reasoning_trace', '')
        if len(reasoning) < self.min_reasoning_length:
            return False
        
        # 检查元数据
        metadata = sample.get('metadata', {})
        if not metadata.get('repository'):
            return False
        
        return True
    
    def _validate_logic(self, sample: Dict[str, Any]) -> bool:
        """验证逻辑正确性"""
        # 检查问答对的一致性
        if sample['type'] == 'qa':
            question = sample.get('question', '').lower()
            answer = sample.get('answer', '').lower()
            
            # 答案应该包含问题的关键词
            question_keywords = self._extract_keywords(question)
            if question_keywords:
                # 至少有一个关键词应该在答案中出现
                if not any(keyword in answer for keyword in question_keywords[:3]):
                    return False
        
        # 检查设计方案的相关性
        elif sample['type'] == 'design':
            demand = sample.get('demand', '').lower()
            scheme = sample.get('scheme', '').lower()
            
            # 方案应该提及需求中的关键词
            demand_keywords = self._extract_keywords(demand)
            if demand_keywords:
                if not any(keyword in scheme for keyword in demand_keywords[:2]):
                    return False
        
        return True
    
    def _validate_code_snippets(self, sample: Dict[str, Any]) -> bool:
        """验证代码片段"""
        code_snippets = sample.get('code_snippets', [])
        
        if not code_snippets:
            # 允许没有代码片段的情况
            return True
        
        for snippet in code_snippets:
            # 检查必需字段
            if not snippet.get('file_path'):
                return False
            
            code = snippet.get('code', '')
            if not code or len(code.strip()) < 10:
                return False
            
            # 检查行号
            line_start = snippet.get('line_start', 0)
            line_end = snippet.get('line_end', 0)
            if line_start <= 0 or line_end < line_start:
                return False
        
        return True
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除常见停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'should', 'could', 'may', 'might', 'can', 'this', 'that',
                     'what', 'how', 'why', 'when', 'where', 'for', 'in', 'on',
                     'at', 'to', 'of', 'and', 'or', 'but', 'with', 'from'}
        
        # 提取单词
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        return keywords[:5]  # 返回前5个关键词
    
    def check_diversity(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查数据多样性
        
        Returns:
            多样性统计信息
        """
        stats = {
            'total_samples': len(samples),
            'qa_count': 0,
            'design_count': 0,
            'unique_files': set(),
            'languages': set(),
            'question_types': []
        }
        
        for sample in samples:
            if sample['type'] == 'qa':
                stats['qa_count'] += 1
                question = sample.get('question', '')
                # 分类问题类型
                if 'what' in question.lower():
                    stats['question_types'].append('what')
                elif 'how' in question.lower():
                    stats['question_types'].append('how')
                elif 'why' in question.lower():
                    stats['question_types'].append('why')
            
            elif sample['type'] == 'design':
                stats['design_count'] += 1
            
            # 统计文件
            for snippet in sample.get('code_snippets', []):
                file_path = snippet.get('file_path', '')
                if file_path:
                    stats['unique_files'].add(file_path)
            
            # 统计语言
            language = sample.get('metadata', {}).get('language', '')
            if language:
                stats['languages'].add(language)
        
        # 转换为可序列化格式
        stats['unique_files'] = len(stats['unique_files'])
        stats['languages'] = list(stats['languages'])
        stats['question_type_distribution'] = {
            qtype: stats['question_types'].count(qtype) 
            for qtype in set(stats['question_types'])
        }
        del stats['question_types']
        
        return stats

