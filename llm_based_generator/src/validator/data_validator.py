"""
数据验证器：检查数据质量，过滤低质量样本
"""
from typing import List, Dict, Any, Set
import re


class DataValidator:
    """验证和过滤训练数据"""
    
    def __init__(self, verbose=False):
        self.min_question_length = 10
        self.min_answer_length = 20
        self.min_reasoning_length = 30
        self.verbose = verbose  # 是否显示详细的验证失败信息
    
    def validate(self, samples: List[Dict[str, Any]], return_failures=False) -> List[Dict[str, Any]]:
        """
        验证样本列表
        
        Args:
            samples: 原始样本列表
            return_failures: 是否返回失败样本及其原因
            
        Returns:
            验证后的样本列表，如果return_failures=True，返回(validated, failures)
        """
        validated = []
        failures = []  # 存储失败的样本和原因
        seen_questions = set()
        seen_demands = set()
        
        # 统计失败原因
        failure_stats = {
            'basic': 0,
            'duplicate': 0,
            'logic': 0,
            'code': 0
        }
        
        for i, sample in enumerate(samples):
            failure_reasons = []
            
            # 基本验证
            basic_result, basic_reason = self._validate_basic(sample)
            if not basic_result:
                failure_stats['basic'] += 1
                failure_reasons.append(f"基本验证失败: {basic_reason}")
                if self.verbose:
                    print(f"样本 {i+1} 基本验证失败: {basic_reason}")
                if return_failures:
                    failures.append({
                        'sample': sample,
                        'reasons': failure_reasons
                    })
                continue
            
            # 检查重复（允许变异版本）
            if sample['type'] == 'qa':
                question = sample.get('question', '').lower().strip()
                # 移除变异标记后检查核心问题
                core_question = question.split('(variant')[0].strip()
                if core_question in seen_questions and '(variant' not in question:
                    failure_stats['duplicate'] += 1
                    failure_reasons.append(f"重复问题: {core_question[:50]}...")
                    if self.verbose:
                        print(f"样本 {i+1} 重复问题: {core_question[:50]}...")
                    if return_failures:
                        failures.append({
                            'sample': sample,
                            'reasons': failure_reasons
                        })
                    continue
                seen_questions.add(core_question)
            
            elif sample['type'] == 'design':
                demand = sample.get('demand', '').lower().strip()
                if demand in seen_demands:
                    failure_stats['duplicate'] += 1
                    failure_reasons.append(f"重复需求: {demand[:50]}...")
                    if self.verbose:
                        print(f"样本 {i+1} 重复需求: {demand[:50]}...")
                    if return_failures:
                        failures.append({
                            'sample': sample,
                            'reasons': failure_reasons
                        })
                    continue
                seen_demands.add(demand)
            
            # 验证逻辑正确性
            logic_result, logic_reason = self._validate_logic(sample)
            if not logic_result:
                failure_stats['logic'] += 1
                failure_reasons.append(f"逻辑验证失败: {logic_reason}")
                if self.verbose:
                    print(f"样本 {i+1} 逻辑验证失败: {logic_reason}")
                if return_failures:
                    failures.append({
                        'sample': sample,
                        'reasons': failure_reasons
                    })
                continue
            
            # 验证代码片段
            code_result, code_reason = self._validate_code_snippets(sample)
            if not code_result:
                failure_stats['code'] += 1
                failure_reasons.append(f"代码片段验证失败: {code_reason}")
                if self.verbose:
                    print(f"样本 {i+1} 代码片段验证失败: {code_reason}")
                if return_failures:
                    failures.append({
                        'sample': sample,
                        'reasons': failure_reasons
                    })
                continue
            
            validated.append(sample)
        
        # 输出统计信息
        if self.verbose or sum(failure_stats.values()) > 0:
            print(f"\n验证统计:")
            print(f"  基本验证失败: {failure_stats['basic']}")
            print(f"  重复样本: {failure_stats['duplicate']}")
            print(f"  逻辑验证失败: {failure_stats['logic']}")
            print(f"  代码片段验证失败: {failure_stats['code']}")
        
        if return_failures:
            return validated, failures
        return validated
    
    def _validate_basic(self, sample: Dict[str, Any]) -> tuple:
        """基本验证：检查必需字段和长度"""
        # 检查类型
        if sample.get('type') not in ['qa', 'design']:
            return False, "类型不是qa或design"
        
        # 检查问题/需求
        if sample['type'] == 'qa':
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            if len(question) < self.min_question_length:
                return False, f"问题太短: {len(question)} < {self.min_question_length}"
            if len(answer) < self.min_answer_length:
                return False, f"答案太短: {len(answer)} < {self.min_answer_length}"
        
        elif sample['type'] == 'design':
            demand = sample.get('demand', '')
            scheme = sample.get('scheme', '')
            
            if len(demand) < self.min_question_length:
                return False, f"需求太短: {len(demand)} < {self.min_question_length}"
            if len(scheme) < self.min_answer_length:
                return False, f"方案太短: {len(scheme)} < {self.min_answer_length}"
        
        # 检查推理过程（支持结构化格式）
        reasoning = sample.get('reasoning_trace', {})
        if isinstance(reasoning, dict):
            # 结构化格式验证
            steps = reasoning.get('steps', [])
            if not steps:
                return False, "推理trace缺少steps"
            # 严格要求至少2步（设计方案至少3步）
            min_steps = 3 if sample.get('type') == 'design' else 2
            if len(steps) < min_steps:
                return False, f"推理步骤太少: {len(steps)} < {min_steps}（要求至少{min_steps}步）"
            # 验证每个步骤的结构
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    return False, f"步骤{i+1}不是字典格式"
                if 'action' not in step:
                    return False, f"步骤{i+1}缺少action字段"
                if 'evidence' not in step:
                    return False, f"步骤{i+1}缺少evidence字段"
                if 'conclusion' not in step:
                    return False, f"步骤{i+1}缺少conclusion字段"
            # 检查summary（严格要求）
            summary = reasoning.get('summary', '')
            if not summary or len(summary) < 20:  # 恢复20字符要求
                return False, f"summary缺失或太短: {len(summary) if summary else 0} < 20"
        elif isinstance(reasoning, str):
            # 字符串格式（向后兼容）
            if len(reasoning) < self.min_reasoning_length:
                return False, f"推理文本太短: {len(reasoning)} < {self.min_reasoning_length}"
        else:
            return False, f"推理trace格式错误: {type(reasoning)}"
        
        # 检查元数据（放宽要求，允许为空）
        metadata = sample.get('metadata', {})
        if not metadata.get('repository'):
            # 允许没有repository，但给出警告
            pass
        
        return True, "通过"
    
    def _validate_logic(self, sample: Dict[str, Any]) -> tuple:
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
                    return False, f"答案中缺少问题的关键词: {question_keywords[:3]}"
        
        # 检查设计方案的相关性（严格要求）
        elif sample['type'] == 'design':
            demand = sample.get('demand', '').lower()
            scheme = sample.get('scheme', '').lower()
            
            # 方案应该提及需求中的关键词
            demand_keywords = self._extract_keywords(demand)
            if demand_keywords:
                # 至少有一个关键词应该在方案中出现
                if not any(keyword in scheme for keyword in demand_keywords[:3]):
                    return False, f"方案中缺少需求的关键词: {demand_keywords[:3]}"
        
        return True, "通过"
    
    def _validate_code_snippets(self, sample: Dict[str, Any]) -> tuple:
        """验证代码片段"""
        code_snippets = sample.get('code_snippets', [])
        
        if not code_snippets:
            # 所有类型都必须有代码片段
            return False, f"{sample.get('type', '样本')}缺少代码片段"
        
        for i, snippet in enumerate(code_snippets):
            # 检查必需字段
            if not snippet.get('file_path'):
                return False, f"代码片段{i+1}缺少file_path"
            
            code = snippet.get('code', '')
            if not code or len(code.strip()) < 10:  # 恢复10字符要求
                return False, f"代码片段{i+1}代码太短: {len(code.strip()) if code else 0} < 10"
            
            # 检查行号（放宽要求）
            line_start = snippet.get('line_start', 0)
            line_end = snippet.get('line_end', 0)
            # 允许line_start为0或1，允许line_end >= line_start或为0
            if line_start < 0:
                return False, f"代码片段{i+1}行号无效: line_start={line_start}"
            if line_end > 0 and line_end < line_start:
                return False, f"代码片段{i+1}行号无效: line_end={line_end} < line_start={line_start}"
        
        return True, "通过"
    
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
