"""
基于LLM的训练数据生成器：使用Qwen模型生成问答对和设计方案
"""
import os
import torch
import time
import re
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random


class LLMGenerator:
    """使用LLM生成训练数据"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化LLM生成器
        
        Args:
            model_path: 模型路径，如果为None则尝试从环境变量或默认位置加载
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        """
        if model_path is None:
            # 尝试从环境变量获取，或使用相对路径
            import os
            model_path = os.getenv('QWEN_MODEL_PATH', 'models/qwen3-4b')
        
        # 解析并验证模型路径
        self.model_path = self._resolve_model_path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _resolve_model_path(self, model_path: str) -> str:
        """
        解析模型路径，确保使用本地路径
        
        Args:
            model_path: 原始模型路径
            
        Returns:
            解析后的绝对路径
        """
        from pathlib import Path
        import os
        
        # 如果是绝对路径且存在，直接返回
        if os.path.isabs(model_path) and Path(model_path).exists():
            return model_path
        
        # 尝试多个可能的路径
        possible_paths = []
        
        # 1. 相对路径（相对于当前工作目录）
        possible_paths.append(Path.cwd() / model_path)
        
        # 2. 相对于脚本所在目录
        script_dir = Path(__file__).parent.parent.parent
        possible_paths.append(script_dir / model_path)
        possible_paths.append(script_dir.parent / model_path)
        
        # 3. 常见的模型目录
        possible_paths.append(Path.cwd() / 'models' / Path(model_path).name)
        possible_paths.append(script_dir.parent / 'models' / Path(model_path).name)
        
        # 4. 如果 model_path 只是目录名（如 'qwen3-4b'），尝试在父目录查找
        if '/' not in model_path and not os.path.isabs(model_path):
            possible_paths.append(Path.cwd().parent / model_path)
            possible_paths.append(script_dir.parent.parent / model_path)
        
        # 检查每个可能的路径
        for path in possible_paths:
            abs_path = path.resolve()
            if abs_path.exists() and abs_path.is_dir():
                # 检查是否包含模型文件
                if (abs_path / 'config.json').exists() or (abs_path / 'tokenizer.json').exists():
                    print(f"找到模型路径: {abs_path}")
                    return str(abs_path)
        
        # 如果都找不到，检查是否是 Hugging Face 模型ID
        if '/' in model_path and not os.path.isabs(model_path) and not Path(model_path).exists():
            # 可能是 Hugging Face 模型ID，但我们需要本地路径
            print(f"\n⚠ 警告: 模型路径 '{model_path}' 不存在")
            print(f"已尝试以下路径:")
            for path in possible_paths:
                print(f"  - {path.resolve()}")
            print(f"\n请使用以下方式之一:")
            print(f"  1. 设置环境变量: export QWEN_MODEL_PATH=/path/to/your/model")
            print(f"  2. 使用命令行参数: --model-path /path/to/your/model")
            print(f"  3. 将模型放在以下位置之一:")
            for path in possible_paths[:2]:
                print(f"     - {path.resolve()}")
            raise FileNotFoundError(
                f"模型路径不存在: {model_path}\n"
                f"请确保模型已下载到本地，或使用 --model-path 指定正确的路径"
            )
        
        # 如果路径存在但可能不是模型目录，仍然返回（让 transformers 处理）
        return model_path
    
    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"正在加载模型: {self.model_path}")
        print(f"使用设备: {self.device}")
        
        # 检查路径是否存在
        from pathlib import Path
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(
                f"模型路径不存在: {self.model_path}\n"
                f"请确保模型已下载到本地，或使用 --model-path 指定正确的路径"
            )
        
        try:
            # 检查关键文件是否存在
            config_file = model_path_obj / 'config.json'
            tokenizer_file = model_path_obj / 'tokenizer.json'
            
            if not config_file.exists() and not tokenizer_file.exists():
                # 尝试查找其他可能的tokenizer文件
                tokenizer_config = model_path_obj / 'tokenizer_config.json'
                if not tokenizer_config.exists():
                    raise FileNotFoundError(
                        f"模型目录中未找到必要的配置文件。\n"
                        f"路径: {self.model_path}\n"
                        f"请确保目录包含 config.json 和 tokenizer.json 等文件"
                    )
            
            # 使用 local_files_only 确保只使用本地文件，避免网络下载
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception as e:
                # 如果 local_files_only 失败，可能是文件不完整，尝试不使用该参数
                print(f"⚠ 警告: 使用 local_files_only 加载失败，尝试允许网络访问...")
                print(f"   如果网络不可用，请确保模型文件完整")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            # 加载模型
            model_kwargs = {
                'trust_remote_code': True,
            }
            
            # 根据设备设置参数
            if self.device == 'cuda' and torch.cuda.is_available():
                model_kwargs['torch_dtype'] = torch.float16
                try:
                    model_kwargs['device_map'] = 'auto'
                except:
                    pass  # 如果device_map不支持，则手动移动
            else:
                model_kwargs['torch_dtype'] = torch.float32
            
            # 尝试使用 local_files_only，如果失败则允许网络访问
            try:
                model_kwargs['local_files_only'] = True
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            except Exception as e:
                # 如果 local_files_only 失败，可能是文件不完整，尝试不使用该参数
                print(f"⚠ 警告: 使用 local_files_only 加载模型失败，尝试允许网络访问...")
                print(f"   如果网络不可用，请确保模型文件完整")
                model_kwargs.pop('local_files_only', None)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            
            # 如果device_map未使用，手动移动模型到设备
            if 'device_map' not in model_kwargs or model_kwargs.get('device_map') is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("模型加载完成")
            
        except FileNotFoundError as e:
            print(f"\n模型加载失败: {str(e)}")
            print(f"\n提示:")
            print(f"  1. 确保模型已下载到本地")
            print(f"  2. 使用 --model-path 指定正确的模型路径")
            print(f"  3. 或设置环境变量 QWEN_MODEL_PATH")
            raise
        except Exception as e:
            print(f"\n模型加载失败: {str(e)}")
            print(f"\n如果错误提示找不到文件，请:")
            print(f"  1. 检查模型路径是否正确: {self.model_path}")
            print(f"  2. 确保模型文件完整（包含 config.json, tokenizer.json 等）")
            print(f"  3. 使用 --model-path 指定正确的路径")
            raise
    
    def generate_qa_pair(self, rule: Dict[str, Any], repo_data: Dict[str, Any], 
                         failure_feedback: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        基于规则生成问答对
        
        Args:
            rule: 规则字典（从AST提取）
            repo_data: 仓库数据
            failure_feedback: 之前的验证失败原因，用于改进生成
            
        Returns:
            问答对字典或None
        """
        rule_type = rule.get('type', '')
        file_path = rule.get('file_path', '')
        code = rule.get('code', '')[:1000]  # 限制代码长度
        
        # 提取函数名和类名（如果存在）
        function_name = rule.get('name', '')
        # 尝试从代码中提取类名（如果规则在类中）
        class_name = rule.get('class_name', '')
        
        # 构建提示词（如果有关键词，添加改进指导）
        prompt = self._build_qa_prompt(rule, code, file_path, rule_type, failure_feedback)
        
        # 生成回答
        response = self._generate_text(prompt, max_length=500)
        
        if not response:
            return None
        
        # 解析响应
        try:
            qa_data = self._parse_qa_response(response, rule, repo_data)
            return qa_data
        except Exception as e:
            print(f"解析QA响应失败: {str(e)}")
            return None
    
    def generate_design_scheme(self, demand: str, repo_data: Dict[str, Any], 
                              failure_feedback: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        生成设计方案
        
        Args:
            demand: 设计需求
            repo_data: 仓库数据
            
        Returns:
            设计方案字典或None
        """
        # 分析仓库架构
        structures = repo_data.get('structures', [])
        code_snippets = repo_data.get('code_snippets', [])[:5]  # 使用前5个代码片段作为上下文
        
        # 构建提示词（如果有关键词，添加改进指导）
        prompt = self._build_design_prompt(demand, structures, code_snippets, failure_feedback)
        
        # 生成回答
        response = self._generate_text(prompt, max_length=800)
        
        if not response:
            return None
        
        # 解析响应
        try:
            design_data = self._parse_design_response(response, demand, repo_data)
            return design_data
        except Exception as e:
            print(f"解析设计方案响应失败: {str(e)}")
            return None
    
    def _build_qa_prompt(self, rule: Dict[str, Any], code: str, file_path: str, rule_type: str,
                        failure_feedback: Optional[str] = None) -> str:
        """构建QA生成提示词"""
        rule_type_map = {
            'conditional_rule': '条件规则',
            'function_rule': '函数规则',
            'class_rule': '类规则',
            'comment_rule': '注释规则'
        }
        
        rule_type_zh = rule_type_map.get(rule_type, rule_type)
        
        # 获取代码行号信息
        line_start = rule.get('line_start', 1)
        line_end = rule.get('line_end', line_start + 10)
        
        # 提取函数名、类名等具体信息
        function_name = rule.get('name', '')
        class_name = rule.get('class_name', '')
        condition = rule.get('condition', '')
        
        # 构建上下文信息
        context_info = []
        if class_name:
            context_info.append(f"所属类: {class_name}")
        if function_name:
            context_info.append(f"函数名: {function_name}")
        if condition:
            context_info.append(f"条件表达式: {condition}")
        context_str = "\n".join(context_info) if context_info else "无"
        
        prompt = f"""你是一个代码分析专家。请基于以下代码片段生成一个高质量的问答对，用于训练代码理解模型。

规则类型: {rule_type_zh}
文件路径: {file_path}
代码行号: {line_start}-{line_end}
上下文信息:
{context_str}

代码片段:
```python
{code}
```

请生成一个问答对，要求：
1. 问题应该深入理解代码的业务逻辑和规则（可以是what/how/why类型）
2. 答案必须明确指出代码的具体位置和名称，避免使用"这段代码"、"该代码"等模糊指代
3. 答案中必须明确提及：
   - 文件路径: {file_path}
   - 代码行号: 第{line_start}行到第{line_end}行
   - {"函数名: " + function_name if function_name else ""}
   - {"类名: " + class_name if class_name else ""}
4. 推理trace必须结构化，包含多个步骤，每个步骤要有动作、证据和结论

请以JSON格式输出，格式如下：
{{
  "question": "问题内容（必须明确指出文件路径和代码位置）",
  "answer": "答案内容，必须明确引用文件路径、行号、函数名/类名等具体信息，例如：'在{file_path}文件的第{line_start}行'，而不是'这段代码'",
  "reasoning_trace": {{
    "steps": [
      {{
        "step": 1,
        "action": "识别代码结构（如：识别条件判断、函数定义等）",
        "evidence": "具体证据来源（如：{file_path}文件第{line_start}行的if语句）",
        "conclusion": "基于证据得出的结论"
      }},
      {{
        "step": 2,
        "action": "分析业务逻辑（如：分析条件表达式的含义）",
        "evidence": "具体证据来源（如：{file_path}文件第{line_start}行条件表达式中的变量名和操作符）",
        "conclusion": "业务规则的具体含义"
      }}
    ],
    "summary": "整体推理过程的总结，说明如何从代码得出最终结论"
  }}
}}

重要要求：
- 问题和答案中必须明确使用文件路径、行号、函数名/类名等具体信息
- 禁止使用"这段代码"、"该代码"、"这里"、"上述代码"等模糊指代
- 使用"在{file_path}文件的第{line_start}行"、"函数{function_name}"、"类{class_name}"等明确表述
- reasoning_trace.steps 必须至少包含2个步骤
- reasoning_trace.summary 必须至少20字符，总结整体推理过程
- 每个步骤的evidence必须明确指出代码位置（文件路径和行号）
- 步骤之间要有逻辑关联，形成推理链
- 只输出JSON，不要其他内容。"""
        
        # 如果有失败反馈，添加到提示词中
        if failure_feedback:
            prompt += f"\n\n重要提示：之前的生成未通过验证，失败原因：{failure_feedback}\n请根据以上原因改进生成，确保满足所有要求。"
        
        return prompt
    
    def _build_design_prompt(self, demand: str, structures: List[Dict[str, Any]], 
                           code_snippets: List[Dict[str, Any]], 
                           failure_feedback: Optional[str] = None) -> str:
        """构建设计方案生成提示词"""
        # 提取架构信息
        architecture_info = self._extract_architecture_info(structures)
        
        # 提取代码上下文
        code_context = "\n\n".join([
            f"文件: {snippet.get('file_path', '')}\n代码:\n{snippet.get('code', '')[:300]}"
            for snippet in code_snippets[:3]
        ])
        
        prompt = f"""你是一个软件架构设计专家。请基于以下代码仓库的架构，设计一个满足需求的新模块。

设计需求: {demand}

仓库架构信息:
{architecture_info}

相关代码示例:
{code_context}

请生成一个详细的设计方案，要求：
1. 方案应该遵循现有代码库的设计模式和架构风格
2. 方案必须明确指出具体的文件路径、类名、函数名等，避免使用"现有代码"、"相关文件"等模糊指代
3. 方案应该详细说明如何集成到现有系统中，明确指出集成点的文件路径和代码位置
4. 方案应该包含具体的实现思路和关键代码结构
5. 推理trace必须结构化，说明设计决策的推理过程

请以JSON格式输出，格式如下：
{{
  "scheme": "详细的设计方案内容，必须明确引用文件路径、类名、函数名等具体信息，例如：'在src/base.py文件的BaseHandler类中'，而不是'现有代码'或'相关文件'",
  "reasoning_trace": {{
    "steps": [
      {{
        "step": 1,
        "action": "分析现有架构（如：识别设计模式、架构风格）",
        "evidence": "具体证据，必须明确指出文件路径和类名（如：src/base.py文件中的BaseHandler类）",
        "conclusion": "现有架构的特点和设计原则"
      }},
      {{
        "step": 2,
        "action": "识别集成点（如：找到可以扩展的基类或接口）",
        "evidence": "具体证据，必须明确指出文件路径和代码位置（如：src/base.py文件第10-20行的BaseHandler类）",
        "conclusion": "集成点的位置和方式"
      }},
      {{
        "step": 3,
        "action": "设计决策（如：选择继承还是组合）",
        "evidence": "具体证据，必须明确指出文件路径和设计模式（如：src/base.py文件中的BaseHandler类采用了策略模式）",
        "conclusion": "设计决策的理由和优势"
      }}
    ],
    "integration_points": [
      {{
        "file": "文件路径（必须是完整的相对路径，如src/base.py）",
        "description": "集成点描述，必须明确指出类名或函数名",
        "rationale": "选择此集成点的理由"
      }}
    ],
    "summary": "整体设计思路和关键决策的总结，必须明确引用文件路径和类名"
  }}
}}

重要要求：
- 禁止使用"现有代码"、"相关文件"、"这段代码"、"该代码"等模糊指代
- 必须明确使用文件路径（如：src/base.py）、类名（如：BaseHandler）、函数名（如：process_request）等具体信息
- reasoning_trace.steps 必须至少包含3个步骤
- reasoning_trace.summary 必须至少20字符，总结整体设计思路
- integration_points 必须包含具体的文件路径和集成方式
- 每个步骤的evidence必须明确指出代码位置（文件路径和行号）或架构特征（类名、函数名）
- 只输出JSON，不要其他内容。"""
        
        # 如果有失败反馈，添加到提示词中
        if failure_feedback:
            prompt += f"\n\n重要提示：之前的生成未通过验证，失败原因：{failure_feedback}\n请根据以上原因改进生成，确保满足所有要求。"
        
        return prompt
    
    def _generate_text(self, prompt: str, max_length: int = 500, temperature: float = 0.7) -> str:
        """
        使用LLM生成文本
        
        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载")
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出（只取新生成的部分）
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"生成文本失败: {str(e)}")
            return ""
    
    def _parse_qa_response(self, response: str, rule: Dict[str, Any], 
                          repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析QA响应"""
        # 尝试提取JSON
        json_str = self._extract_json(response)
        
        question = ""
        answer = ""
        reasoning_trace = None
        
        if json_str:
            try:
                data = json.loads(json_str)
                question = data.get('question', '')
                answer = data.get('answer', '')
                reasoning_data = data.get('reasoning_trace', {})
                
                # 处理结构化的推理trace
                if isinstance(reasoning_data, dict) and 'steps' in reasoning_data:
                    reasoning_trace = reasoning_data
                elif isinstance(reasoning_data, str):
                    # 如果是字符串，尝试转换为结构化格式
                    reasoning_trace = self._convert_text_to_structured_trace(
                        reasoning_data, rule
                    )
                else:
                    reasoning_trace = self._create_default_trace(rule)
            except json.JSONDecodeError as e:
                # JSON解析失败，尝试修复常见问题
                try:
                    # 尝试修复常见的JSON问题：移除尾随逗号、修复引号等
                    json_str_fixed = self._fix_json_common_issues(json_str)
                    data = json.loads(json_str_fixed)
                    question = data.get('question', '')
                    answer = data.get('answer', '')
                    reasoning_data = data.get('reasoning_trace', {})
                    
                    if isinstance(reasoning_data, dict) and 'steps' in reasoning_data:
                        reasoning_trace = reasoning_data
                    else:
                        reasoning_trace = self._create_default_trace(rule)
                except Exception as e2:
                    # 如果修复也失败，从文本中提取
                    question, answer, reasoning_trace = self._parse_qa_from_text(response, rule)
            except Exception as e:
                # 其他异常，从文本中提取
                question, answer, reasoning_trace = self._parse_qa_from_text(response, rule)
        else:
            question, answer, reasoning_trace = self._parse_qa_from_text(response, rule)
        
        # 确保推理trace是结构化的
        if not reasoning_trace or not isinstance(reasoning_trace, dict):
            reasoning_trace = self._create_default_trace(rule)
        
        # 验证推理trace结构
        if 'steps' not in reasoning_trace or not reasoning_trace['steps']:
            reasoning_trace = self._create_default_trace(rule)
        
        # 后处理：替换模糊指代为明确信息
        question = self._replace_vague_references(question, rule)
        answer = self._replace_vague_references(answer, rule)
        
        # 处理推理trace中的模糊指代
        if isinstance(reasoning_trace, dict) and 'steps' in reasoning_trace:
            for step in reasoning_trace['steps']:
                if 'evidence' in step:
                    step['evidence'] = self._replace_vague_references(step['evidence'], rule)
                if 'conclusion' in step:
                    step['conclusion'] = self._replace_vague_references(step['conclusion'], rule)
            if 'summary' in reasoning_trace:
                reasoning_trace['summary'] = self._replace_vague_references(reasoning_trace['summary'], rule)
        
        # 构建代码片段
        code_snippets = [{
            'file_path': rule.get('file_path', ''),
            'line_start': rule.get('line_start', 1),
            'line_end': rule.get('line_end', rule.get('line_start', 1) + 10),
            'code': rule.get('code', '')[:500]
        }]
        
        return {
            'type': 'qa',
            'question': question or f"What is the rule in {rule.get('file_path', '')}?",
            'answer': answer or "Generated by LLM",
            'code_snippets': code_snippets,
            'reasoning_trace': reasoning_trace,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rules_extracted': [rule.get('description', '')],
                'language': self._detect_language_from_path(rule.get('file_path', '')),
                'rule_type': rule.get('type', ''),
                'complexity': self._estimate_complexity(rule),
                'generation_method': 'llm'
            }
        }
    
    def _parse_design_response(self, response: str, demand: str, 
                               repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析设计方案响应"""
        # 尝试提取JSON
        json_str = self._extract_json(response)
        
        scheme = ""
        reasoning_trace = None
        integration_points = []
        
        if json_str:
            try:
                data = json.loads(json_str)
                scheme = data.get('scheme', '')
                reasoning_data = data.get('reasoning_trace', {})
                integration_points = data.get('integration_points', [])
                
                # 处理结构化的推理trace
                if isinstance(reasoning_data, dict) and 'steps' in reasoning_data:
                    reasoning_trace = reasoning_data
                elif isinstance(reasoning_data, str):
                    # 如果是字符串，尝试转换为结构化格式
                    reasoning_trace = self._convert_text_to_structured_trace_design(
                        reasoning_data, demand, repo_data
                    )
                else:
                    reasoning_trace = self._create_default_design_trace(demand, repo_data)
            except json.JSONDecodeError as e:
                # JSON解析失败，尝试修复
                try:
                    json_str_fixed = self._fix_json_common_issues(json_str)
                    data = json.loads(json_str_fixed)
                    scheme = data.get('scheme', '')
                    reasoning_data = data.get('reasoning_trace', {})
                    integration_points = data.get('integration_points', [])
                    
                    if isinstance(reasoning_data, dict) and 'steps' in reasoning_data:
                        reasoning_trace = reasoning_data
                    else:
                        reasoning_trace = self._create_default_design_trace(demand, repo_data)
                except Exception as e2:
                    # 如果修复也失败，使用默认值
                    scheme = response[:500] if len(response) > 500 else response  # 限制长度
                    reasoning_trace = self._create_default_design_trace(demand, repo_data)
                    integration_points = []
            except Exception as e:
                # 其他异常，使用默认值
                scheme = response[:500] if len(response) > 500 else response
                reasoning_trace = self._create_default_design_trace(demand, repo_data)
                integration_points = []
        else:
            scheme = response
            reasoning_trace = self._create_default_design_trace(demand, repo_data)
            integration_points = []
        
        # 确保推理trace是结构化的
        if not reasoning_trace or not isinstance(reasoning_trace, dict):
            reasoning_trace = self._create_default_design_trace(demand, repo_data)
        
        # 验证推理trace结构
        if 'steps' not in reasoning_trace or not reasoning_trace['steps']:
            reasoning_trace = self._create_default_design_trace(demand, repo_data)
        
        # 处理integration_points格式
        if integration_points and isinstance(integration_points[0], str):
            # 如果是字符串列表，转换为对象格式
            integration_points = [
                {'file': ip, 'description': f'Integration point: {ip}', 'rationale': 'Based on architecture analysis'}
                for ip in integration_points[:5]
            ]
        
        # 查找相关代码片段
        code_snippets = self._find_relevant_snippets_for_design(demand, repo_data)
        
        # 后处理：替换设计方案中的模糊指代
        # 从代码片段中提取文件路径信息用于替换
        if code_snippets:
            main_file = code_snippets[0].get('file_path', '')
            # 创建一个临时rule用于替换模糊指代
            temp_rule = {
                'file_path': main_file,
                'line_start': code_snippets[0].get('line_start', 1),
                'line_end': code_snippets[0].get('line_end', 1)
            }
            scheme = self._replace_vague_references_design(scheme, code_snippets)
            
            # 处理推理trace中的模糊指代
            if isinstance(reasoning_trace, dict) and 'steps' in reasoning_trace:
                for step in reasoning_trace['steps']:
                    if 'evidence' in step:
                        step['evidence'] = self._replace_vague_references_design(step['evidence'], code_snippets)
                    if 'conclusion' in step:
                        step['conclusion'] = self._replace_vague_references_design(step['conclusion'], code_snippets)
                if 'summary' in reasoning_trace:
                    reasoning_trace['summary'] = self._replace_vague_references_design(reasoning_trace['summary'], code_snippets)
        
        return {
            'type': 'design',
            'demand': demand,
            'scheme': scheme or "Generated by LLM",
            'code_snippets': code_snippets,
            'reasoning_trace': reasoning_trace,
            'metadata': {
                'repository': repo_data.get('repository', ''),
                'rules_extracted': [ip.get('file', ip) if isinstance(ip, dict) else ip 
                                   for ip in integration_points[:3]],
                'language': self._detect_primary_language(repo_data),
                'generation_method': 'llm'
            }
        }
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON，改进版本，处理格式问题"""
        import re
        
        # 方法1: 尝试找到第一个完整的JSON对象（通过匹配大括号）
        text_cleaned = text.strip()
        
        # 移除可能的markdown代码块标记
        if text_cleaned.startswith('```'):
            # 移除开头的```json或```
            text_cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text_cleaned)
        if text_cleaned.endswith('```'):
            # 移除结尾的```
            text_cleaned = re.sub(r'\n?```\s*$', '', text_cleaned)
        
        text_cleaned = text_cleaned.strip()
        
        # 查找第一个 { 和最后一个 }
        start = text_cleaned.find('{')
        if start < 0:
            return None
        
        # 从第一个 { 开始，找到匹配的 }
        brace_count = 0
        end = start
        
        for i in range(start, len(text_cleaned)):
            char = text_cleaned[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if brace_count == 0 and end > start:
            json_str = text_cleaned[start:end]
            
            # 验证是否是有效的JSON
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # 方法2: 如果方法1失败，尝试使用正则表达式提取
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text_cleaned, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # 方法3: 尝试修复常见的JSON格式问题
        # 移除可能的注释和多余内容
        lines = text_cleaned.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
                if stripped.endswith('}') and stripped.count('{') <= stripped.count('}'):
                    break
        
        if json_lines:
            json_str = '\n'.join(json_lines)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _fix_json_common_issues(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        import re
        
        # 移除尾随逗号（在对象和数组的最后一个元素后）
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 修复单引号为双引号（仅在键和字符串值中）
        # 注意：这可能会误修复，但通常能解决大部分问题
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # 移除可能的注释
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _parse_qa_from_text(self, text: str, rule: Dict[str, Any]) -> tuple:
        """从文本中解析问答对"""
        # 简单的文本解析逻辑
        lines = text.split('\n')
        question = ""
        answer = ""
        reasoning_trace = None
        
        in_answer = False
        in_reasoning = False
        reasoning_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'question' in line.lower() or '问题' in line:
                question = line.split(':', 1)[-1].strip()
            elif 'answer' in line.lower() or '答案' in line:
                answer = line.split(':', 1)[-1].strip()
                in_answer = True
                in_reasoning = False
            elif 'reasoning' in line.lower() or '推理' in line:
                in_answer = False
                in_reasoning = True
                reasoning_text = line.split(':', 1)[-1].strip()
            elif in_answer:
                answer += " " + line
            elif in_reasoning:
                reasoning_text += " " + line
        
        # 将文本推理转换为结构化格式
        if reasoning_text:
            reasoning_trace = self._convert_text_to_structured_trace(reasoning_text, rule)
        else:
            reasoning_trace = self._create_default_trace(rule)
        
        return question, answer, reasoning_trace
    
    def _create_default_trace(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认的结构化推理trace"""
        line_start = rule.get('line_start', 1)
        file_path = rule.get('file_path', '')
        
        return {
            'steps': [
                {
                    'step': 1,
                    'action': '识别代码结构',
                    'evidence': f'{file_path} 第{line_start}行',
                    'conclusion': '识别出代码规则的基本结构'
                },
                {
                    'step': 2,
                    'action': '分析业务逻辑',
                    'evidence': f'代码片段内容',
                    'conclusion': '理解代码的业务规则和功能'
                }
            ],
            'summary': '基于代码分析得出的业务规则和逻辑'
        }
    
    def _create_default_design_trace(self, demand: str, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认的设计方案推理trace"""
        structures = repo_data.get('structures', [])
        sample_file = structures[0].get('file_path', '') if structures else ''
        
        return {
            'steps': [
                {
                    'step': 1,
                    'action': '分析现有架构',
                    'evidence': f'代码仓库中的结构信息',
                    'conclusion': '识别出现有架构的设计模式和风格'
                },
                {
                    'step': 2,
                    'action': '识别集成点',
                    'evidence': f'{sample_file} 中的基类和接口',
                    'conclusion': '找到可以扩展的集成点'
                },
                {
                    'step': 3,
                    'action': '设计决策',
                    'evidence': '现有代码的设计模式',
                    'conclusion': '选择合适的设计方案以保持一致性'
                }
            ],
            'integration_points': [
                {
                    'file': sample_file or 'src/base.py',
                    'description': '通过继承基类实现新功能',
                    'rationale': '保持与现有代码的一致性'
                }
            ],
            'summary': f'基于现有架构设计满足需求的新模块'
        }
    
    def _convert_text_to_structured_trace(self, text: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        """将文本推理转换为结构化格式"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        steps = []
        
        line_start = rule.get('line_start', 1)
        file_path = rule.get('file_path', '')
        
        # 尝试从文本中提取步骤
        step_num = 1
        for i, line in enumerate(lines[:5]):  # 最多5步
            if any(keyword in line.lower() for keyword in ['step', '步骤', 'first', '首先', 'then', '然后']):
                action = line
                evidence = f'{file_path} 第{line_start}行' if i < len(lines) - 1 else '代码分析'
                conclusion = lines[i+1] if i+1 < len(lines) else '基于代码分析得出结论'
                
                steps.append({
                    'step': step_num,
                    'action': action[:100],
                    'evidence': evidence,
                    'conclusion': conclusion[:200]
                })
                step_num += 1
        
        # 如果无法提取步骤，创建默认步骤
        if not steps:
            steps = [
                {
                    'step': 1,
                    'action': '分析代码结构',
                    'evidence': f'{file_path} 第{line_start}行',
                    'conclusion': text[:200] if text else '基于代码分析'
                }
            ]
        
        return {
            'steps': steps,
            'summary': text[:300] if len(text) > 300 else text
        }
    
    def _convert_text_to_structured_trace_design(self, text: str, demand: str, 
                                                  repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """将设计方案文本推理转换为结构化格式"""
        return self._create_default_design_trace(demand, repo_data)
    
    def _estimate_complexity(self, rule: Dict[str, Any]) -> str:
        """估算代码复杂度"""
        code = rule.get('code', '')
        line_count = len(code.split('\n'))
        
        if line_count < 20:
            return 'simple'
        elif line_count < 100:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_architecture_info(self, structures: List[Dict[str, Any]]) -> str:
        """提取架构信息"""
        info_parts = []
        
        for structure in structures[:5]:  # 限制数量
            file_path = structure.get('file_path', '')
            classes = structure.get('classes', [])
            functions = structure.get('functions', [])
            
            if classes or functions:
                info_parts.append(f"文件: {file_path}")
                if classes:
                    info_parts.append(f"  类: {', '.join([c.get('name', '') for c in classes[:3]])}")
                if functions:
                    info_parts.append(f"  函数: {', '.join([f.get('name', '') for f in functions[:5]])}")
        
        return "\n".join(info_parts) if info_parts else "无架构信息"
    
    def _find_relevant_snippets_for_design(self, demand: str, repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为设计方案查找相关代码片段"""
        code_snippets = repo_data.get('code_snippets', [])
        relevant = []
        
        # 根据需求关键词查找
        keywords = []
        if 'auth' in demand.lower():
            keywords = ['auth', 'authentication', 'token']
        elif 'rate' in demand.lower():
            keywords = ['adapter', 'handler', 'send']
        elif 'log' in demand.lower():
            keywords = ['log', 'logger']
        elif 'cache' in demand.lower():
            keywords = ['cache', 'store']
        else:
            keywords = ['base', 'class', 'handler']
        
        for snippet in code_snippets[:5]:
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
        
        # 如果没有找到，使用前3个代码片段
        if not relevant and code_snippets:
            for snippet in code_snippets[:3]:
                file_path = snippet.get('file_path', '')
                code_lines = snippet.get('code', '').split('\n')
                relevant.append({
                    'file_path': file_path,
                    'line_start': 1,
                    'line_end': min(20, len(code_lines)),
                    'code': '\n'.join(code_lines[:20])[:500]
                })
        
        return relevant
    
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
    
    def generate_batch_qa(self, repo_data: Dict[str, Any], num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        批量生成问答对（增强多样性策略）
        
        Args:
            repo_data: 仓库数据
            num_samples: 生成样本数量
            
        Returns:
            问答对列表
        """
        samples = []
        rules = repo_data.get('rules', [])
        
        if not rules:
            print("警告: 没有找到规则，无法生成问答对")
            return samples
        
        # 按规则类型分组
        rules_by_type = {}
        for rule in rules:
            rule_type = rule.get('type', 'unknown')
            if rule_type not in rules_by_type:
                rules_by_type[rule_type] = []
            rules_by_type[rule_type].append(rule)
        
        # 按复杂度分组
        rules_by_complexity = {'simple': [], 'medium': [], 'complex': []}
        for rule in rules:
            complexity = self._estimate_complexity(rule)
            rules_by_complexity[complexity].append(rule)
        
        # 多样性采样策略
        selected_rules = self._diverse_sample_rules(
            rules, rules_by_type, rules_by_complexity, num_samples
        )
        
        print(f"开始生成 {num_samples} 个问答对...")
        print(f"可用规则总数: {len(rules)}")
        print(f"已选择规则数: {len(selected_rules)}")
        if len(selected_rules) < num_samples:
            print(f"警告: 规则数量不足，将循环使用规则以达到目标数量")
        print(f"规则类型分布: {self._get_type_distribution(selected_rules)}")
        print(f"复杂度分布: {self._get_complexity_distribution(selected_rules)}")
        
        start_time = time.time()
        last_estimate_time = start_time
        
        # 如果规则不够，循环使用
        rule_index = 0
        failed_count = 0
        
        while len(samples) < num_samples:
            # 如果规则用完了，重新开始
            if rule_index >= len(selected_rules):
                rule_index = 0
                # 打乱顺序以增加多样性
                random.shuffle(selected_rules)
            
            rule = selected_rules[rule_index]
            rule_index += 1
            current_time = time.time()
            
            # 每10个样本或每30秒更新一次时间预估
            if len(samples) % 10 == 0 or (current_time - last_estimate_time) >= 30:
                elapsed_time = current_time - start_time
                if len(samples) > 0:
                    avg_time_per_sample = elapsed_time / len(samples)
                    remaining_samples = num_samples - len(samples)
                    estimated_remaining = avg_time_per_sample * remaining_samples
                else:
                    estimated_remaining = 0
                
                # 格式化时间
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining)
                
                print(f"生成进度: {len(samples)}/{num_samples} | "
                      f"已用时间: {elapsed_str} | "
                      f"预估剩余: {remaining_str} | "
                      f"失败: {failed_count}", end='\r')
                last_estimate_time = current_time
            else:
                print(f"生成进度: {len(samples)}/{num_samples} | 失败: {failed_count}", end='\r')
            
            try:
                qa_pair = self.generate_qa_pair(rule, repo_data)
                if qa_pair:
                    samples.append(qa_pair)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # 只打印前5个错误
                    print(f"\n生成失败 (规则 {rule_index}): {str(e)}")
                continue
        
        total_time = time.time() - start_time
        total_time_str = self._format_time(total_time)
        print(f"\n完成: 生成了 {len(samples)} 个问答对，失败 {failed_count} 次，总耗时: {total_time_str}")
        return samples
    
    def generate_batch_qa_with_retry(self, repo_data: Dict[str, Any], num_samples: int, 
                                     validator, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        批量生成问答对（带验证反馈循环生成）
        
        Args:
            repo_data: 仓库数据
            num_samples: 目标样本数量
            validator: 验证器实例
            max_retries: 最大重试次数
            
        Returns:
            验证通过的问答对列表
        """
        all_validated = []
        rules = repo_data.get('rules', [])
        
        if not rules:
            print("警告: 没有找到规则，无法生成问答对")
            return all_validated
        
        # 按规则类型分组
        rules_by_type = {}
        for rule in rules:
            rule_type = rule.get('type', 'unknown')
            if rule_type not in rules_by_type:
                rules_by_type[rule_type] = []
            rules_by_type[rule_type].append(rule)
        
        # 按复杂度分组
        rules_by_complexity = {'simple': [], 'medium': [], 'complex': []}
        for rule in rules:
            complexity = self._estimate_complexity(rule)
            rules_by_complexity[complexity].append(rule)
        
        # 多样性采样策略
        selected_rules = self._diverse_sample_rules(
            rules, rules_by_type, rules_by_complexity, num_samples
        )
        
        print(f"开始生成 {num_samples} 个问答对（带验证反馈循环生成）...")
        print(f"可用规则总数: {len(rules)}")
        print(f"已选择规则数: {len(selected_rules)}")
        
        # 存储失败样本和原因，用于改进生成
        failure_history = {}  # {rule_id: [failure_reasons]}
        
        rule_index = 0
        retry_count = 0
        start_time = time.time()
        last_estimate_time = start_time
        total_generated = 0
        total_failed = 0
        
        while len(all_validated) < num_samples and retry_count < max_retries:
            batch_samples = []
            rule_index = 0
            batch_start_time = time.time()
            
            # 生成一批样本
            batch_target = num_samples - len(all_validated)
            while len(batch_samples) < batch_target and rule_index < len(selected_rules) * 2:
                if rule_index >= len(selected_rules):
                    rule_index = 0
                    random.shuffle(selected_rules)
                
                rule = selected_rules[rule_index]
                rule_id = f"{rule.get('file_path', '')}_{rule.get('line_start', 0)}"
                rule_index += 1
                
                # 获取之前的失败原因
                failure_feedback = None
                if rule_id in failure_history:
                    # 获取最近的失败原因列表（最多3个）
                    recent_failures = failure_history[rule_id][-3:]
                    all_reasons = []
                    for failure_list in recent_failures:
                        all_reasons.extend(failure_list)
                    failure_feedback = "; ".join(all_reasons[:5])  # 最多使用5个失败原因
                
                # 显示进度
                current_time = time.time()
                if len(batch_samples) % 5 == 0 or (current_time - last_estimate_time) >= 10:
                    elapsed_time = current_time - start_time
                    if len(all_validated) > 0:
                        avg_time_per_sample = elapsed_time / len(all_validated)
                        remaining_samples = num_samples - len(all_validated)
                        estimated_remaining = avg_time_per_sample * remaining_samples
                    else:
                        estimated_remaining = 0
                    
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time(estimated_remaining)
                    
                    print(f"\r生成进度: 已验证通过 {len(all_validated)}/{num_samples} | "
                          f"本轮生成 {len(batch_samples)}/{batch_target} | "
                          f"已用时间: {elapsed_str} | "
                          f"预估剩余: {remaining_str} | "
                          f"失败: {total_failed}", end='', flush=True)
                    last_estimate_time = current_time
                
                try:
                    qa_pair = self.generate_qa_pair(rule, repo_data, failure_feedback)
                    if qa_pair:
                        batch_samples.append(qa_pair)
                        total_generated += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    total_failed += 1
                    continue
            
            # 验证这批样本
            validated, failures = validator.validate(batch_samples, return_failures=True)
            all_validated.extend(validated)
            
            # 记录失败原因
            for failure in failures:
                sample = failure['sample']
                # 尝试找到对应的rule_id
                code_snippets = sample.get('code_snippets', [])
                if code_snippets:
                    snippet = code_snippets[0]
                    rule_id = f"{snippet.get('file_path', '')}_{snippet.get('line_start', 0)}"
                    if rule_id not in failure_history:
                        failure_history[rule_id] = []
                    failure_history[rule_id].append(failure['reasons'])
            
            retry_count += 1
            remaining = num_samples - len(all_validated)
            
            # 显示本轮验证结果
            batch_validated = len(validated)
            batch_total = len(batch_samples)
            pass_rate = batch_validated / batch_total * 100 if batch_total > 0 else 0
            
            print(f"\r本轮验证: 生成 {batch_total} 个，通过 {batch_validated} 个 ({pass_rate:.1f}%) | "
                  f"累计通过 {len(all_validated)}/{num_samples}", end='', flush=True)
            
            if remaining > 0 and retry_count < max_retries:
                print(f"\n验证后剩余 {remaining} 个，开始第 {retry_count + 1} 轮重试生成...")
        
        total_time = time.time() - start_time
        total_time_str = self._format_time(total_time)
        print(f"\n完成: 验证通过 {len(all_validated)}/{num_samples} 个问答对")
        print(f"总耗时: {total_time_str} | 生成总数: {total_generated} | 失败: {total_failed}")
        if len(all_validated) < num_samples:
            print(f"警告: 未达到目标数量，已重试 {retry_count} 次")
        
        return all_validated[:num_samples]
    
    def _diverse_sample_rules(self, all_rules: List[Dict[str, Any]], 
                             rules_by_type: Dict[str, List[Dict[str, Any]]],
                             rules_by_complexity: Dict[str, List[Dict[str, Any]]],
                             num_samples: int) -> List[Dict[str, Any]]:
        """多样性采样规则"""
        selected = []
        
        # 目标分布：规则类型
        type_distribution = {
            'conditional_rule': 0.30,
            'function_rule': 0.30,
            'class_rule': 0.25,
            'comment_rule': 0.15
        }
        
        # 目标分布：复杂度
        complexity_distribution = {
            'simple': 0.40,
            'medium': 0.45,
            'complex': 0.15
        }
        
        # 按类型采样
        for rule_type, ratio in type_distribution.items():
            if rule_type not in rules_by_type or not rules_by_type[rule_type]:
                continue
            
            type_samples = int(num_samples * ratio)
            available_rules = rules_by_type[rule_type]
            
            if len(available_rules) <= type_samples:
                selected.extend(available_rules)
            else:
                selected.extend(random.sample(available_rules, type_samples))
        
        # 如果样本不够，按复杂度补充
        if len(selected) < num_samples:
            remaining = num_samples - len(selected)
            for complexity, ratio in complexity_distribution.items():
                if len(selected) >= num_samples:
                    break
                
                if complexity not in rules_by_complexity or not rules_by_complexity[complexity]:
                    continue
                
                # 过滤已选中的规则
                available = [r for r in rules_by_complexity[complexity] if r not in selected]
                if not available:
                    continue
                
                complexity_samples = min(int(remaining * ratio), len(available))
                selected.extend(random.sample(available, complexity_samples))
        
        # 如果还不够，随机补充
        if len(selected) < num_samples:
            remaining = num_samples - len(selected)
            available = [r for r in all_rules if r not in selected]
            if available:
                selected.extend(random.sample(available, min(remaining, len(available))))
        
        # 如果超过目标数量，随机选择
        if len(selected) > num_samples:
            selected = random.sample(selected, num_samples)
        
        # 打乱顺序
        random.shuffle(selected)
        
        return selected[:num_samples]
    
    def _get_type_distribution(self, rules: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取规则类型分布"""
        distribution = {}
        for rule in rules:
            rule_type = rule.get('type', 'unknown')
            distribution[rule_type] = distribution.get(rule_type, 0) + 1
        return distribution
    
    def _get_complexity_distribution(self, rules: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取复杂度分布"""
        distribution = {'simple': 0, 'medium': 0, 'complex': 0}
        for rule in rules:
            complexity = self._estimate_complexity(rule)
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def generate_batch_design(self, repo_data: Dict[str, Any], num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        批量生成设计方案（增强多样性策略）
        
        Args:
            repo_data: 仓库数据
            num_samples: 生成样本数量
            
        Returns:
            设计方案列表
        """
        samples = []
        
        # 基础设计需求模板（按类别分类）
        design_demands_by_category = {
            'authentication': [
                "Design a new authentication module based on the repository architecture.",
                "Propose an OAuth2 authentication system that integrates with the existing codebase.",
                "Design a token-based authentication mechanism following repository patterns.",
            ],
            'performance': [
                "Propose a design for adding rate-limiting to the architecture.",
                "Design a caching mechanism based on the current architecture.",
                "Propose a design for adding request throttling capabilities.",
            ],
            'observability': [
                "Design a logging system that integrates with the existing codebase.",
                "Propose a design for adding monitoring capabilities.",
                "Design a distributed tracing system based on the repository architecture.",
            ],
            'resilience': [
                "Design an error handling module following the repository patterns.",
                "Propose a design for adding retry logic with exponential backoff.",
                "Design a circuit breaker pattern implementation.",
            ],
            'configuration': [
                "Design a configuration management system.",
                "Propose a design for environment-based configuration loading.",
                "Design a dynamic configuration update mechanism.",
            ],
            'extensibility': [
                "Design a plugin system based on the existing architecture.",
                "Propose a design for adding validation layer.",
                "Design a middleware system following repository patterns.",
            ]
        }
        
        # 基于仓库特点生成变体需求
        all_demands = []
        for category, demands in design_demands_by_category.items():
            all_demands.extend(demands)
        
        # 分析仓库特点，生成相关需求
        repo_language = self._detect_primary_language(repo_data)
        structures = repo_data.get('structures', [])
        
        # 根据仓库特点添加定制需求
        if structures:
            # 如果有很多类，添加面向对象设计需求
            class_count = sum(len(s.get('classes', [])) for s in structures[:10])
            if class_count > 5:
                all_demands.append("Design a new class hierarchy that extends the existing architecture.")
        
        # 多样性采样需求
        selected_demands = self._diverse_sample_demands(
            all_demands, design_demands_by_category, num_samples, repo_data
        )
        
        print(f"开始生成 {num_samples} 个设计方案...")
        print(f"已选择需求数: {len(selected_demands)}")
        if len(selected_demands) < num_samples:
            print(f"警告: 需求数量不足，将循环使用需求以达到目标数量")
        print(f"需求类别分布: {self._get_demand_category_distribution(selected_demands, design_demands_by_category)}")
        
        start_time = time.time()
        last_estimate_time = start_time
        
        # 如果需求不够，循环使用
        demand_index = 0
        failed_count = 0
        
        while len(samples) < num_samples:
            # 如果需求用完了，重新开始
            if demand_index >= len(selected_demands):
                demand_index = 0
                # 打乱顺序以增加多样性
                random.shuffle(selected_demands)
            
            demand = selected_demands[demand_index]
            demand_index += 1
            current_time = time.time()
            
            # 每5个样本或每30秒更新一次时间预估
            if len(samples) % 5 == 0 or (current_time - last_estimate_time) >= 30:
                elapsed_time = current_time - start_time
                if len(samples) > 0:
                    avg_time_per_sample = elapsed_time / len(samples)
                    remaining_samples = num_samples - len(samples)
                    estimated_remaining = avg_time_per_sample * remaining_samples
                else:
                    estimated_remaining = 0
                
                # 格式化时间
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining)
                
                print(f"生成进度: {len(samples)}/{num_samples} | "
                      f"已用时间: {elapsed_str} | "
                      f"预估剩余: {remaining_str} | "
                      f"失败: {failed_count}", end='\r')
                last_estimate_time = current_time
            else:
                print(f"生成进度: {len(samples)}/{num_samples} | 失败: {failed_count}", end='\r')
            
            try:
                design = self.generate_design_scheme(demand, repo_data)
                if design:
                    samples.append(design)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # 只打印前5个错误
                    print(f"\n生成失败 (需求 {demand_index}): {str(e)}")
                continue
        
        total_time = time.time() - start_time
        total_time_str = self._format_time(total_time)
        print(f"\n完成: 生成了 {len(samples)} 个设计方案，失败 {failed_count} 次，总耗时: {total_time_str}")
        return samples
    
    def generate_batch_design_with_retry(self, repo_data: Dict[str, Any], num_samples: int,
                                        validator, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        批量生成设计方案（带验证反馈循环生成）
        
        Args:
            repo_data: 仓库数据
            num_samples: 目标样本数量
            validator: 验证器实例
            max_retries: 最大重试次数
            
        Returns:
            验证通过的设计方案列表
        """
        all_validated = []
        
        # 基础设计需求模板（按类别分类）
        design_demands_by_category = {
            'authentication': [
                "Design a new authentication module based on the repository architecture.",
                "Propose an OAuth2 authentication system that integrates with the existing codebase.",
                "Design a token-based authentication mechanism following repository patterns.",
            ],
            'performance': [
                "Propose a design for adding rate-limiting to the architecture.",
                "Design a caching mechanism based on the current architecture.",
                "Propose a design for adding request throttling capabilities.",
            ],
            'observability': [
                "Design a logging system that integrates with the existing codebase.",
                "Propose a design for adding monitoring capabilities.",
                "Design a distributed tracing system based on the repository architecture.",
            ],
            'resilience': [
                "Design an error handling module following the repository patterns.",
                "Propose a design for adding retry logic with exponential backoff.",
                "Design a circuit breaker pattern implementation.",
            ],
            'configuration': [
                "Design a configuration management system.",
                "Propose a design for environment-based configuration loading.",
                "Design a dynamic configuration update mechanism.",
            ],
            'extensibility': [
                "Design a plugin system based on the existing architecture.",
                "Propose a design for adding validation layer.",
                "Design a middleware system following repository patterns.",
            ]
        }
        
        # 基于仓库特点生成变体需求
        all_demands = []
        for category, demands in design_demands_by_category.items():
            all_demands.extend(demands)
        
        # 分析仓库特点，生成相关需求
        repo_language = self._detect_primary_language(repo_data)
        structures = repo_data.get('structures', [])
        
        # 根据仓库特点添加定制需求
        if structures:
            class_count = sum(len(s.get('classes', [])) for s in structures[:10])
            if class_count > 5:
                all_demands.append("Design a new class hierarchy that extends the existing architecture.")
        
        # 多样性采样需求
        selected_demands = self._diverse_sample_demands(
            all_demands, design_demands_by_category, num_samples, repo_data
        )
        
        print(f"开始生成 {num_samples} 个设计方案（带验证反馈循环生成）...")
        print(f"已选择需求数: {len(selected_demands)}")
        
        # 存储失败样本和原因
        failure_history = {}  # {demand: [failure_reasons]}
        
        demand_index = 0
        retry_count = 0
        start_time = time.time()
        last_estimate_time = start_time
        total_generated = 0
        total_failed = 0
        
        while len(all_validated) < num_samples and retry_count < max_retries:
            batch_samples = []
            demand_index = 0
            
            # 生成一批样本
            batch_target = num_samples - len(all_validated)
            while len(batch_samples) < batch_target and demand_index < len(selected_demands) * 2:
                if demand_index >= len(selected_demands):
                    demand_index = 0
                    random.shuffle(selected_demands)
                
                demand = selected_demands[demand_index]
                demand_index += 1
                
                # 获取之前的失败原因
                failure_feedback = None
                if demand in failure_history:
                    # 获取最近的失败原因列表（最多3个）
                    recent_failures = failure_history[demand][-3:]
                    all_reasons = []
                    for failure_list in recent_failures:
                        all_reasons.extend(failure_list)
                    failure_feedback = "; ".join(all_reasons[:5])  # 最多使用5个失败原因
                
                # 显示进度
                current_time = time.time()
                if len(batch_samples) % 3 == 0 or (current_time - last_estimate_time) >= 10:
                    elapsed_time = current_time - start_time
                    if len(all_validated) > 0:
                        avg_time_per_sample = elapsed_time / len(all_validated)
                        remaining_samples = num_samples - len(all_validated)
                        estimated_remaining = avg_time_per_sample * remaining_samples
                    else:
                        estimated_remaining = 0
                    
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time(estimated_remaining)
                    
                    print(f"\r生成进度: 已验证通过 {len(all_validated)}/{num_samples} | "
                          f"本轮生成 {len(batch_samples)}/{batch_target} | "
                          f"已用时间: {elapsed_str} | "
                          f"预估剩余: {remaining_str} | "
                          f"失败: {total_failed}", end='', flush=True)
                    last_estimate_time = current_time
                
                try:
                    design = self.generate_design_scheme(demand, repo_data, failure_feedback)
                    if design:
                        batch_samples.append(design)
                        total_generated += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    total_failed += 1
                    continue
            
            # 验证这批样本
            validated, failures = validator.validate(batch_samples, return_failures=True)
            all_validated.extend(validated)
            
            # 记录失败原因
            for failure in failures:
                sample = failure['sample']
                demand = sample.get('demand', '')
                if demand:
                    if demand not in failure_history:
                        failure_history[demand] = []
                    failure_history[demand].append(failure['reasons'])
            
            retry_count += 1
            remaining = num_samples - len(all_validated)
            
            # 显示本轮验证结果
            batch_validated = len(validated)
            batch_total = len(batch_samples)
            pass_rate = batch_validated / batch_total * 100 if batch_total > 0 else 0
            
            print(f"\r本轮验证: 生成 {batch_total} 个，通过 {batch_validated} 个 ({pass_rate:.1f}%) | "
                  f"累计通过 {len(all_validated)}/{num_samples}", end='', flush=True)
            
            if remaining > 0 and retry_count < max_retries:
                print(f"\n验证后剩余 {remaining} 个，开始第 {retry_count + 1} 轮重试生成...")
        
        total_time = time.time() - start_time
        total_time_str = self._format_time(total_time)
        print(f"\n完成: 验证通过 {len(all_validated)}/{num_samples} 个设计方案")
        print(f"总耗时: {total_time_str} | 生成总数: {total_generated} | 失败: {total_failed}")
        if len(all_validated) < num_samples:
            print(f"警告: 未达到目标数量，已重试 {retry_count} 次")
        
        return all_validated[:num_samples]
    
    def _diverse_sample_demands(self, all_demands: List[str],
                               demands_by_category: Dict[str, List[str]],
                               num_samples: int,
                               repo_data: Dict[str, Any]) -> List[str]:
        """多样性采样需求"""
        selected = []
        
        # 目标分布：每个类别大致均匀
        categories = list(demands_by_category.keys())
        samples_per_category = max(1, num_samples // len(categories))
        
        # 按类别采样
        for category in categories:
            if category not in demands_by_category:
                continue
            
            category_demands = demands_by_category[category]
            if len(category_demands) <= samples_per_category:
                selected.extend(category_demands)
            else:
                selected.extend(random.sample(category_demands, samples_per_category))
        
        # 如果不够，生成变体
        if len(selected) < num_samples:
            remaining = num_samples - len(selected)
            for i in range(remaining):
                base_demand = random.choice(all_demands)
                variant_markers = [
                    f"with enhanced error handling",
                    f"optimized for performance",
                    f"with comprehensive logging",
                    f"following SOLID principles",
                    f"with async support",
                ]
                variant = random.choice(variant_markers)
                selected.append(f"{base_demand} {variant}")
        
        # 如果超过，随机选择
        if len(selected) > num_samples:
            selected = random.sample(selected, num_samples)
        
        # 打乱顺序
        random.shuffle(selected)
        
        return selected[:num_samples]
    
    def _get_demand_category_distribution(self, demands: List[str],
                                          demands_by_category: Dict[str, List[str]]) -> Dict[str, int]:
        """获取需求类别分布"""
        distribution = {}
        category_map = {}
        for category, category_demands in demands_by_category.items():
            for demand in category_demands:
                category_map[demand] = category
        
        for demand in demands:
            # 查找需求所属类别
            category = 'other'
            for cat, cat_demands in demands_by_category.items():
                if any(cat_d in demand for cat_d in cat_demands):
                    category = cat
                    break
            
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间为可读字符串"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}小时{minutes}分{secs}秒"
    
    def _replace_vague_references(self, text: str, rule: Dict[str, Any]) -> str:
        """替换文本中的模糊指代为明确信息"""
        if not text:
            return text
        
        file_path = rule.get('file_path', '')
        line_start = rule.get('line_start', 1)
        line_end = rule.get('line_end', line_start)
        function_name = rule.get('name', '')
        class_name = rule.get('class_name', '')
        
        # 获取文件名（不含路径）
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        
        # 模糊指代模式及其替换
        replacements = [
            # "这段代码" -> "在file_path文件的第line_start行"
            (r'这段代码', f'在{file_path}文件的第{line_start}行'),
            (r'该代码', f'{file_path}文件第{line_start}行的代码'),
            (r'这里的代码', f'{file_path}文件第{line_start}行的代码'),
            (r'上述代码', f'{file_path}文件第{line_start}行的代码'),
            (r'此代码', f'{file_path}文件第{line_start}行的代码'),
            (r'代码片段', f'{file_path}文件第{line_start}-{line_end}行的代码'),
            
            # "这个函数" -> "函数function_name"
            (r'这个函数', f'函数{function_name}' if function_name else f'{file_path}文件第{line_start}行的函数'),
            (r'该函数', f'函数{function_name}' if function_name else f'{file_path}文件第{line_start}行的函数'),
            (r'此函数', f'函数{function_name}' if function_name else f'{file_path}文件第{line_start}行的函数'),
            
            # "这个类" -> "类class_name"
            (r'这个类', f'类{class_name}' if class_name else f'{file_path}文件第{line_start}行的类'),
            (r'该类', f'类{class_name}' if class_name else f'{file_path}文件第{line_start}行的类'),
            (r'此类', f'类{class_name}' if class_name else f'{file_path}文件第{line_start}行的类'),
            
            # "这里" -> "在file_path文件的第line_start行"
            (r'这里', f'在{file_path}文件的第{line_start}行'),
            (r'此处', f'在{file_path}文件的第{line_start}行'),
            
            # "文件" -> "file_path文件"
            (r'该文件', f'{file_path}文件'),
            (r'此文件', f'{file_path}文件'),
        ]
        
        result = text
        for pattern, replacement in replacements:
            # 使用正则表达式替换，保持大小写
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # 如果文本中没有明确提到文件路径，在开头添加
        if file_path and file_path not in result and len(result) > 20:
            # 检查是否已经有明确的文件引用
            if not any(keyword in result for keyword in ['文件', 'file', file_name]):
                # 在第一个句号后添加文件信息
                sentences = result.split('。', 1)
                if len(sentences) > 1:
                    result = f"{sentences[0]}。在{file_path}文件的第{line_start}行，{sentences[1]}"
        
        return result
    
    def _replace_vague_references_design(self, text: str, code_snippets: List[Dict[str, Any]]) -> str:
        """替换设计方案文本中的模糊指代为明确信息"""
        if not text or not code_snippets:
            return text
        
        # 使用第一个代码片段作为主要参考
        main_snippet = code_snippets[0]
        file_path = main_snippet.get('file_path', '')
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        
        # 收集所有文件路径
        all_files = [s.get('file_path', '') for s in code_snippets if s.get('file_path')]
        
        # 模糊指代模式及其替换
        replacements = [
            (r'现有代码', f'{file_path}文件中的代码'),
            (r'相关代码', f'{file_path}文件中的代码'),
            (r'这段代码', f'{file_path}文件中的代码'),
            (r'该代码', f'{file_path}文件中的代码'),
            (r'代码库', '代码仓库'),
            (r'现有系统', f'现有系统（主要参考{file_path}文件）'),
            (r'相关文件', f'{", ".join(all_files[:3])}等文件' if all_files else '相关文件'),
            (r'该文件', f'{file_path}文件'),
            (r'此文件', f'{file_path}文件'),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
