"""
仓库解析器：从本地路径或GitHub URL解析代码仓库
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
try:
    import git
except ImportError:
    git = None
from .ast_extractor import ASTExtractor


class RepositoryParser:
    """解析代码仓库，提取代码片段和业务规则"""
    
    def __init__(self):
        self.ast_extractor = ASTExtractor()
        self.supported_extensions = {'.py', '.java', '.js', '.ts', '.cpp', '.c', '.h', '.go', '.rs'}
        self.max_file_size = 100 * 1024  # 100KB
    
    def parse(self, repo_path: str, max_files: int = 50) -> Dict[str, Any]:
        """
        解析仓库
        
        Args:
            repo_path: 本地路径或GitHub URL
            max_files: 最大解析文件数
            
        Returns:
            包含代码片段、规则、元数据的字典
        """
        if repo_path.startswith('http://') or repo_path.startswith('https://'):
            return self._parse_github(repo_path, max_files)
        else:
            return self._parse_local(repo_path, max_files)
    
    def _parse_local(self, repo_path: str, max_files: int) -> Dict[str, Any]:
        """解析本地仓库"""
        repo_path = Path(repo_path)
        
        # 如果是相对路径，尝试基于当前工作目录解析
        if not repo_path.is_absolute():
            # 尝试相对于当前工作目录
            abs_path = Path.cwd() / repo_path
            if abs_path.exists():
                repo_path = abs_path
            else:
                # 尝试相对于脚本所在目录
                script_dir = Path(__file__).parent.parent.parent
                abs_path = script_dir / repo_path
                if abs_path.exists():
                    repo_path = abs_path
        
        if not repo_path.exists():
            raise ValueError(
                f"Repository path does not exist: {repo_path}\n"
                f"提示: 请使用有效的本地路径或GitHub URL，例如:\n"
                f"  - 本地路径: ./your_repo 或 /path/to/repo\n"
                f"  - GitHub URL: https://github.com/psf/requests"
            )
        
        # 尝试检测Git仓库
        repo_url = str(repo_path)
        if git is not None:
            try:
                repo = git.Repo(repo_path)
                repo_url = repo.remotes.origin.url if repo.remotes else str(repo_path)
            except:
                pass
        
        files = []
        code_snippets = []
        rules = []
        structures = []
        
        # 遍历文件（优化：优先处理核心代码文件）
        file_count = 0
        skip_dirs = {'__pycache__', 'node_modules', 'venv', 'env', '.git', '.github', 
                     'tests', 'test', 'docs', 'doc', '.pytest_cache', '.mypy_cache'}
        
        # 收集所有符合条件的文件路径
        all_files = []
        for root, dirs, filenames in os.walk(repo_path):
            # 跳过隐藏目录和常见忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]
            
            # 跳过tests目录（加快速度）
            if 'test' in root.lower() and 'src' not in root.lower():
                dirs[:] = []  # 不遍历tests目录
                continue
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # 检查文件扩展名
                if file_path.suffix not in self.supported_extensions:
                    continue
                
                # 跳过测试文件
                if 'test' in filename.lower() and filename.startswith('test_'):
                    continue
                
                # 检查文件大小（快速检查）
                try:
                    file_size = file_path.stat().st_size
                    if file_size > self.max_file_size or file_size == 0:
                        continue
                except:
                    continue
                
                relative_path = str(file_path.relative_to(repo_path))
                # 优先处理src目录下的文件
                priority = 0 if 'src' in relative_path.lower() or relative_path.startswith('src') else 1
                all_files.append((priority, file_path, relative_path, file_size))
        
        # 按优先级排序，优先处理核心代码
        all_files.sort(key=lambda x: (x[0], x[3]))  # 按优先级和文件大小排序
        
        # 处理文件
        for priority, file_path, relative_path, file_size in all_files:
            if file_count >= max_files:
                break
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # 提取规则（仅Python文件）
                if file_path.suffix == '.py':
                    file_rules = self.ast_extractor.extract_rules(code, relative_path)
                    rules.extend(file_rules)
                    
                    structure = self.ast_extractor.extract_code_structure(code, relative_path)
                    structures.append(structure)
                
                # 保存代码片段
                code_snippets.append({
                    'file_path': relative_path,
                    'code': code,
                    'language': self._detect_language(file_path.suffix)
                })
                
                files.append({
                    'path': relative_path,
                    'size': len(code),
                    'language': self._detect_language(file_path.suffix)
                })
                
                file_count += 1
            
            except Exception as e:
                continue
        
        return {
            'repository': repo_url,
            'files': files,
            'code_snippets': code_snippets,
            'rules': rules,
            'structures': structures,
            'total_files': len(files),
            'total_rules': len(rules)
        }
    
    def _parse_github(self, github_url: str, max_files: int) -> Dict[str, Any]:
        """解析GitHub仓库"""
        # 提取仓库信息
        match = re.match(r'https://github\.com/([^/]+)/([^/]+)', github_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {github_url}")
        
        owner, repo_name = match.groups()
        
        # 使用GitHub API获取文件树
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/main?recursive=1"
        
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code != 200:
                # 尝试master分支
                api_url = api_url.replace('main', 'master')
                response = requests.get(api_url, timeout=10)
            
            if response.status_code != 200:
                # 如果API失败，使用网页抓取
                return self._parse_github_web(github_url, max_files)
            
            tree_data = response.json()
            files = []
            code_snippets = []
            rules = []
            structures = []
            
            file_count = 0
            for item in tree_data.get('tree', []):
                if file_count >= max_files:
                    break
                
                if item['type'] != 'blob':
                    continue
                
                file_path = item['path']
                file_ext = Path(file_path).suffix
                
                if file_ext not in self.supported_extensions:
                    continue
                
                # 跳过测试和示例文件（可选）
                if any(skip in file_path.lower() for skip in ['test', 'example', '__pycache__']):
                    continue
                
                # 获取文件内容
                try:
                    content_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/{file_path}"
                    content_response = requests.get(content_url, timeout=10)
                    
                    if content_response.status_code == 200:
                        code = content_response.text
                        
                        if len(code) > self.max_file_size:
                            continue
                        
                        # 提取规则（仅Python文件）
                        if file_ext == '.py':
                            file_rules = self.ast_extractor.extract_rules(code, file_path)
                            rules.extend(file_rules)
                            
                            structure = self.ast_extractor.extract_code_structure(code, file_path)
                            structures.append(structure)
                        
                        code_snippets.append({
                            'file_path': file_path,
                            'code': code,
                            'language': self._detect_language(file_ext)
                        })
                        
                        files.append({
                            'path': file_path,
                            'size': len(code),
                            'language': self._detect_language(file_ext)
                        })
                        
                        file_count += 1
                
                except Exception as e:
                    continue
            
            return {
                'repository': github_url,
                'files': files,
                'code_snippets': code_snippets,
                'rules': rules,
                'structures': structures,
                'total_files': len(files),
                'total_rules': len(rules)
            }
        
        except Exception as e:
            # 如果API失败，使用网页抓取
            return self._parse_github_web(github_url, max_files)
    
    def _parse_github_web(self, github_url: str, max_files: int) -> Dict[str, Any]:
        """使用网页抓取解析GitHub仓库（备用方法）"""
        # 这是一个简化的实现，实际可能需要更复杂的处理
        return {
            'repository': github_url,
            'files': [],
            'code_snippets': [],
            'rules': [],
            'structures': [],
            'total_files': 0,
            'total_rules': 0,
            'error': 'Web scraping not fully implemented. Please use GitHub API or local repository.'
        }
    
    def _detect_language(self, extension: str) -> str:
        """根据文件扩展名检测语言"""
        lang_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return lang_map.get(extension, 'unknown')
    
    def extract_readme(self, repo_path: str) -> Optional[str]:
        """提取README文件内容"""
        if repo_path.startswith('http://') or repo_path.startswith('https://'):
            # GitHub README
            match = re.match(r'https://github\.com/([^/]+)/([^/]+)', repo_path)
            if match:
                owner, repo_name = match.groups()
                readme_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/README.md"
                try:
                    response = requests.get(readme_url, timeout=10)
                    if response.status_code == 200:
                        return response.text
                except:
                    pass
        else:
            # 本地README
            readme_paths = ['README.md', 'README.txt', 'README.rst']
            for readme_path in readme_paths:
                full_path = Path(repo_path) / readme_path
                if full_path.exists():
                    try:
                        return full_path.read_text(encoding='utf-8', errors='ignore')
                    except:
                        pass
        return None
