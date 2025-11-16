"""Technical validation tools for code snippet validation and technical review."""

import re
import ast
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import docker
from pathlib import Path

logger = logging.getLogger(__name__)


class TechnicalValidationTool:
    """Tool for validating code snippets, technical accuracy, and running technical reviews."""
    
    def __init__(self):
        """Initialize technical validation tool."""
        
        # Supported languages and their validation methods
        self.supported_languages = {
            'python': {
                'file_extension': '.py',
                'syntax_checker': self._validate_python_syntax,
                'linter': 'ruff',
                'formatter': 'black',
                'runner': 'python'
            },
            'javascript': {
                'file_extension': '.js',
                'syntax_checker': self._validate_javascript_syntax,
                'linter': 'eslint',
                'formatter': 'prettier',
                'runner': 'node'
            },
            'typescript': {
                'file_extension': '.ts',
                'syntax_checker': self._validate_typescript_syntax,
                'linter': 'eslint',
                'formatter': 'prettier',
                'runner': 'ts-node'
            },
            'sql': {
                'file_extension': '.sql',
                'syntax_checker': self._validate_sql_syntax,
                'linter': 'sqlfluff',
                'formatter': 'sqlformat',
                'runner': None
            },
            'bash': {
                'file_extension': '.sh',
                'syntax_checker': self._validate_bash_syntax,
                'linter': 'shellcheck',
                'formatter': None,
                'runner': 'bash'
            }
        }
        
        # Docker client for sandboxed execution
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized for sandboxed code execution")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
        
        # Technical concept validation patterns
        self.technical_patterns = {
            'api_endpoints': r'(GET|POST|PUT|DELETE|PATCH)\s+/[^\s]*',
            'code_blocks': r'```(\w+)?\s*([\s\S]*?)```',
            'function_definitions': r'(def|function|const|let|var)\s+\w+\s*\(',
            'class_definitions': r'(class|interface|type)\s+\w+',
            'import_statements': r'(import|from|require|include)\s+[\w\s,{}.*]+',
            'database_queries': r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\s+',
            'version_numbers': r'\b\d+\.\d+(\.\d+)?(-[\w\d]+)?\b',
            'file_paths': r'[/\\]?[\w\-_]+([/\\][\w\-_.]+)*\.[a-zA-Z0-9]+',
            'urls': r'https?://[^\s<>"\']+',
            'environment_variables': r'\$\{?\w+\}?|process\.env\.\w+|os\.environ',
            'configuration_patterns': r'(config|settings|env|environment)\s*[=:]\s*',
            'error_handling': r'(try|catch|except|finally|throw|raise|error)',
            'async_patterns': r'(async|await|promise|then|callback)'
        }
    
    def validate_technical_content(
        self,
        content: str,
        domain_focus: List[str] = None,
        check_code_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Validate technical content for accuracy and completeness.
        
        Args:
            content: Content to validate
            domain_focus: List of domain focuses (ai, it_insurance, etc.)
            check_code_execution: Whether to execute code snippets for validation
            
        Returns:
            Dictionary with technical validation results
        """
        logger.info("Starting technical content validation")
        
        try:
            # Step 1: Extract and analyze code snippets
            code_analysis = self._extract_and_analyze_code(content)
            
            # Step 2: Validate technical concepts and terminology
            concept_validation = self._validate_technical_concepts(content, domain_focus)
            
            # Step 3: Check technical accuracy patterns
            accuracy_check = self._check_technical_accuracy_patterns(content)
            
            # Step 4: Validate code snippets (if any)
            code_validation = {}
            if code_analysis['code_snippets'] and check_code_execution:
                code_validation = self._validate_code_snippets_sync(code_analysis['code_snippets'])
            
            # Step 5: Check for technical depth and completeness
            depth_analysis = self._analyze_technical_depth(content, domain_focus)
            
            # Step 6: Calculate overall technical validation score
            technical_score = self._calculate_technical_validation_score(
                code_analysis,
                concept_validation,
                accuracy_check,
                code_validation,
                depth_analysis
            )
            
            return {
                'technical_validation_score': technical_score,
                'code_analysis': code_analysis,
                'concept_validation': concept_validation,
                'accuracy_check': accuracy_check,
                'code_validation': code_validation,
                'depth_analysis': depth_analysis,
                'recommendations': self._generate_technical_recommendations(
                    technical_score, code_analysis, concept_validation, accuracy_check
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Technical validation failed: {e}")
            return {
                'technical_validation_score': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_and_analyze_code(self, content: str) -> Dict[str, Any]:
        """Extract and analyze code snippets from content."""
        
        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\s*([\s\S]*?)```', content)
        
        code_snippets = []
        languages_found = set()
        
        for language_hint, code in code_blocks:
            if language_hint:
                language_hint = language_hint.lower()
                languages_found.add(language_hint)
            else:
                # Try to detect language
                language_hint = self._detect_code_language(code)
                if language_hint:
                    languages_found.add(language_hint)
            
            code_snippet = {
                'language': language_hint or 'unknown',
                'code': code.strip(),
                'line_count': len(code.strip().split('\n')),
                'character_count': len(code.strip()),
                'snippet_id': f"snippet_{len(code_snippets)}"
            }
            
            code_snippets.append(code_snippet)
        
        # Analyze inline code
        inline_code = re.findall(r'`([^`\n]+)`', content)
        inline_technical_terms = [
            code for code in inline_code 
            if any(pattern in code.lower() for pattern in ['()', '.', '_', '-', '='])
        ]
        
        return {
            'code_snippets': code_snippets,
            'total_code_blocks': len(code_blocks),
            'languages_found': list(languages_found),
            'inline_code_count': len(inline_code),
            'inline_technical_terms': inline_technical_terms[:10],  # Top 10
            'has_executable_code': any(
                snippet['language'] in self.supported_languages 
                for snippet in code_snippets
            ),
            'total_code_lines': sum(snippet['line_count'] for snippet in code_snippets)
        }
    
    def _detect_code_language(self, code: str) -> Optional[str]:
        """Detect programming language from code snippet."""
        
        code_lower = code.lower().strip()
        
        # Python indicators
        if any(pattern in code_lower for pattern in ['def ', 'import ', 'from ', 'print(', 'if __name__']):
            return 'python'
        
        # JavaScript/TypeScript indicators
        if any(pattern in code_lower for pattern in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
            if 'interface ' in code_lower or ': string' in code_lower or ': number' in code_lower:
                return 'typescript'
            return 'javascript'
        
        # SQL indicators
        if any(pattern in code_lower for pattern in ['select ', 'insert ', 'update ', 'delete ', 'create table']):
            return 'sql'
        
        # Bash/Shell indicators
        if any(pattern in code_lower for pattern in ['#!/bin/bash', 'echo ', 'cd ', 'ls ', 'chmod ']):
            return 'bash'
        
        # HTML indicators
        if any(pattern in code for pattern in ['<div', '<span', '<html', '<!DOCTYPE']):
            return 'html'
        
        # CSS indicators
        if re.search(r'\{[^}]*:\s*[^}]*\}', code):
            return 'css'
        
        return None
    
    def _validate_technical_concepts(self, content: str, domain_focus: List[str] = None) -> Dict[str, Any]:
        """Validate technical concepts and terminology for accuracy."""
        
        domain_focus = domain_focus or []
        
        # Extract technical terms and concepts
        technical_patterns_found = {}
        for pattern_name, pattern in self.technical_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                technical_patterns_found[pattern_name] = matches[:5]  # Top 5 matches
        
        # Domain-specific concept validation
        concept_validation_results = {}
        
        for domain in domain_focus:
            if domain == 'ai':
                concept_validation_results['ai'] = self._validate_ai_concepts(content)
            elif domain == 'agentic_ai':
                concept_validation_results['agentic_ai'] = self._validate_agentic_ai_concepts(content)
            elif domain == 'it_insurance':
                concept_validation_results['it_insurance'] = self._validate_it_insurance_concepts(content)
            elif domain == 'ai_software_engineering':
                concept_validation_results['ai_software_engineering'] = self._validate_ai_se_concepts(content)
        
        # Check for technical inaccuracies or outdated information
        accuracy_issues = self._check_for_technical_inaccuracies(content)
        
        # Calculate concept validation score
        validation_score = self._calculate_concept_validation_score(
            technical_patterns_found,
            concept_validation_results,
            accuracy_issues
        )
        
        return {
            'concept_validation_score': validation_score,
            'technical_patterns_found': technical_patterns_found,
            'domain_concept_validation': concept_validation_results,
            'accuracy_issues': accuracy_issues,
            'technical_depth_indicators': len(technical_patterns_found),
            'domain_specific_accuracy': {
                domain: results.get('accuracy_score', 0.8)
                for domain, results in concept_validation_results.items()
            }
        }
    
    def _validate_ai_concepts(self, content: str) -> Dict[str, Any]:
        """Validate AI-related concepts and terminology."""
        
        content_lower = content.lower()
        
        # AI concept indicators
        ai_concepts = {
            'machine_learning': ['machine learning', 'ml model', 'training data', 'algorithm'],
            'deep_learning': ['neural network', 'deep learning', 'tensorflow', 'pytorch'],
            'nlp': ['natural language processing', 'nlp', 'tokenization', 'embedding'],
            'llm': ['large language model', 'llm', 'gpt', 'transformer', 'attention'],
            'ai_ethics': ['bias', 'fairness', 'interpretability', 'responsible ai'],
            'computer_vision': ['computer vision', 'image recognition', 'cnn', 'convolutional']
        }
        
        concepts_found = {}
        for concept_area, terms in ai_concepts.items():
            found_terms = [term for term in terms if term in content_lower]
            if found_terms:
                concepts_found[concept_area] = found_terms
        
        # Check for potential inaccuracies
        potential_issues = []
        
        # Common AI misconceptions
        ai_misconceptions = [
            ('ai is sentient', 'AI systems are not sentient or conscious'),
            ('ai will replace all jobs', 'AI augments human capabilities rather than replacing all jobs'),
            ('ai is perfect', 'AI systems have limitations and can make errors'),
            ('ai learns like humans', 'AI learning differs significantly from human learning')
        ]
        
        for misconception, correction in ai_misconceptions:
            if any(word in content_lower for word in misconception.split()):
                potential_issues.append({
                    'issue': f"Potential misconception detected: {misconception}",
                    'suggestion': correction
                })
        
        accuracy_score = max(0.5, 1.0 - len(potential_issues) * 0.1)
        
        return {
            'concepts_found': concepts_found,
            'concepts_covered_count': len(concepts_found),
            'potential_issues': potential_issues,
            'accuracy_score': round(accuracy_score, 3),
            'domain_depth': 'high' if len(concepts_found) >= 3 else 'medium' if len(concepts_found) >= 1 else 'low'
        }
    
    def _validate_agentic_ai_concepts(self, content: str) -> Dict[str, Any]:
        """Validate agentic AI and multi-agent system concepts."""
        
        content_lower = content.lower()
        
        # Agentic AI concepts
        agentic_concepts = {
            'multi_agent': ['multi-agent', 'agent orchestration', 'agent coordination', 'agent communication'],
            'langgraph': ['langgraph', 'stategraph', 'conditional edges', 'checkpointing'],
            'react_patterns': ['react pattern', 'reasoning', 'acting', 'observation', 'thought'],
            'tool_calling': ['tool calling', 'function calling', 'tool use', 'agent tools'],
            'state_management': ['state management', 'state persistence', 'agent state', 'memory'],
            'agent_architecture': ['agent architecture', 'agent framework', 'autonomous agent']
        }
        
        concepts_found = {}
        for concept_area, terms in agentic_concepts.items():
            found_terms = [term for term in terms if term in content_lower]
            if found_terms:
                concepts_found[concept_area] = found_terms
        
        # Check for LangGraph-specific accuracy
        langgraph_terms = ['langgraph', 'stategraph', 'node', 'edge', 'conditional_edge']
        langgraph_accuracy = sum(1 for term in langgraph_terms if term in content_lower) / len(langgraph_terms)
        
        accuracy_score = min(1.0, 0.7 + langgraph_accuracy * 0.3)
        
        return {
            'concepts_found': concepts_found,
            'concepts_covered_count': len(concepts_found),
            'langgraph_accuracy': round(langgraph_accuracy, 3),
            'accuracy_score': round(accuracy_score, 3),
            'domain_depth': 'high' if len(concepts_found) >= 3 else 'medium' if len(concepts_found) >= 1 else 'low'
        }
    
    def _validate_it_insurance_concepts(self, content: str) -> Dict[str, Any]:
        """Validate IT insurance and insurtech concepts."""
        
        content_lower = content.lower()
        
        # IT Insurance concepts
        insurance_concepts = {
            'cyber_insurance': ['cyber insurance', 'cyber liability', 'data breach', 'cybersecurity coverage'],
            'digital_transformation': ['digital transformation', 'digitalization', 'modernization'],
            'insurtech': ['insurtech', 'insurance technology', 'fintech', 'digital insurance'],
            'regulatory': ['gdpr', 'compliance', 'regulation', 'regulatory requirement'],
            'risk_management': ['risk assessment', 'risk management', 'risk mitigation', 'threat analysis'],
            'claims_processing': ['claims processing', 'claims automation', 'digital claims'],
            'underwriting': ['underwriting', 'risk evaluation', 'policy pricing']
        }
        
        concepts_found = {}
        for concept_area, terms in insurance_concepts.items():
            found_terms = [term for term in terms if term in content_lower]
            if found_terms:
                concepts_found[concept_area] = found_terms
        
        # Check for regulatory accuracy (important for insurance)
        regulatory_terms = ['gdpr', 'hipaa', 'sox', 'pci dss', 'iso 27001']
        regulatory_mentions = sum(1 for term in regulatory_terms if term in content_lower)
        
        accuracy_score = min(1.0, 0.8 + (regulatory_mentions > 0) * 0.2)
        
        return {
            'concepts_found': concepts_found,
            'concepts_covered_count': len(concepts_found),
            'regulatory_mentions': regulatory_mentions,
            'accuracy_score': round(accuracy_score, 3),
            'domain_depth': 'high' if len(concepts_found) >= 3 else 'medium' if len(concepts_found) >= 1 else 'low'
        }
    
    def _validate_ai_se_concepts(self, content: str) -> Dict[str, Any]:
        """Validate AI software engineering concepts."""
        
        content_lower = content.lower()
        
        # AI Software Engineering concepts
        ai_se_concepts = {
            'ai_development': ['ai development', 'model training', 'model deployment', 'mlops'],
            'code_generation': ['code generation', 'ai coding', 'copilot', 'code completion'],
            'testing_ai': ['ai testing', 'model validation', 'unit testing', 'integration testing'],
            'ai_frameworks': ['tensorflow', 'pytorch', 'huggingface', 'langchain', 'llamaindex'],
            'deployment': ['model deployment', 'containerization', 'kubernetes', 'docker'],
            'monitoring': ['model monitoring', 'performance monitoring', 'ai observability']
        }
        
        concepts_found = {}
        for concept_area, terms in ai_se_concepts.items():
            found_terms = [term for term in terms if term in content_lower]
            if found_terms:
                concepts_found[concept_area] = found_terms
        
        # Check for framework accuracy
        framework_accuracy = self._check_framework_references(content_lower)
        
        accuracy_score = min(1.0, 0.7 + framework_accuracy * 0.3)
        
        return {
            'concepts_found': concepts_found,
            'concepts_covered_count': len(concepts_found),
            'framework_accuracy': framework_accuracy,
            'accuracy_score': round(accuracy_score, 3),
            'domain_depth': 'high' if len(concepts_found) >= 3 else 'medium' if len(concepts_found) >= 1 else 'low'
        }
    
    def _check_framework_references(self, content_lower: str) -> float:
        """Check accuracy of framework and library references."""
        
        framework_patterns = {
            'tensorflow': ['tensorflow', 'tf.', 'keras'],
            'pytorch': ['pytorch', 'torch.', 'torchvision'],
            'langchain': ['langchain', 'langchain_'],
            'huggingface': ['huggingface', 'transformers', 'datasets'],
            'fastapi': ['fastapi', 'uvicorn', '@app.'],
            'docker': ['docker', 'dockerfile', 'containerize']
        }
        
        accurate_references = 0
        total_references = 0
        
        for framework, patterns in framework_patterns.items():
            if framework in content_lower:
                total_references += 1
                # Check if other related terms are also present (indicates accurate usage)
                related_found = sum(1 for pattern in patterns[1:] if pattern in content_lower)
                if related_found > 0:
                    accurate_references += 1
        
        return accurate_references / total_references if total_references > 0 else 1.0
    
    def _check_technical_accuracy_patterns(self, content: str) -> Dict[str, Any]:
        """Check for common technical accuracy issues."""
        
        accuracy_issues = []
        content_lower = content.lower()
        
        # Common technical inaccuracies
        inaccuracy_patterns = [
            # AI/ML inaccuracies
            ('ai is 100% accurate', 'AI systems have inherent limitations and error rates'),
            ('machine learning guarantees', 'Machine learning provides probabilistic outputs'),
            ('neural networks think', 'Neural networks process information but do not think'),
            
            # Programming inaccuracies
            ('python is compiled', 'Python is an interpreted language'),
            ('javascript on server', 'Check context - JavaScript can run server-side with Node.js'),
            
            # Security inaccuracies
            ('passwords in plain text', 'Passwords should never be stored in plain text'),
            ('http for sensitive data', 'Use HTTPS for sensitive data transmission'),
            
            # General tech inaccuracies
            ('cloud is always cheaper', 'Cloud costs depend on usage patterns and requirements'),
            ('blockchain solves everything', 'Blockchain has specific use cases and limitations')
        ]
        
        for inaccuracy_pattern, explanation in inaccuracy_patterns:
            if any(word in content_lower for word in inaccuracy_pattern.split()):
                # More sophisticated checking could be added here
                partial_matches = sum(1 for word in inaccuracy_pattern.split() if word in content_lower)
                if partial_matches >= len(inaccuracy_pattern.split()) * 0.6:  # 60% of words match
                    accuracy_issues.append({
                        'issue_type': 'potential_inaccuracy',
                        'pattern': inaccuracy_pattern,
                        'explanation': explanation,
                        'confidence': partial_matches / len(inaccuracy_pattern.split())
                    })
        
        # Check for outdated version references
        version_issues = self._check_version_references(content)
        accuracy_issues.extend(version_issues)
        
        # Check for deprecated technology references
        deprecated_issues = self._check_deprecated_references(content)
        accuracy_issues.extend(deprecated_issues)
        
        accuracy_score = max(0.3, 1.0 - len(accuracy_issues) * 0.15)
        
        return {
            'accuracy_score': round(accuracy_score, 3),
            'accuracy_issues': accuracy_issues,
            'issues_count': len(accuracy_issues),
            'critical_issues': [issue for issue in accuracy_issues if issue.get('confidence', 0) > 0.8]
        }
    
    def _check_version_references(self, content: str) -> List[Dict[str, Any]]:
        """Check for potentially outdated version references."""
        
        version_issues = []
        
        # Known outdated versions (as of 2024)
        outdated_versions = {
            'python 2': 'Python 2 is deprecated - use Python 3.x',
            'node.js 12': 'Node.js 12 is no longer supported - use Node.js 18+ or 20+',
            'tensorflow 1': 'TensorFlow 1.x is deprecated - use TensorFlow 2.x',
            'angular.js': 'AngularJS is deprecated - use modern Angular (2+)',
            'jquery': 'Consider modern alternatives to jQuery for new projects'
        }
        
        content_lower = content.lower()
        
        for outdated_ref, suggestion in outdated_versions.items():
            if outdated_ref in content_lower:
                version_issues.append({
                    'issue_type': 'outdated_version',
                    'pattern': outdated_ref,
                    'explanation': suggestion,
                    'confidence': 0.9
                })
        
        return version_issues
    
    def _check_deprecated_references(self, content: str) -> List[Dict[str, Any]]:
        """Check for references to deprecated technologies or practices."""
        
        deprecated_issues = []
        content_lower = content.lower()
        
        # Deprecated technologies and practices
        deprecated_refs = {
            'flash': 'Adobe Flash is discontinued - use modern web technologies',
            'internet explorer': 'Internet Explorer is deprecated - focus on modern browsers',
            'xml-rpc': 'XML-RPC is largely replaced by REST APIs',
            'soap': 'SOAP is largely replaced by REST for most use cases',
            'ftp': 'Consider SFTP or HTTPS for secure file transfers'
        }
        
        for deprecated_ref, suggestion in deprecated_refs.items():
            if deprecated_ref in content_lower:
                deprecated_issues.append({
                    'issue_type': 'deprecated_technology',
                    'pattern': deprecated_ref,
                    'explanation': suggestion,
                    'confidence': 0.8
                })
        
        return deprecated_issues
    
    def _validate_code_snippets_sync(self, code_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate code snippets for syntax and potential execution."""
        
        validation_results = []
        
        for snippet in code_snippets:
            language = snippet['language']
            code = snippet['code']
            snippet_id = snippet['snippet_id']
            
            logger.info(f"Validating {language} code snippet: {snippet_id}")
            
            result = {
                'snippet_id': snippet_id,
                'language': language,
                'validation_status': 'unknown',
                'syntax_valid': False,
                'execution_result': None,
                'issues': [],
                'recommendations': []
            }
            
            if language in self.supported_languages:
                # Syntax validation
                syntax_result = self._validate_code_syntax_sync(code, language)
                result.update(syntax_result)
                
                # Execution validation (if safe and Docker available)
                if result['syntax_valid'] and self.docker_client:
                    execution_result = self._execute_code_safely_sync(code, language)
                    result['execution_result'] = execution_result
            else:
                result['issues'].append(f"Language '{language}' not supported for validation")
                result['recommendations'].append(f"Manual review required for {language} code")
            
            validation_results.append(result)
        
        # Calculate overall code validation score
        valid_snippets = sum(1 for r in validation_results if r['syntax_valid'])
        total_snippets = len(validation_results)
        
        code_validation_score = valid_snippets / total_snippets if total_snippets > 0 else 1.0
        
        return {
            'code_validation_score': round(code_validation_score, 3),
            'snippets_validated': validation_results,
            'total_snippets': total_snippets,
            'valid_snippets': valid_snippets,
            'execution_attempted': any(r.get('execution_result') for r in validation_results)
        }
    
    def _validate_code_syntax_sync(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax for a specific language."""
        
        if language == 'python':
            return self._validate_python_syntax(code)
        elif language in ['javascript', 'typescript']:
            return self._validate_javascript_syntax_sync(code)
        elif language == 'sql':
            return self._validate_sql_syntax(code)
        elif language == 'bash':
            return self._validate_bash_syntax(code)
        else:
            return {
                'syntax_valid': False,
                'issues': [f"Syntax validation not implemented for {language}"],
                'recommendations': [f"Manual syntax review needed for {language} code"]
            }
    
    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python code syntax."""
        
        try:
            # Parse Python code
            ast.parse(code)
            
            return {
                'syntax_valid': True,
                'validation_status': 'passed',
                'issues': [],
                'recommendations': []
            }
            
        except SyntaxError as e:
            return {
                'syntax_valid': False,
                'validation_status': 'syntax_error',
                'issues': [f"Python syntax error: {e.msg} at line {e.lineno}"],
                'recommendations': ["Fix Python syntax errors before using this code"]
            }
        except Exception as e:
            return {
                'syntax_valid': False,
                'validation_status': 'validation_failed',
                'issues': [f"Python validation failed: {str(e)}"],
                'recommendations': ["Manual Python code review required"]
            }
    
    def _validate_javascript_syntax_sync(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript/TypeScript syntax."""
        
        # Simple validation using Node.js syntax check
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()
                
                # Use node.js to check syntax
                result = subprocess.run(
                    ['node', '--check', temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                os.unlink(temp_file.name)
                
                if result.returncode == 0:
                    return {
                        'syntax_valid': True,
                        'validation_status': 'passed',
                        'issues': [],
                        'recommendations': []
                    }
                else:
                    return {
                        'syntax_valid': False,
                        'validation_status': 'syntax_error',
                        'issues': [f"JavaScript syntax error: {result.stderr}"],
                        'recommendations': ["Fix JavaScript syntax errors before using this code"]
                    }
                    
        except FileNotFoundError:
            return {
                'syntax_valid': False,
                'validation_status': 'validator_unavailable',
                'issues': ["Node.js not available for JavaScript validation"],
                'recommendations': ["Install Node.js for JavaScript code validation"]
            }
        except Exception as e:
            return {
                'syntax_valid': False,
                'validation_status': 'validation_failed',
                'issues': [f"JavaScript validation failed: {str(e)}"],
                'recommendations': ["Manual JavaScript code review required"]
            }
    
    def _validate_sql_syntax(self, code: str) -> Dict[str, Any]:
        """Validate SQL syntax (basic validation)."""
        
        # Basic SQL syntax validation using regex patterns
        sql_keywords = r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|INDEX|TABLE|DATABASE)\b'
        
        if not re.search(sql_keywords, code, re.IGNORECASE):
            return {
                'syntax_valid': False,
                'validation_status': 'no_sql_keywords',
                'issues': ["No SQL keywords detected in supposed SQL code"],
                'recommendations': ["Verify this is valid SQL code"]
            }
        
        # Check for basic syntax patterns
        issues = []
        
        # Check for unmatched parentheses
        if code.count('(') != code.count(')'):
            issues.append("Unmatched parentheses in SQL code")
        
        # Check for proper statement endings (semicolons)
        statements = [s.strip() for s in code.split(';') if s.strip()]
        if len(statements) > 1 and not code.rstrip().endswith(';'):
            issues.append("Multiple SQL statements should end with semicolons")
        
        syntax_valid = len(issues) == 0
        
        return {
            'syntax_valid': syntax_valid,
            'validation_status': 'basic_check_passed' if syntax_valid else 'issues_found',
            'issues': issues,
            'recommendations': ["Use a proper SQL parser for comprehensive validation"] if issues else []
        }
    
    def _validate_bash_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Bash script syntax."""
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()
                
                # Use bash -n to check syntax without execution
                result = subprocess.run(
                    ['bash', '-n', temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                os.unlink(temp_file.name)
                
                if result.returncode == 0:
                    return {
                        'syntax_valid': True,
                        'validation_status': 'passed',
                        'issues': [],
                        'recommendations': []
                    }
                else:
                    return {
                        'syntax_valid': False,
                        'validation_status': 'syntax_error',
                        'issues': [f"Bash syntax error: {result.stderr}"],
                        'recommendations': ["Fix Bash syntax errors before using this script"]
                    }
                    
        except Exception as e:
            return {
                'syntax_valid': False,
                'validation_status': 'validation_failed',
                'issues': [f"Bash validation failed: {str(e)}"],
                'recommendations': ["Manual Bash script review required"]
            }
    
    def _execute_code_safely_sync(self, code: str, language: str) -> Dict[str, Any]:
        """Execute code safely in a sandboxed environment using Docker."""
        
        if not self.docker_client:
            return {
                'execution_attempted': False,
                'result': 'Docker not available for safe execution'
            }
        
        try:
            # Choose appropriate Docker image
            images = {
                'python': 'python:3.11-alpine',
                'javascript': 'node:18-alpine',
                'bash': 'alpine:latest'
            }
            
            image = images.get(language, 'alpine:latest')
            
            # Prepare execution command
            if language == 'python':
                command = f'python -c "{code}"'
            elif language == 'javascript':
                command = f'node -e "{code}"'
            elif language == 'bash':
                command = f'sh -c "{code}"'
            else:
                return {
                    'execution_attempted': False,
                    'result': f'Execution not supported for {language}'
                }
            
            # Execute in container with resource limits
            container = self.docker_client.containers.run(
                image,
                command,
                detach=True,
                remove=True,
                network_disabled=True,  # No network access
                mem_limit='128m',       # 128MB memory limit
                cpu_quota=50000,        # 50% CPU limit
                read_only=True,         # Read-only filesystem
                security_opt=['no-new-privileges:true']
            )
            
            # Wait for execution with timeout
            try:
                result = container.wait(timeout=10)
                logs = container.logs().decode('utf-8')
                
                return {
                    'execution_attempted': True,
                    'exit_code': result['StatusCode'],
                    'output': logs[:500],  # Limit output
                    'execution_successful': result['StatusCode'] == 0,
                    'execution_time': 'within_timeout'
                }
                
            except Exception as timeout_error:
                # Kill container on timeout
                try:
                    container.kill()
                except:
                    pass
                
                return {
                    'execution_attempted': True,
                    'result': 'Execution timed out (10 second limit)',
                    'execution_successful': False,
                    'timeout': True
                }
                
        except Exception as e:
            logger.warning(f"Safe code execution failed: {e}")
            return {
                'execution_attempted': False,
                'result': f'Execution failed: {str(e)}',
                'execution_successful': False,
                'error': str(e)
            }
    
    def _analyze_technical_depth(self, content: str, domain_focus: List[str]) -> Dict[str, Any]:
        """Analyze technical depth and completeness of content."""
        
        # Count technical indicators
        technical_indicators = {
            'code_examples': len(re.findall(r'```[\s\S]*?```', content)),
            'api_references': len(re.findall(r'(GET|POST|PUT|DELETE)\s+/[^\s]*', content, re.IGNORECASE)),
            'configuration_examples': len(re.findall(r'(config|settings|env)', content, re.IGNORECASE)),
            'error_handling_mentions': len(re.findall(r'(error|exception|try|catch|fail)', content, re.IGNORECASE)),
            'best_practices_mentions': len(re.findall(r'best practice', content, re.IGNORECASE)),
            'architecture_discussions': len(re.findall(r'(architecture|design pattern|framework)', content, re.IGNORECASE)),
            'performance_considerations': len(re.findall(r'(performance|optimization|scalability)', content, re.IGNORECASE)),
            'security_considerations': len(re.findall(r'(security|authentication|authorization|encryption)', content, re.IGNORECASE))
        }
        
        # Calculate depth score based on technical indicators
        depth_weights = {
            'code_examples': 0.2,
            'api_references': 0.15,
            'configuration_examples': 0.1,
            'error_handling_mentions': 0.15,
            'best_practices_mentions': 0.1,
            'architecture_discussions': 0.15,
            'performance_considerations': 0.1,
            'security_considerations': 0.05
        }
        
        # Normalize indicators (cap at reasonable maximums)
        max_values = {
            'code_examples': 5,
            'api_references': 10,
            'configuration_examples': 8,
            'error_handling_mentions': 15,
            'best_practices_mentions': 5,
            'architecture_discussions': 8,
            'performance_considerations': 10,
            'security_considerations': 12
        }
        
        normalized_indicators = {}
        for indicator, count in technical_indicators.items():
            max_val = max_values.get(indicator, 10)
            normalized_indicators[indicator] = min(1.0, count / max_val)
        
        # Calculate weighted depth score
        depth_score = sum(
            normalized_indicators[indicator] * depth_weights[indicator]
            for indicator in depth_weights.keys()
        )
        
        # Assess completeness for domain
        completeness_assessment = self._assess_domain_completeness(content, domain_focus, technical_indicators)
        
        return {
            'technical_depth_score': round(min(1.0, depth_score), 3),
            'technical_indicators': technical_indicators,
            'normalized_indicators': {k: round(v, 3) for k, v in normalized_indicators.items()},
            'completeness_assessment': completeness_assessment,
            'depth_level': self._categorize_depth_level(depth_score),
            'missing_elements': self._identify_missing_technical_elements(technical_indicators, domain_focus)
        }
    
    def _assess_domain_completeness(
        self,
        content: str,
        domain_focus: List[str],
        technical_indicators: Dict[str, int]
    ) -> Dict[str, Any]:
        """Assess completeness for specific technical domains."""
        
        completeness = {}
        
        for domain in domain_focus or []:
            if domain == 'ai_software_engineering':
                # Should have code examples, API references, best practices
                required_elements = ['code_examples', 'best_practices_mentions', 'error_handling_mentions']
                present_elements = sum(1 for elem in required_elements if technical_indicators[elem] > 0)
                completeness[domain] = present_elements / len(required_elements)
                
            elif domain == 'agentic_ai':
                # Should have architecture discussions, code examples, configuration
                required_elements = ['code_examples', 'architecture_discussions', 'configuration_examples']
                present_elements = sum(1 for elem in required_elements if technical_indicators[elem] > 0)
                completeness[domain] = present_elements / len(required_elements)
                
            elif domain == 'it_insurance':
                # Should have security considerations, compliance mentions
                security_mentions = technical_indicators['security_considerations']
                compliance_mentions = len(re.findall(r'(compliance|regulation|gdpr|hipaa)', content, re.IGNORECASE))
                completeness[domain] = min(1.0, (security_mentions + compliance_mentions) / 5)
        
        return completeness
    
    def _categorize_depth_level(self, depth_score: float) -> str:
        """Categorize technical depth level."""
        
        if depth_score >= 0.8:
            return 'comprehensive'
        elif depth_score >= 0.6:
            return 'detailed'
        elif depth_score >= 0.4:
            return 'moderate'
        elif depth_score >= 0.2:
            return 'basic'
        else:
            return 'superficial'
    
    def _identify_missing_technical_elements(
        self,
        technical_indicators: Dict[str, int],
        domain_focus: List[str]
    ) -> List[str]:
        """Identify missing technical elements that could improve content."""
        
        missing_elements = []
        
        # General missing elements
        if technical_indicators['code_examples'] == 0:
            missing_elements.append("Add practical code examples to demonstrate concepts")
        
        if technical_indicators['error_handling_mentions'] == 0:
            missing_elements.append("Include error handling and troubleshooting guidance")
        
        if technical_indicators['best_practices_mentions'] == 0:
            missing_elements.append("Add best practices and recommendations")
        
        if technical_indicators['performance_considerations'] == 0:
            missing_elements.append("Consider adding performance optimization tips")
        
        # Domain-specific missing elements
        for domain in domain_focus or []:
            if domain == 'ai_software_engineering':
                if technical_indicators['api_references'] == 0:
                    missing_elements.append("Include API usage examples for AI development")
                if technical_indicators['configuration_examples'] == 0:
                    missing_elements.append("Add configuration examples for AI frameworks")
                    
            elif domain == 'it_insurance':
                if technical_indicators['security_considerations'] == 0:
                    missing_elements.append("Add security considerations for insurance systems")
        
        return missing_elements[:6]  # Limit to 6 recommendations
    
    def _calculate_concept_validation_score(
        self,
        technical_patterns: Dict[str, Any],
        concept_validation: Dict[str, Any],
        accuracy_check: Dict[str, Any]
    ) -> float:
        """Calculate overall concept validation score."""
        
        # Base score from technical depth
        depth_score = min(1.0, len(technical_patterns) / 5)  # Up to 5 patterns is good
        
        # Average domain-specific validation scores
        domain_scores = [
            results.get('accuracy_score', 0.8)
            for results in concept_validation.values()
        ]
        avg_domain_score = sum(domain_scores) / len(domain_scores) if domain_scores else 0.8
        
        # Accuracy check score
        accuracy_score = accuracy_check.get('accuracy_score', 0.8)
        
        # Weighted combination
        concept_score = (
            depth_score * 0.3 +
            avg_domain_score * 0.4 +
            accuracy_score * 0.3
        )
        
        return round(min(1.0, concept_score), 3)
    
    def _calculate_technical_validation_score(
        self,
        code_analysis: Dict[str, Any],
        concept_validation: Dict[str, Any],
        accuracy_check: Dict[str, Any],
        code_validation: Dict[str, Any],
        depth_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall technical validation score."""
        
        # Weighted scoring components
        weights = {
            'concept_validation': 0.3,
            'accuracy_check': 0.25,
            'depth_analysis': 0.25,
            'code_validation': 0.2
        }
        
        scores = {
            'concept_validation': concept_validation.get('concept_validation_score', 0.8),
            'accuracy_check': accuracy_check.get('accuracy_score', 0.8),
            'depth_analysis': depth_analysis.get('technical_depth_score', 0.6),
            'code_validation': code_validation.get('code_validation_score', 1.0) if code_validation else 1.0
        }
        
        # Calculate weighted score
        technical_score = sum(scores[component] * weights[component] for component in weights.keys())
        
        return round(min(1.0, max(0.0, technical_score)), 3)
    
    def _generate_technical_recommendations(
        self,
        technical_score: float,
        code_analysis: Dict[str, Any],
        concept_validation: Dict[str, Any],
        accuracy_check: Dict[str, Any]
    ) -> List[str]:
        """Generate technical improvement recommendations."""
        
        recommendations = []
        
        # Overall score recommendations
        if technical_score < 0.6:
            recommendations.append("Technical content needs significant improvement - consider expert review")
        elif technical_score < 0.8:
            recommendations.append("Technical content is good but could benefit from additional depth")
        else:
            recommendations.append("Technical content quality is excellent")
        
        # Code-specific recommendations
        if code_analysis.get('total_code_blocks', 0) == 0:
            recommendations.append("Consider adding code examples to illustrate technical concepts")
        elif not code_analysis.get('has_executable_code', False):
            recommendations.append("Add executable code examples for better practical value")
        
        # Accuracy recommendations
        accuracy_issues = accuracy_check.get('accuracy_issues', [])
        if accuracy_issues:
            recommendations.append(f"Address {len(accuracy_issues)} potential technical accuracy issues")
            
            critical_issues = [issue for issue in accuracy_issues if issue.get('confidence', 0) > 0.8]
            if critical_issues:
                recommendations.append("Review critical technical accuracy concerns before publishing")
        
        # Domain-specific recommendations
        for domain, results in concept_validation.get('domain_concept_validation', {}).items():
            if results.get('accuracy_score', 1.0) < 0.8:
                recommendations.append(f"Improve {domain}-specific technical accuracy and depth")
        
        return recommendations[:6]  # Limit to 6 recommendations