"""Compliance checking tool for regulatory, legal, and ethical content validation."""

import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ComplianceTool:
    """Tool for checking content compliance with regulatory, legal, and ethical standards."""
    
    def __init__(self):
        """Initialize compliance checking tool."""
        
        # Regulatory frameworks and their requirements
        self.regulatory_frameworks = {
            'gdpr': {
                'name': 'General Data Protection Regulation',
                'scope': 'EU data protection',
                'key_requirements': [
                    'lawful basis for processing',
                    'data subject rights',
                    'privacy by design',
                    'data protection impact assessment',
                    'breach notification',
                    'consent requirements'
                ],
                'prohibited_content': [
                    'non-consented personal data processing',
                    'unlawful data collection practices',
                    'inadequate privacy protection'
                ],
                'required_disclosures': [
                    'privacy policy requirements',
                    'data collection purposes',
                    'user rights information'
                ]
            },
            'hipaa': {
                'name': 'Health Insurance Portability and Accountability Act',
                'scope': 'US healthcare data protection',
                'key_requirements': [
                    'patient privacy protection',
                    'secure data transmission',
                    'access controls',
                    'audit trails',
                    'business associate agreements'
                ],
                'prohibited_content': [
                    'unauthorized phi disclosure',
                    'insecure health data practices',
                    'non-compliant data sharing'
                ]
            },
            'sox': {
                'name': 'Sarbanes-Oxley Act',
                'scope': 'US financial reporting',
                'key_requirements': [
                    'financial accuracy',
                    'internal controls',
                    'audit requirements',
                    'executive accountability'
                ],
                'prohibited_content': [
                    'misleading financial information',
                    'inadequate disclosure',
                    'audit interference'
                ]
            },
            'pci_dss': {
                'name': 'Payment Card Industry Data Security Standard',
                'scope': 'Payment card data protection',
                'key_requirements': [
                    'secure network architecture',
                    'cardholder data protection',
                    'encryption requirements',
                    'access controls',
                    'monitoring and testing'
                ],
                'prohibited_content': [
                    'insecure payment practices',
                    'unencrypted cardholder data',
                    'weak access controls'
                ]
            }
        }
        
        # Copyright and intellectual property patterns
        self.copyright_patterns = {
            'direct_quotes': r'"([^"]{50,})"',  # Long direct quotes
            'code_snippets': r'```[\s\S]*?```',  # Code blocks
            'image_references': r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown images
            'trademark_symbols': r'[®™©]',  # Trademark symbols
            'attribution_patterns': r'(?:source|credit|courtesy|via):\s*([^\n]+)'
        }
        
        # Bias and ethical compliance patterns
        self.bias_patterns = {
            'gender_bias': [
                r'\b(?:he|she)\s+(?:always|never|typically|usually)',
                r'\b(?:men|women)\s+(?:are|can\'t|should|shouldn\'t)',
                r'\b(?:guys|girls)\b(?!\s+(?:and|or))'
            ],
            'racial_bias': [
                r'\b(?:those people|these people)\b',
                r'\b(?:typical|stereotypical)\s+(?:\w+\s+)*(?:person|behavior)'
            ],
            'age_bias': [
                r'\b(?:millennials|boomers|gen\s*z)\s+(?:are|always|never)',
                r'\b(?:old|young)\s+people\s+(?:are|can\'t|should)'
            ],
            'religious_bias': [
                r'\b(?:all|most)\s+(?:christians|muslims|jews|buddhists|hindus)\b',
                r'\breligious\s+(?:extremist|fanatic)\b'
            ]
        }
        
        # Disclaimer and disclosure requirements
        self.disclosure_requirements = {
            'ai_generated': [
                'ai-generated content disclosure',
                'artificial intelligence usage notification',
                'automated content warning'
            ],
            'affiliate_marketing': [
                'affiliate link disclosure',
                'paid promotion notification',
                'sponsorship disclosure'
            ],
            'medical_advice': [
                'not medical advice disclaimer',
                'consult healthcare professional',
                'medical professional consultation'
            ],
            'financial_advice': [
                'not financial advice disclaimer',
                'consult financial advisor',
                'investment risk disclosure'
            ]
        }
        
        # Prohibited content patterns
        self.prohibited_content_patterns = {
            'discriminatory': [
                r'\b(?:inferior|superior)\s+(?:race|gender|religion)',
                r'\b(?:ban|exclude|prohibit)\s+(?:all\s+)?(?:\w+\s+)*(?:people|individuals)'
            ],
            'misleading_claims': [
                r'\b(?:guaranteed|100%|never fails|always works)\b',
                r'\bget rich quick\b',
                r'\b(?:miracle|magic|instant)\s+(?:cure|solution|fix)'
            ],
            'privacy_violations': [
                r'\b(?:personal data|private information)\s+(?:collection|harvesting)',
                r'\btrack(?:ing)?\s+without\s+(?:consent|permission)'
            ]
        }
    
    def check_regulatory_compliance(
        self,
        content: str,
        applicable_regulations: List[str] = None,
        domain_context: str = "general"
    ) -> Dict[str, Any]:
        """
        Check content compliance with specified regulatory frameworks.
        
        Args:
            content: Content to check for compliance
            applicable_regulations: List of regulations to check against
            domain_context: Domain context for compliance checking
            
        Returns:
            Dictionary with compliance analysis results
        """
        logger.info(f"Checking regulatory compliance for domain: {domain_context}")
        
        try:
            # Determine applicable regulations if not specified
            if not applicable_regulations:
                applicable_regulations = self._determine_applicable_regulations(content, domain_context)
            
            compliance_results = {}
            
            # Check each applicable regulation
            for regulation in applicable_regulations:
                if regulation in self.regulatory_frameworks:
                    compliance_results[regulation] = self._check_regulation_compliance(
                        content,
                        regulation,
                        self.regulatory_frameworks[regulation]
                    )
            
            # Calculate overall compliance score
            overall_compliance_score = self._calculate_overall_compliance_score(compliance_results)
            
            # Identify critical compliance issues
            critical_issues = self._identify_critical_compliance_issues(compliance_results)
            
            return {
                'overall_compliance_score': overall_compliance_score,
                'applicable_regulations': applicable_regulations,
                'regulation_compliance': compliance_results,
                'critical_issues': critical_issues,
                'compliance_status': self._determine_compliance_status(overall_compliance_score, critical_issues),
                'recommendations': self._generate_compliance_recommendations(
                    compliance_results, critical_issues, overall_compliance_score
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Regulatory compliance check failed: {e}")
            return {
                'overall_compliance_score': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_applicable_regulations(self, content: str, domain_context: str) -> List[str]:
        """Determine which regulations apply based on content and domain."""
        
        applicable = []
        content_lower = content.lower()
        
        # GDPR - applies to data processing content
        if any(term in content_lower for term in ['personal data', 'privacy', 'data protection', 'user data', 'gdpr']):
            applicable.append('gdpr')
        
        # HIPAA - applies to healthcare content
        if any(term in content_lower for term in ['healthcare', 'medical', 'patient', 'health information', 'hipaa']):
            applicable.append('hipaa')
        
        # SOX - applies to financial content
        if any(term in content_lower for term in ['financial', 'accounting', 'audit', 'securities', 'sox']):
            applicable.append('sox')
        
        # PCI DSS - applies to payment processing content
        if any(term in content_lower for term in ['payment', 'credit card', 'cardholder', 'pci']):
            applicable.append('pci_dss')
        
        # Domain-specific regulations
        if domain_context == 'it_insurance':
            # Insurance-specific regulations
            if 'gdpr' not in applicable:
                applicable.append('gdpr')  # Often applies to insurance
            if any(term in content_lower for term in ['insurance', 'policy', 'claims', 'underwriting']):
                applicable.append('sox')  # Financial reporting aspects
        
        return applicable
    
    def _check_regulation_compliance(
        self,
        content: str,
        regulation_name: str,
        regulation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check compliance with a specific regulation."""
        
        content_lower = content.lower()
        
        # Check for prohibited content
        violations = []
        for prohibited_item in regulation_config.get('prohibited_content', []):
            if prohibited_item.lower() in content_lower:
                violations.append({
                    'violation_type': 'prohibited_content',
                    'description': prohibited_item,
                    'severity': 'high'
                })
        
        # Check for required disclosures
        missing_disclosures = []
        for disclosure in regulation_config.get('required_disclosures', []):
            disclosure_keywords = disclosure.lower().split()
            if not any(keyword in content_lower for keyword in disclosure_keywords):
                missing_disclosures.append({
                    'disclosure_type': 'missing_required_disclosure',
                    'description': disclosure,
                    'severity': 'medium'
                })
        
        # Check key requirements coverage
        requirements_coverage = {}
        for requirement in regulation_config.get('key_requirements', []):
            requirement_keywords = requirement.lower().split()
            coverage_score = sum(1 for keyword in requirement_keywords if keyword in content_lower) / len(requirement_keywords)
            requirements_coverage[requirement] = {
                'coverage_score': round(coverage_score, 3),
                'covered': coverage_score >= 0.5
            }
        
        # Calculate regulation compliance score
        total_violations = len(violations)
        total_missing_disclosures = len(missing_disclosures)
        requirements_met = sum(1 for req in requirements_coverage.values() if req['covered'])
        total_requirements = len(requirements_coverage)
        
        # Scoring logic
        compliance_score = 1.0
        
        # Deduct for violations (major penalty)
        compliance_score -= total_violations * 0.3
        
        # Deduct for missing disclosures
        compliance_score -= total_missing_disclosures * 0.15
        
        # Adjust for requirements coverage
        requirements_ratio = requirements_met / total_requirements if total_requirements > 0 else 1.0
        compliance_score *= requirements_ratio
        
        compliance_score = max(0.0, min(1.0, compliance_score))
        
        return {
            'regulation_name': regulation_name,
            'compliance_score': round(compliance_score, 3),
            'violations': violations,
            'missing_disclosures': missing_disclosures,
            'requirements_coverage': requirements_coverage,
            'requirements_met': requirements_met,
            'total_requirements': total_requirements,
            'compliance_level': self._categorize_compliance_level(compliance_score)
        }
    
    def _categorize_compliance_level(self, score: float) -> str:
        """Categorize compliance score into levels."""
        
        if score >= 0.95:
            return 'excellent'
        elif score >= 0.85:
            return 'good'
        elif score >= 0.70:
            return 'acceptable'
        elif score >= 0.50:
            return 'concerning'
        else:
            return 'non_compliant'
    
    def check_copyright_compliance(self, content: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check content for copyright and intellectual property compliance."""
        
        logger.info("Checking copyright compliance")
        
        try:
            # Check for direct quotes and their attribution
            quote_analysis = self._analyze_quotes_and_attribution(content)
            
            # Check for code snippets and their licensing
            code_analysis = self._analyze_code_snippets_licensing(content)
            
            # Check for image usage and attribution
            image_analysis = self._analyze_image_usage(content)
            
            # Check for proper source attribution
            attribution_analysis = self._analyze_source_attribution(content, sources or [])
            
            # Check for trademark usage
            trademark_analysis = self._analyze_trademark_usage(content)
            
            # Calculate overall copyright compliance score
            copyright_score = self._calculate_copyright_compliance_score(
                quote_analysis,
                code_analysis,
                image_analysis,
                attribution_analysis,
                trademark_analysis
            )
            
            return {
                'copyright_compliance_score': copyright_score,
                'quote_analysis': quote_analysis,
                'code_analysis': code_analysis,
                'image_analysis': image_analysis,
                'attribution_analysis': attribution_analysis,
                'trademark_analysis': trademark_analysis,
                'compliance_recommendations': self._generate_copyright_recommendations(
                    quote_analysis, code_analysis, image_analysis, attribution_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Copyright compliance check failed: {e}")
            return {
                'copyright_compliance_score': 0.7,
                'error': str(e)
            }
    
    def _analyze_quotes_and_attribution(self, content: str) -> Dict[str, Any]:
        """Analyze direct quotes and their attribution."""
        
        # Find direct quotes
        quotes = re.findall(r'"([^"]{50,})"', content)  # Quotes 50+ characters
        
        attribution_issues = []
        properly_attributed = 0
        
        for quote in quotes:
            # Look for attribution near the quote
            quote_position = content.find(f'"{quote}"')
            
            # Check surrounding context for attribution
            context_start = max(0, quote_position - 200)
            context_end = min(len(content), quote_position + len(quote) + 200)
            context = content[context_start:context_end]
            
            # Look for attribution patterns
            attribution_found = bool(re.search(
                r'(?:according to|said|stated|quoted|source:|via|from|by)\s+[A-Z][^.]*',
                context,
                re.IGNORECASE
            ))
            
            if attribution_found:
                properly_attributed += 1
            else:
                attribution_issues.append({
                    'quote': quote[:100] + "..." if len(quote) > 100 else quote,
                    'issue': 'Direct quote lacks clear attribution',
                    'severity': 'medium'
                })
        
        attribution_score = properly_attributed / len(quotes) if quotes else 1.0
        
        return {
            'total_quotes': len(quotes),
            'properly_attributed': properly_attributed,
            'attribution_score': round(attribution_score, 3),
            'attribution_issues': attribution_issues,
            'long_quotes_count': len([q for q in quotes if len(q) > 150])
        }
    
    def _analyze_code_snippets_licensing(self, content: str) -> Dict[str, Any]:
        """Analyze code snippets for licensing and attribution requirements."""
        
        # Extract code blocks
        code_blocks = re.findall(r'```[\w]*\s*([\s\S]*?)```', content)
        
        licensing_issues = []
        code_with_attribution = 0
        
        for i, code in enumerate(code_blocks):
            code_lines = code.strip().split('\n')
            
            # Check for licensing comments
            has_license_comment = any(
                line.strip().startswith(('#', '//', '/*', '<!--')) and
                any(keyword in line.lower() for keyword in ['license', 'copyright', 'author', 'source'])
                for line in code_lines[:5]  # Check first 5 lines
            )
            
            # Check for attribution in surrounding context
            code_start = content.find(f'```')  # Simplified - should find exact match
            context_start = max(0, code_start - 100)
            context_end = min(len(content), code_start + len(code) + 200)
            context = content[context_start:context_end]
            
            has_context_attribution = bool(re.search(
                r'(?:source|from|via|adapted from|based on|credit):\s*[^\n]+',
                context,
                re.IGNORECASE
            ))
            
            if has_license_comment or has_context_attribution:
                code_with_attribution += 1
            else:
                # Check if code is likely original vs copied
                if len(code_lines) > 10 or any(
                    complex_pattern in code.lower()
                    for complex_pattern in ['class ', 'function ', 'import ', 'from ']
                ):
                    licensing_issues.append({
                        'code_snippet_index': i,
                        'issue': 'Substantial code snippet lacks attribution or licensing information',
                        'severity': 'medium',
                        'code_preview': code[:100] + "..." if len(code) > 100 else code
                    })
        
        attribution_score = code_with_attribution / len(code_blocks) if code_blocks else 1.0
        
        return {
            'total_code_blocks': len(code_blocks),
            'code_with_attribution': code_with_attribution,
            'attribution_score': round(attribution_score, 3),
            'licensing_issues': licensing_issues,
            'substantial_code_blocks': len([c for c in code_blocks if len(c.split('\n')) > 10])
        }
    
    def _analyze_image_usage(self, content: str) -> Dict[str, Any]:
        """Analyze image usage and attribution."""
        
        # Find image references (Markdown format)
        images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        
        image_issues = []
        images_with_attribution = 0
        
        for alt_text, image_url in images:
            # Check if alt text includes attribution
            has_attribution = any(
                keyword in alt_text.lower()
                for keyword in ['source:', 'credit:', 'via', 'courtesy', 'by']
            )
            
            # Check for stock photo or free image sources
            is_likely_free = any(
                domain in image_url.lower()
                for domain in ['unsplash', 'pexels', 'pixabay', 'freepik', 'wikimedia']
            )
            
            if has_attribution or is_likely_free:
                images_with_attribution += 1
            else:
                image_issues.append({
                    'image_url': image_url,
                    'alt_text': alt_text,
                    'issue': 'Image lacks attribution or source information',
                    'severity': 'medium'
                })
        
        attribution_score = images_with_attribution / len(images) if images else 1.0
        
        return {
            'total_images': len(images),
            'images_with_attribution': images_with_attribution,
            'attribution_score': round(attribution_score, 3),
            'image_issues': image_issues,
            'likely_stock_images': len([url for _, url in images if any(domain in url.lower() for domain in ['unsplash', 'pexels', 'pixabay'])])
        }
    
    def _analyze_source_attribution(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze proper attribution of sources used in content."""
        
        # Extract citations
        citation_pattern = r'\[(\d+)\]|\[Citation (\d+)\]'
        citations = re.findall(citation_pattern, content)
        
        citation_numbers = []
        for citation in citations:
            if citation[0]:
                try:
                    citation_numbers.append(int(citation[0]))
                except ValueError:
                    pass
            elif citation[1]:
                try:
                    citation_numbers.append(int(citation[1]))
                except ValueError:
                    pass
        
        # Check if all sources are properly cited
        unique_citations = set(citation_numbers)
        sources_cited = len(unique_citations)
        total_sources = len(sources)
        
        # Check for orphaned citations (citations without corresponding sources)
        orphaned_citations = [n for n in unique_citations if n > total_sources]
        
        # Check for unused sources
        unused_sources = total_sources - sources_cited if sources_cited <= total_sources else 0
        
        attribution_score = sources_cited / total_sources if total_sources > 0 else 1.0
        
        # Penalize orphaned citations
        if orphaned_citations:
            attribution_score *= 0.8
        
        return {
            'attribution_score': round(min(1.0, attribution_score), 3),
            'sources_cited': sources_cited,
            'total_sources': total_sources,
            'citation_coverage': round(attribution_score, 3),
            'orphaned_citations': orphaned_citations,
            'unused_sources': unused_sources,
            'attribution_issues': [
                f"Orphaned citation: [{n}]" for n in orphaned_citations
            ] + [
                f"{unused_sources} sources not cited" if unused_sources > 0 else ""
            ]
        }
    
    def _analyze_trademark_usage(self, content: str) -> Dict[str, Any]:
        """Analyze trademark usage and compliance."""
        
        # Find trademark symbols
        trademark_symbols = re.findall(r'([A-Za-z0-9\s]+)[®™©]', content)
        
        # Common trademarks to check
        known_trademarks = [
            'OpenAI', 'ChatGPT', 'GPT-4', 'Microsoft', 'Windows', 'Azure',
            'Google', 'Google Cloud', 'Amazon', 'AWS', 'Apple', 'iPhone',
            'Meta', 'Facebook', 'Instagram'
        ]
        
        trademark_usage = []
        potential_issues = []
        
        for trademark in known_trademarks:
            if trademark in content:
                has_symbol = bool(re.search(rf'{re.escape(trademark)}\s*[®™]', content))
                
                trademark_usage.append({
                    'trademark': trademark,
                    'has_symbol': has_symbol,
                    'usage_count': content.count(trademark)
                })
                
                # Check for potential misuse
                trademark_context = self._get_trademark_context(content, trademark)
                if self._is_potentially_infringing_usage(trademark_context):
                    potential_issues.append({
                        'trademark': trademark,
                        'issue': 'Potential trademark misuse - review usage context',
                        'context': trademark_context[:100] + "..."
                    })
        
        compliance_score = 1.0 - (len(potential_issues) * 0.2)  # Deduct for potential issues
        
        return {
            'trademark_compliance_score': round(max(0.0, compliance_score), 3),
            'trademarks_found': trademark_usage,
            'potential_issues': potential_issues,
            'trademark_symbols_used': len(trademark_symbols),
            'proper_attribution': len([t for t in trademark_usage if t['has_symbol']])
        }
    
    def _get_trademark_context(self, content: str, trademark: str) -> str:
        """Get context around trademark usage."""
        
        position = content.find(trademark)
        if position == -1:
            return ""
        
        start = max(0, position - 50)
        end = min(len(content), position + len(trademark) + 50)
        
        return content[start:end]
    
    def _is_potentially_infringing_usage(self, context: str) -> bool:
        """Check if trademark usage might be infringing."""
        
        context_lower = context.lower()
        
        # Infringing usage indicators
        infringing_patterns = [
            'our product',
            'we offer',
            'buy our',
            'developed by us',
            'owned by',
            'created by us'
        ]
        
        return any(pattern in context_lower for pattern in infringing_patterns)
    
    def check_ethical_compliance(self, content: str) -> Dict[str, Any]:
        """Check content for ethical compliance including bias and discrimination."""
        
        logger.info("Checking ethical compliance")
        
        try:
            # Check for bias patterns
            bias_analysis = self._analyze_bias_patterns(content)
            
            # Check for discriminatory language
            discrimination_check = self._check_discriminatory_language(content)
            
            # Check for misleading claims
            misleading_claims_check = self._check_misleading_claims(content)
            
            # Check for required disclaimers
            disclaimer_check = self._check_required_disclaimers(content)
            
            # Calculate overall ethical compliance score
            ethical_score = self._calculate_ethical_compliance_score(
                bias_analysis,
                discrimination_check,
                misleading_claims_check,
                disclaimer_check
            )
            
            return {
                'ethical_compliance_score': ethical_score,
                'bias_analysis': bias_analysis,
                'discrimination_check': discrimination_check,
                'misleading_claims_check': misleading_claims_check,
                'disclaimer_check': disclaimer_check,
                'ethical_issues_found': (
                    bias_analysis.get('bias_issues_count', 0) +
                    discrimination_check.get('discrimination_issues_count', 0) +
                    misleading_claims_check.get('misleading_claims_count', 0)
                ),
                'recommendations': self._generate_ethical_recommendations(
                    bias_analysis, discrimination_check, misleading_claims_check, disclaimer_check
                )
            }
            
        except Exception as e:
            logger.error(f"Ethical compliance check failed: {e}")
            return {
                'ethical_compliance_score': 0.7,
                'error': str(e)
            }
    
    def _analyze_bias_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze content for bias patterns."""
        
        bias_issues = defaultdict(list)
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    bias_issues[bias_type].append({
                        'text': match.group(),
                        'position': match.start(),
                        'context': content[max(0, match.start()-50):match.end()+50]
                    })
        
        total_bias_issues = sum(len(issues) for issues in bias_issues.values())
        bias_score = max(0.0, 1.0 - (total_bias_issues * 0.2))
        
        return {
            'bias_score': round(bias_score, 3),
            'bias_issues': dict(bias_issues),
            'bias_issues_count': total_bias_issues,
            'bias_types_found': list(bias_issues.keys())
        }
    
    def _check_discriminatory_language(self, content: str) -> Dict[str, Any]:
        """Check for discriminatory language."""
        
        discriminatory_issues = []
        content_lower = content.lower()
        
        # Discriminatory patterns
        discriminatory_phrases = [
            'all [group] are',
            'typical [group] behavior',
            '[group] people always',
            '[group] can\'t',
            'not suitable for [group]'
        ]
        
        # Protected groups to check for
        protected_groups = [
            'women', 'men', 'elderly', 'disabled', 'minorities',
            'immigrants', 'refugees', 'lgbt', 'gay', 'lesbian'
        ]
        
        for group in protected_groups:
            # Check for generalizations about the group
            generalization_patterns = [
                f'all {group}',
                f'{group} are always',
                f'{group} never',
                f'{group} can\'t',
                f'typical {group}'
            ]
            
            for pattern in generalization_patterns:
                if pattern in content_lower:
                    discriminatory_issues.append({
                        'issue_type': 'generalization',
                        'affected_group': group,
                        'pattern': pattern,
                        'severity': 'high'
                    })
        
        discrimination_score = max(0.0, 1.0 - (len(discriminatory_issues) * 0.3))
        
        return {
            'discrimination_score': round(discrimination_score, 3),
            'discrimination_issues': discriminatory_issues,
            'discrimination_issues_count': len(discriminatory_issues),
            'protected_groups_mentioned': [
                group for group in protected_groups if group in content_lower
            ]
        }
    
    def _check_misleading_claims(self, content: str) -> Dict[str, Any]:
        """Check for misleading claims and exaggerations."""
        
        misleading_issues = []
        content_lower = content.lower()
        
        # Misleading claim patterns
        misleading_patterns = [
            r'\b(?:guaranteed|100% guaranteed|never fails|always works)\b',
            r'\b(?:get rich quick|instant success|overnight)\b',
            r'\b(?:miracle|magic|secret)\s+(?:solution|cure|method)\b',
            r'\b(?:unlimited|infinite|endless)\b',
            r'\b(?:completely|totally|absolutely)\s+(?:free|safe|secure)\b'
        ]
        
        for pattern in misleading_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                misleading_issues.append({
                    'claim': match.group(),
                    'issue_type': 'misleading_claim',
                    'severity': 'medium',
                    'position': match.start()
                })
        
        # Check for exaggerated benefits claims
        benefit_patterns = [
            r'\b(?:increases|improves|boosts)\s+(?:by\s+)?(\d+)%',
            r'\b(\d+)x\s+(?:faster|better|more|improvement)'
        ]
        
        for pattern in benefit_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    if value > 1000:  # Suspiciously high improvement claims
                        misleading_issues.append({
                            'claim': match.group(),
                            'issue_type': 'exaggerated_benefit',
                            'severity': 'medium',
                            'value': value
                        })
                except ValueError:
                    pass
        
        misleading_score = max(0.0, 1.0 - (len(misleading_issues) * 0.25))
        
        return {
            'misleading_claims_score': round(misleading_score, 3),
            'misleading_claims': misleading_issues,
            'misleading_claims_count': len(misleading_issues),
            'exaggerated_claims': [
                issue for issue in misleading_issues 
                if issue['issue_type'] == 'exaggerated_benefit'
            ]
        }
    
    def _check_required_disclaimers(self, content: str) -> Dict[str, Any]:
        """Check for required disclaimers based on content type."""
        
        content_lower = content.lower()
        missing_disclaimers = []
        present_disclaimers = []
        
        # Check each disclaimer requirement
        for disclaimer_type, requirements in self.disclosure_requirements.items():
            content_needs_disclaimer = False
            
            # Determine if disclaimer is needed
            if disclaimer_type == 'ai_generated':
                content_needs_disclaimer = any(
                    term in content_lower
                    for term in ['ai-generated', 'artificial intelligence', 'machine learning', 'automated']
                )
            elif disclaimer_type == 'medical_advice':
                content_needs_disclaimer = any(
                    term in content_lower
                    for term in ['medical', 'health', 'treatment', 'diagnosis', 'symptoms', 'cure']
                )
            elif disclaimer_type == 'financial_advice':
                content_needs_disclaimer = any(
                    term in content_lower
                    for term in ['investment', 'trading', 'financial advice', 'portfolio', 'stocks', 'cryptocurrency']
                )
            elif disclaimer_type == 'affiliate_marketing':
                content_needs_disclaimer = any(
                    term in content_lower
                    for term in ['affiliate', 'sponsored', 'partnership', 'paid promotion']
                )
            
            if content_needs_disclaimer:
                # Check if appropriate disclaimer is present
                disclaimer_present = any(
                    requirement.lower() in content_lower
                    for requirement in requirements
                )
                
                if disclaimer_present:
                    present_disclaimers.append(disclaimer_type)
                else:
                    missing_disclaimers.append({
                        'disclaimer_type': disclaimer_type,
                        'required_elements': requirements,
                        'severity': 'high' if disclaimer_type in ['medical_advice', 'financial_advice'] else 'medium'
                    })
        
        disclaimer_score = 1.0 - (len(missing_disclaimers) * 0.3)
        
        return {
            'disclaimer_score': round(max(0.0, disclaimer_score), 3),
            'missing_disclaimers': missing_disclaimers,
            'present_disclaimers': present_disclaimers,
            'disclaimer_requirements_met': len(present_disclaimers),
            'disclaimer_violations': len(missing_disclaimers)
        }
    
    def _calculate_overall_compliance_score(self, compliance_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score across all regulations."""
        
        if not compliance_results:
            return 0.8  # Default score if no regulations apply
        
        scores = [result['compliance_score'] for result in compliance_results.values()]
        
        # Use minimum score (most restrictive)
        overall_score = min(scores) if scores else 0.8
        
        return round(overall_score, 3)
    
    def _calculate_copyright_compliance_score(
        self,
        quote_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any],
        attribution_analysis: Dict[str, Any],
        trademark_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall copyright compliance score."""
        
        weights = {
            'quote_analysis': 0.25,
            'code_analysis': 0.25,
            'image_analysis': 0.20,
            'attribution_analysis': 0.20,
            'trademark_analysis': 0.10
        }
        
        scores = {
            'quote_analysis': quote_analysis.get('attribution_score', 1.0),
            'code_analysis': code_analysis.get('attribution_score', 1.0),
            'image_analysis': image_analysis.get('attribution_score', 1.0),
            'attribution_analysis': attribution_analysis.get('attribution_score', 1.0),
            'trademark_analysis': trademark_analysis.get('trademark_compliance_score', 1.0)
        }
        
        overall_score = sum(scores[component] * weights[component] for component in weights.keys())
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def _calculate_ethical_compliance_score(
        self,
        bias_analysis: Dict[str, Any],
        discrimination_check: Dict[str, Any],
        misleading_claims_check: Dict[str, Any],
        disclaimer_check: Dict[str, Any]
    ) -> float:
        """Calculate overall ethical compliance score."""
        
        weights = {
            'bias_analysis': 0.3,
            'discrimination_check': 0.3,
            'misleading_claims_check': 0.25,
            'disclaimer_check': 0.15
        }
        
        scores = {
            'bias_analysis': bias_analysis.get('bias_score', 1.0),
            'discrimination_check': discrimination_check.get('discrimination_score', 1.0),
            'misleading_claims_check': misleading_claims_check.get('misleading_claims_score', 1.0),
            'disclaimer_check': disclaimer_check.get('disclaimer_score', 1.0)
        }
        
        overall_score = sum(scores[component] * weights[component] for component in weights.keys())
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def _identify_critical_compliance_issues(self, compliance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical compliance issues that must be addressed."""
        
        critical_issues = []
        
        for regulation, results in compliance_results.items():
            # High severity violations
            violations = results.get('violations', [])
            critical_violations = [v for v in violations if v.get('severity') == 'high']
            
            for violation in critical_violations:
                critical_issues.append({
                    'regulation': regulation,
                    'issue_type': 'violation',
                    'description': violation['description'],
                    'severity': 'critical'
                })
            
            # Low compliance scores
            if results.get('compliance_score', 1.0) < 0.5:
                critical_issues.append({
                    'regulation': regulation,
                    'issue_type': 'low_compliance_score',
                    'description': f"Overall compliance score too low: {results['compliance_score']}",
                    'severity': 'critical'
                })
        
        return critical_issues
    
    def _determine_compliance_status(self, overall_score: float, critical_issues: List[Dict[str, Any]]) -> str:
        """Determine overall compliance status."""
        
        if critical_issues:
            return 'non_compliant'
        elif overall_score >= 0.9:
            return 'fully_compliant'
        elif overall_score >= 0.8:
            return 'mostly_compliant'
        elif overall_score >= 0.6:
            return 'partially_compliant'
        else:
            return 'non_compliant'
    
    def _generate_compliance_recommendations(
        self,
        compliance_results: Dict[str, Any],
        critical_issues: List[Dict[str, Any]],
        overall_score: float
    ) -> List[str]:
        """Generate compliance improvement recommendations."""
        
        recommendations = []
        
        # Critical issues first
        if critical_issues:
            recommendations.append("CRITICAL: Address all critical compliance issues before publishing")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                recommendations.append(f"  - {issue['description']}")
        
        # Regulation-specific recommendations
        for regulation, results in compliance_results.items():
            compliance_score = results.get('compliance_score', 1.0)
            
            if compliance_score < 0.7:
                framework_name = self.regulatory_frameworks[regulation]['name']
                recommendations.append(f"Improve {framework_name} compliance (current score: {compliance_score:.2f})")
                
                # Add specific recommendations based on violations
                violations = results.get('violations', [])
                if violations:
                    recommendations.append(f"Address {regulation.upper()} violations: {violations[0]['description']}")
                
                missing_disclosures = results.get('missing_disclosures', [])
                if missing_disclosures:
                    recommendations.append(f"Add required {regulation.upper()} disclosures")
        
        # General recommendations
        if overall_score >= 0.9:
            recommendations.append("Compliance score is excellent - content meets regulatory standards")
        elif overall_score >= 0.8:
            recommendations.append("Compliance score is good - minor improvements may be beneficial")
        elif overall_score >= 0.6:
            recommendations.append("Compliance score needs improvement - review all identified issues")
        else:
            recommendations.append("Compliance score is insufficient - comprehensive review required")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _generate_copyright_recommendations(
        self,
        quote_analysis: Dict[str, Any],
        code_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any],
        attribution_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate copyright compliance recommendations."""
        
        recommendations = []
        
        # Quote attribution recommendations
        quote_issues = quote_analysis.get('attribution_issues', [])
        if quote_issues:
            recommendations.append("Add proper attribution for direct quotes")
        
        # Code licensing recommendations
        code_issues = code_analysis.get('licensing_issues', [])
        if code_issues:
            recommendations.append("Add licensing information or attribution for code snippets")
        
        # Image attribution recommendations
        image_issues = image_analysis.get('image_issues', [])
        if image_issues:
            recommendations.append("Add attribution or use royalty-free images")
        
        # Source attribution recommendations
        attribution_issues = attribution_analysis.get('attribution_issues', [])
        if attribution_issues:
            recommendations.append("Improve source citation coverage and accuracy")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _generate_ethical_recommendations(
        self,
        bias_analysis: Dict[str, Any],
        discrimination_check: Dict[str, Any],
        misleading_claims_check: Dict[str, Any],
        disclaimer_check: Dict[str, Any]
    ) -> List[str]:
        """Generate ethical compliance recommendations."""
        
        recommendations = []
        
        # Bias recommendations
        bias_issues = bias_analysis.get('bias_issues_count', 0)
        if bias_issues > 0:
            recommendations.append("Review and revise language to reduce bias and promote inclusivity")
        
        # Discrimination recommendations
        discrimination_issues = discrimination_check.get('discrimination_issues_count', 0)
        if discrimination_issues > 0:
            recommendations.append("Remove discriminatory language and generalizations about groups")
        
        # Misleading claims recommendations
        misleading_claims = misleading_claims_check.get('misleading_claims_count', 0)
        if misleading_claims > 0:
            recommendations.append("Revise exaggerated or misleading claims to be more accurate")
        
        # Disclaimer recommendations
        missing_disclaimers = disclaimer_check.get('disclaimer_violations', 0)
        if missing_disclaimers > 0:
            recommendations.append("Add required disclaimers for medical, financial, or promotional content")
        
        # General ethical recommendations
        if not recommendations:
            recommendations.append("Content meets ethical standards - no major issues identified")
        
        return recommendations[:5]  # Limit to 5 recommendations