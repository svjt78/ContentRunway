"""PII scanning tool using Microsoft Presidio for detecting personally identifiable information."""

from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

# Microsoft Presidio imports
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Microsoft Presidio not available. Install with: pip install presidio-analyzer presidio-anonymizer")
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class PIIScannerTool:
    """Tool for detecting and handling personally identifiable information using Microsoft Presidio."""
    
    def __init__(self):
        """Initialize PII scanner with Presidio engines."""
        self.analyzer = None
        self.anonymizer = None
        
        if PRESIDIO_AVAILABLE:
            try:
                # Configure NLP engine (spaCy)
                nlp_configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
                }
                
                # Initialize analyzer
                provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
                nlp_engine = provider.create_engine()
                
                self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                self.anonymizer = AnonymizerEngine()
                
                # Add custom recognizers
                self._add_custom_recognizers()
                
                logger.info("Microsoft Presidio PII scanner initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {e}")
                logger.info("Falling back to regex-based PII detection")
                self.analyzer = None
                self.anonymizer = None
        
        # Fallback regex patterns for common PII
        self.fallback_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'SSN': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'CREDIT_CARD': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'IP_ADDRESS': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'API_KEY': r'\b[A-Za-z0-9]{32,}\b',  # Common API key pattern
            'URL_WITH_TOKENS': r'https?://[^\s]*[?&](token|key|auth|secret)=[A-Za-z0-9]+[^\s]*'
        }
        
        # Risk levels for different PII types
        self.pii_risk_levels = {
            'EMAIL_ADDRESS': 'medium',
            'PHONE_NUMBER': 'high',
            'SSN': 'critical',
            'CREDIT_CARD': 'critical',
            'PERSON': 'low',
            'LOCATION': 'low',
            'ORGANIZATION': 'low',
            'URL': 'low',
            'IP_ADDRESS': 'medium',
            'API_KEY': 'critical',
            'DATE_TIME': 'low',
            'NRP': 'low'  # Nationality/Religion/Political affiliation
        }
    
    def _add_custom_recognizers(self):
        """Add custom recognizers for domain-specific PII patterns."""
        
        if not self.analyzer:
            return
        
        # API Key recognizer
        api_key_pattern = Pattern(
            name="api_key_pattern",
            regex=r'\b[A-Za-z0-9]{32,}\b',
            score=0.8
        )
        
        api_key_recognizer = PatternRecognizer(
            supported_entity="API_KEY",
            patterns=[api_key_pattern],
            name="api_key_recognizer"
        )
        
        # Database connection string recognizer
        db_connection_pattern = Pattern(
            name="db_connection_pattern", 
            regex=r'(postgresql|mysql|mongodb)://[^\s]*:[^\s]*@[^\s]+',
            score=0.9
        )
        
        db_recognizer = PatternRecognizer(
            supported_entity="DB_CONNECTION",
            patterns=[db_connection_pattern],
            name="db_connection_recognizer"
        )
        
        # Add recognizers to analyzer
        try:
            self.analyzer.registry.add_recognizer(api_key_recognizer)
            self.analyzer.registry.add_recognizer(db_recognizer)
        except Exception as e:
            logger.warning(f"Failed to add custom recognizers: {e}")
    
    def scan_content_for_pii(
        self,
        content: str,
        anonymize: bool = False,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Scan content for personally identifiable information.
        
        Args:
            content: Text content to scan
            anonymize: Whether to return anonymized version
            include_context: Whether to include surrounding context for PII findings
            
        Returns:
            Dictionary with PII analysis results
        """
        logger.info("Starting PII scanning")
        
        try:
            if self.analyzer and PRESIDIO_AVAILABLE:
                # Use Presidio for PII detection
                pii_results = self._scan_with_presidio(content, anonymize, include_context)
            else:
                # Use fallback regex patterns
                pii_results = self._scan_with_fallback(content, anonymize, include_context)
            
            # Calculate risk assessment
            risk_assessment = self._calculate_pii_risk(pii_results['findings'])
            
            # Generate recommendations
            recommendations = self._generate_pii_recommendations(
                pii_results['findings'],
                risk_assessment
            )
            
            return {
                'pii_detected': len(pii_results['findings']) > 0,
                'total_pii_findings': len(pii_results['findings']),
                'findings': pii_results['findings'],
                'risk_assessment': risk_assessment,
                'anonymized_content': pii_results.get('anonymized_content'),
                'recommendations': recommendations,
                'scan_method': 'presidio' if self.analyzer else 'fallback',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"PII scanning failed: {e}")
            return {
                'pii_detected': False,
                'total_pii_findings': 0,
                'findings': [],
                'risk_assessment': {'overall_risk': 'unknown'},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _scan_with_presidio(
        self,
        content: str,
        anonymize: bool,
        include_context: bool
    ) -> Dict[str, Any]:
        """Scan content using Microsoft Presidio."""
        
        try:
            # Analyze content for PII
            analyzer_results = self.analyzer.analyze(
                text=content,
                entities=None,  # Detect all supported entities
                language='en',
                score_threshold=0.3  # Lower threshold to catch more potential PII
            )
            
            # Convert results to our format
            findings = []
            
            for result in analyzer_results:
                # Extract the actual PII text
                pii_text = content[result.start:result.end]
                
                # Get surrounding context if requested
                context = ""
                if include_context:
                    context_start = max(0, result.start - 50)
                    context_end = min(len(content), result.end + 50)
                    context = content[context_start:context_end]
                    
                    # Highlight the PII in context
                    context = context.replace(pii_text, f"**{pii_text}**")
                
                finding = {
                    'entity_type': result.entity_type,
                    'text': pii_text,
                    'start_position': result.start,
                    'end_position': result.end,
                    'confidence_score': round(result.score, 3),
                    'context': context,
                    'risk_level': self.pii_risk_levels.get(result.entity_type, 'medium'),
                    'recognition_source': result.recognition_metadata.get('recognizer_name', 'unknown') if result.recognition_metadata else 'unknown'
                }
                
                findings.append(finding)
            
            # Generate anonymized content if requested
            anonymized_content = None
            if anonymize and findings:
                try:
                    anonymizer_results = self.anonymizer.anonymize(
                        text=content,
                        analyzer_results=analyzer_results
                    )
                    anonymized_content = anonymizer_results.text
                except Exception as e:
                    logger.warning(f"Anonymization failed: {e}")
                    anonymized_content = self._fallback_anonymize(content, findings)
            
            return {
                'findings': findings,
                'anonymized_content': anonymized_content
            }
            
        except Exception as e:
            logger.error(f"Presidio scanning failed: {e}")
            raise
    
    def _scan_with_fallback(
        self,
        content: str,
        anonymize: bool,
        include_context: bool
    ) -> Dict[str, Any]:
        """Scan content using fallback regex patterns."""
        
        findings = []
        
        for entity_type, pattern in self.fallback_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                pii_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Get context if requested
                context = ""
                if include_context:
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(content), end_pos + 50)
                    context = content[context_start:context_end]
                    context = context.replace(pii_text, f"**{pii_text}**")
                
                finding = {
                    'entity_type': entity_type,
                    'text': pii_text,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'confidence_score': 0.8,  # Default confidence for regex matches
                    'context': context,
                    'risk_level': self.pii_risk_levels.get(entity_type, 'medium'),
                    'recognition_source': 'regex_fallback'
                }
                
                findings.append(finding)
        
        # Generate anonymized content if requested
        anonymized_content = None
        if anonymize and findings:
            anonymized_content = self._fallback_anonymize(content, findings)
        
        return {
            'findings': findings,
            'anonymized_content': anonymized_content
        }
    
    def _fallback_anonymize(self, content: str, findings: List[Dict[str, Any]]) -> str:
        """Fallback anonymization using simple replacement."""
        
        anonymized = content
        
        # Sort findings by position (reverse order to maintain positions)
        sorted_findings = sorted(findings, key=lambda x: x['start_position'], reverse=True)
        
        # Replacement patterns
        replacements = {
            'EMAIL': '[EMAIL_REDACTED]',
            'EMAIL_ADDRESS': '[EMAIL_REDACTED]',
            'PHONE_NUMBER': '[PHONE_REDACTED]',
            'SSN': '[SSN_REDACTED]',
            'CREDIT_CARD': '[CARD_REDACTED]',
            'IP_ADDRESS': '[IP_REDACTED]',
            'API_KEY': '[API_KEY_REDACTED]',
            'PERSON': '[NAME_REDACTED]',
            'LOCATION': '[LOCATION_REDACTED]',
            'ORGANIZATION': '[ORG_REDACTED]'
        }
        
        # Replace PII with placeholders
        for finding in sorted_findings:
            start = finding['start_position']
            end = finding['end_position']
            entity_type = finding['entity_type']
            
            replacement = replacements.get(entity_type, '[PII_REDACTED]')
            anonymized = anonymized[:start] + replacement + anonymized[end:]
        
        return anonymized
    
    def _calculate_pii_risk(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall PII risk assessment."""
        
        if not findings:
            return {
                'overall_risk': 'none',
                'risk_score': 0.0,
                'critical_findings': 0,
                'high_risk_findings': 0,
                'medium_risk_findings': 0,
                'low_risk_findings': 0
            }
        
        # Count findings by risk level
        risk_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for finding in findings:
            risk_level = finding.get('risk_level', 'medium')
            risk_counts[risk_level] += 1
        
        # Calculate overall risk score
        risk_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }
        
        total_weight = sum(risk_counts[level] * risk_weights[level] for level in risk_weights)
        max_possible_weight = len(findings)  # If all were critical
        
        risk_score = total_weight / max_possible_weight if max_possible_weight > 0 else 0.0
        
        # Determine overall risk level
        if risk_counts['critical'] > 0:
            overall_risk = 'critical'
        elif risk_counts['high'] > 2:
            overall_risk = 'high'
        elif risk_counts['high'] > 0 or risk_counts['medium'] > 3:
            overall_risk = 'medium'
        elif risk_counts['medium'] > 0 or risk_counts['low'] > 5:
            overall_risk = 'low'
        else:
            overall_risk = 'minimal'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': round(risk_score, 3),
            'critical_findings': risk_counts['critical'],
            'high_risk_findings': risk_counts['high'],
            'medium_risk_findings': risk_counts['medium'],
            'low_risk_findings': risk_counts['low'],
            'total_findings': len(findings),
            'most_common_pii_types': self._get_most_common_pii_types(findings)
        }
    
    def _get_most_common_pii_types(self, findings: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """Get most common PII types found in content."""
        
        from collections import Counter
        
        entity_counts = Counter(finding['entity_type'] for finding in findings)
        
        return [
            {'entity_type': entity_type, 'count': count}
            for entity_type, count in entity_counts.most_common(5)
        ]
    
    def _generate_pii_recommendations(
        self,
        findings: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for handling PII findings."""
        
        recommendations = []
        
        overall_risk = risk_assessment.get('overall_risk', 'none')
        
        if overall_risk == 'critical':
            recommendations.append("CRITICAL: Remove all critical PII before publishing - content contains sensitive personal data")
            recommendations.append("Review content manually to ensure no sensitive information remains")
        elif overall_risk == 'high':
            recommendations.append("HIGH RISK: Carefully review and remove high-risk PII before publishing")
        elif overall_risk == 'medium':
            recommendations.append("MEDIUM RISK: Review identified PII and consider if removal is necessary")
        elif overall_risk == 'low':
            recommendations.append("LOW RISK: Minimal PII detected - review identified items for context appropriateness")
        elif overall_risk == 'minimal':
            recommendations.append("MINIMAL RISK: Very low-risk PII detected - content appears safe for publishing")
        else:
            recommendations.append("NO PII DETECTED: Content appears safe from PII perspective")
        
        # Specific recommendations based on findings
        critical_findings = [f for f in findings if f.get('risk_level') == 'critical']
        if critical_findings:
            critical_types = set(f['entity_type'] for f in critical_findings)
            recommendations.append(f"Immediately address critical PII types: {', '.join(critical_types)}")
        
        high_risk_findings = [f for f in findings if f.get('risk_level') == 'high']
        if high_risk_findings:
            recommendations.append("Review and likely remove high-risk PII such as phone numbers and detailed personal information")
        
        # Email-specific recommendations
        email_findings = [f for f in findings if 'EMAIL' in f['entity_type']]
        if email_findings:
            if len(email_findings) == 1:
                recommendations.append("Consider if the email address is necessary for the content context")
            else:
                recommendations.append("Multiple email addresses detected - ensure they are necessary and appropriate")
        
        # Organization/Person name recommendations
        person_findings = [f for f in findings if f['entity_type'] in ['PERSON', 'ORGANIZATION']]
        if person_findings and overall_risk in ['low', 'minimal']:
            recommendations.append("Person/organization names detected - ensure proper context and consent if applicable")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def validate_content_safety(
        self,
        content: str,
        allowed_pii_types: List[str] = None,
        max_risk_level: str = 'medium'
    ) -> Dict[str, Any]:
        """
        Validate if content is safe for publishing based on PII criteria.
        
        Args:
            content: Content to validate
            allowed_pii_types: List of PII types that are acceptable
            max_risk_level: Maximum acceptable risk level
            
        Returns:
            Dictionary with safety validation results
        """
        
        allowed_pii_types = allowed_pii_types or ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE_TIME']
        
        # Scan for PII
        scan_results = self.scan_content_for_pii(content, anonymize=False, include_context=True)
        
        findings = scan_results.get('findings', [])
        risk_assessment = scan_results.get('risk_assessment', {})
        
        # Determine if content is safe
        is_safe = True
        blocking_issues = []
        warnings = []
        
        # Check overall risk level
        overall_risk = risk_assessment.get('overall_risk', 'none')
        risk_hierarchy = ['none', 'minimal', 'low', 'medium', 'high', 'critical']
        
        if risk_hierarchy.index(overall_risk) > risk_hierarchy.index(max_risk_level):
            is_safe = False
            blocking_issues.append(f"Overall PII risk level ({overall_risk}) exceeds maximum allowed ({max_risk_level})")
        
        # Check individual findings
        for finding in findings:
            entity_type = finding['entity_type']
            risk_level = finding.get('risk_level', 'medium')
            
            if entity_type not in allowed_pii_types:
                if risk_level in ['critical', 'high']:
                    is_safe = False
                    blocking_issues.append(f"Prohibited PII type detected: {entity_type} (risk: {risk_level})")
                else:
                    warnings.append(f"Potentially problematic PII detected: {entity_type}")
        
        # Check for critical PII regardless of settings
        critical_findings = [f for f in findings if f.get('risk_level') == 'critical']
        if critical_findings:
            is_safe = False
            for finding in critical_findings:
                blocking_issues.append(f"Critical PII must be removed: {finding['entity_type']}")
        
        return {
            'is_safe_for_publishing': is_safe,
            'safety_score': 1.0 - risk_assessment.get('risk_score', 0.0),
            'blocking_issues': blocking_issues,
            'warnings': warnings,
            'pii_summary': {
                'total_findings': len(findings),
                'critical_count': len([f for f in findings if f.get('risk_level') == 'critical']),
                'high_risk_count': len([f for f in findings if f.get('risk_level') == 'high']),
                'prohibited_types_found': [
                    f['entity_type'] for f in findings 
                    if f['entity_type'] not in allowed_pii_types
                ]
            },
            'recommendations': self._generate_safety_recommendations(
                is_safe, blocking_issues, warnings
            )
        }
    
    def _generate_safety_recommendations(
        self,
        is_safe: bool,
        blocking_issues: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generate safety recommendations based on validation results."""
        
        recommendations = []
        
        if not is_safe:
            recommendations.append("CONTENT NOT SAFE FOR PUBLISHING - Address all blocking issues before proceeding")
            
            if blocking_issues:
                recommendations.append("Review and remove/redact the following critical issues:")
                recommendations.extend([f"  - {issue}" for issue in blocking_issues[:3]])
        else:
            recommendations.append("Content appears safe for publishing from PII perspective")
        
        if warnings:
            recommendations.append("Consider reviewing the following potential concerns:")
            recommendations.extend([f"  - {warning}" for warning in warnings[:3]])
        
        if not blocking_issues and not warnings:
            recommendations.append("No PII concerns identified - content is clear for publishing")
        
        return recommendations
    
    def get_pii_categories_summary(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of PII categories found."""
        
        if not findings:
            return {
                'categories_found': [],
                'risk_distribution': {},
                'total_unique_categories': 0
            }
        
        from collections import Counter
        
        # Count by entity type
        entity_counts = Counter(finding['entity_type'] for finding in findings)
        
        # Count by risk level
        risk_counts = Counter(finding.get('risk_level', 'medium') for finding in findings)
        
        # Get unique categories with details
        categories_found = []
        for entity_type, count in entity_counts.items():
            # Get representative finding for this category
            sample_finding = next(f for f in findings if f['entity_type'] == entity_type)
            
            categories_found.append({
                'entity_type': entity_type,
                'count': count,
                'risk_level': sample_finding.get('risk_level', 'medium'),
                'avg_confidence': round(
                    sum(f['confidence_score'] for f in findings if f['entity_type'] == entity_type) / count,
                    3
                )
            })
        
        return {
            'categories_found': categories_found,
            'risk_distribution': dict(risk_counts),
            'total_unique_categories': len(entity_counts),
            'most_common_category': entity_counts.most_common(1)[0] if entity_counts else None
        }
    
    def create_pii_removal_plan(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a plan for removing or addressing PII findings."""
        
        if not findings:
            return {
                'removal_required': False,
                'plan_items': [],
                'estimated_effort': 'none'
            }
        
        plan_items = []
        
        # Group findings by risk level and type
        critical_findings = [f for f in findings if f.get('risk_level') == 'critical']
        high_risk_findings = [f for f in findings if f.get('risk_level') == 'high']
        medium_risk_findings = [f for f in findings if f.get('risk_level') == 'medium']
        low_risk_findings = [f for f in findings if f.get('risk_level') == 'low']
        
        # Critical items (must remove)
        for finding in critical_findings:
            plan_items.append({
                'priority': 'critical',
                'action': 'remove',
                'entity_type': finding['entity_type'],
                'text_to_remove': finding['text'],
                'reason': 'Critical PII must be removed before publishing',
                'position': f"Position {finding['start_position']}-{finding['end_position']}"
            })
        
        # High risk items (strongly recommend removal)
        for finding in high_risk_findings:
            plan_items.append({
                'priority': 'high',
                'action': 'review_and_likely_remove',
                'entity_type': finding['entity_type'],
                'text_to_review': finding['text'],
                'reason': 'High-risk PII should be removed unless essential for content',
                'position': f"Position {finding['start_position']}-{finding['end_position']}"
            })
        
        # Medium risk items (review for necessity)
        for finding in medium_risk_findings[:5]:  # Limit to 5 items
            plan_items.append({
                'priority': 'medium',
                'action': 'review',
                'entity_type': finding['entity_type'],
                'text_to_review': finding['text'],
                'reason': 'Review if this information is necessary for the content',
                'position': f"Position {finding['start_position']}-{finding['end_position']}"
            })
        
        # Estimate effort
        total_items = len(critical_findings) + len(high_risk_findings) + len(medium_risk_findings)
        
        if total_items == 0:
            effort = 'none'
        elif total_items <= 3:
            effort = 'minimal'
        elif total_items <= 8:
            effort = 'moderate'
        else:
            effort = 'significant'
        
        return {
            'removal_required': len(critical_findings) > 0 or len(high_risk_findings) > 0,
            'plan_items': plan_items,
            'estimated_effort': effort,
            'summary': {
                'critical_actions': len(critical_findings),
                'high_priority_actions': len(high_risk_findings),
                'review_actions': len(medium_risk_findings),
                'low_priority_items': len(low_risk_findings)
            }
        }
    
    def anonymize_content_for_review(self, content: str, anonymization_level: str = 'medium') -> Dict[str, Any]:
        """
        Create anonymized version of content for safe review.
        
        Args:
            content: Original content
            anonymization_level: Level of anonymization ('minimal', 'medium', 'aggressive')
            
        Returns:
            Dictionary with anonymized content and mapping
        """
        
        # Scan for PII
        scan_results = self.scan_content_for_pii(content, anonymize=False, include_context=False)
        findings = scan_results.get('findings', [])
        
        if not findings:
            return {
                'anonymized_content': content,
                'anonymization_applied': False,
                'pii_mapping': {},
                'safety_level': 'safe'
            }
        
        # Determine what to anonymize based on level
        entities_to_anonymize = set()
        
        if anonymization_level == 'minimal':
            # Only anonymize critical and high-risk PII
            entities_to_anonymize = {
                finding['entity_type'] for finding in findings 
                if finding.get('risk_level') in ['critical', 'high']
            }
        elif anonymization_level == 'medium':
            # Anonymize critical, high, and medium risk PII
            entities_to_anonymize = {
                finding['entity_type'] for finding in findings 
                if finding.get('risk_level') in ['critical', 'high', 'medium']
            }
        else:  # aggressive
            # Anonymize all PII
            entities_to_anonymize = {finding['entity_type'] for finding in findings}
        
        # Filter findings to anonymize
        findings_to_anonymize = [
            f for f in findings 
            if f['entity_type'] in entities_to_anonymize
        ]
        
        # Create anonymized version
        if findings_to_anonymize:
            if self.anonymizer and PRESIDIO_AVAILABLE:
                # Use Presidio anonymizer
                analyzer_results = []
                for finding in findings_to_anonymize:
                    # Convert to Presidio format (simplified)
                    analyzer_results.append({
                        'entity_type': finding['entity_type'],
                        'start': finding['start_position'],
                        'end': finding['end_position'],
                        'score': finding['confidence_score']
                    })
                
                try:
                    anonymized = self._fallback_anonymize(content, findings_to_anonymize)
                except Exception as e:
                    logger.warning(f"Presidio anonymization failed: {e}")
                    anonymized = self._fallback_anonymize(content, findings_to_anonymize)
            else:
                # Use fallback anonymization
                anonymized = self._fallback_anonymize(content, findings_to_anonymize)
        else:
            anonymized = content
        
        # Create PII mapping for reference
        pii_mapping = {}
        for i, finding in enumerate(findings_to_anonymize):
            placeholder_key = f"PII_{i+1}_{finding['entity_type']}"
            pii_mapping[placeholder_key] = {
                'original_text': finding['text'],
                'entity_type': finding['entity_type'],
                'risk_level': finding.get('risk_level', 'medium'),
                'position': f"{finding['start_position']}-{finding['end_position']}"
            }
        
        return {
            'anonymized_content': anonymized,
            'anonymization_applied': len(findings_to_anonymize) > 0,
            'pii_mapping': pii_mapping,
            'safety_level': 'safe' if len(findings_to_anonymize) == len([f for f in findings if f.get('risk_level') in ['critical', 'high']]) else 'review_needed',
            'anonymization_stats': {
                'original_pii_count': len(findings),
                'anonymized_pii_count': len(findings_to_anonymize),
                'anonymization_level': anonymization_level
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        if self.search_session and not self.search_session.closed:
            await self.search_session.close()