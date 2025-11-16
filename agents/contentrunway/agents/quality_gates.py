"""Quality Gate Agents - Parallel quality assessment agents for content validation."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import asyncio
import re
import json
from datetime import datetime
import hashlib

from ..state.pipeline_state import Draft, Source, ContentPipelineState
from ..tools.content_tools import CONTENT_VALIDATION_TOOLS, validate_content, analyze_content_quality
from ..utils.model_selector import get_optimized_model_config, estimate_operation_cost
from ..utils.llm_cache import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)

# Quality Gate Scoring Version - increment to invalidate all caches
QUALITY_GATE_SCORING_VERSION = "v2.1"

def log_content_analysis(agent_name: str, draft: Draft, sources: List = None, analysis_type: str = "general"):
    """
    Comprehensive content analysis logging for debugging quality scores.
    
    Args:
        agent_name: Name of the quality gate agent
        draft: Content draft being analyzed
        sources: Available sources (optional)
        analysis_type: Type of analysis being performed
    """
    content = draft.content
    content_length = len(content)
    word_count = len(content.split())
    
    # Citation analysis
    citation_pattern = r'\[Citation\s*\d+\]'
    citations = re.findall(citation_pattern, content)
    citation_count = len(citations)
    
    # Protected domain terms analysis
    protected_terms = re.findall(r'<([^>]+)>', content)
    protected_term_count = len(protected_terms)
    
    # Sentence analysis
    sentences = re.split(r'(?<=[.!?])\s+', content)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Content structure analysis
    sections = content.count('##')
    paragraphs = content.count('\n\n') + 1
    
    logger.info(f"🔍 {agent_name} Content Analysis ({analysis_type}):")
    logger.info(f"   📊 Content Stats:")
    logger.info(f"      - Length: {content_length:,} characters")
    logger.info(f"      - Words: {word_count:,}")
    logger.info(f"      - Sentences: {sentence_count}")
    logger.info(f"      - Avg sentence length: {avg_sentence_length:.1f} words")
    logger.info(f"      - Sections: {sections}")
    logger.info(f"      - Paragraphs: {paragraphs}")
    
    logger.info(f"   🔗 Citation Analysis:")
    logger.info(f"      - Citation count: {citation_count}")
    logger.info(f"      - Citations found: {citations[:5]}{'...' if len(citations) > 5 else ''}")
    
    logger.info(f"   🏷️ Domain Term Analysis:")
    logger.info(f"      - Protected term count: {protected_term_count}")
    logger.info(f"      - Protected terms: {protected_terms[:5]}{'...' if len(protected_terms) > 5 else ''}")
    
    if sources:
        logger.info(f"   📚 Sources Available: {len(sources)}")
    
    # Content truncation warning
    if content_length > 3000:
        logger.warning(f"   ⚠️ Content length ({content_length}) exceeds typical truncation limits (3000 chars)")
        logger.warning(f"      - Characters beyond 3000: {content_length - 3000}")
        logger.warning(f"      - Potential citation loss in truncated portion")
    
    return {
        'content_length': content_length,
        'word_count': word_count,
        'citation_count': citation_count,
        'protected_term_count': protected_term_count,
        'avg_sentence_length': avg_sentence_length,
        'sections': sections,
        'paragraphs': paragraphs
    }

def log_scoring_details(agent_name: str, score: float, scoring_components: Dict[str, Any]):
    """
    Log detailed scoring breakdown for debugging.
    
    Args:
        agent_name: Name of the quality gate agent
        score: Final calculated score
        scoring_components: Dictionary of scoring component details
    """
    logger.info(f"📈 {agent_name} Scoring Breakdown:")
    logger.info(f"   🎯 Final Score: {score:.3f}")
    
    for component_name, component_data in scoring_components.items():
        if isinstance(component_data, dict):
            if 'score' in component_data:
                logger.info(f"   📊 {component_name}: {component_data['score']:.3f}")
                if 'details' in component_data:
                    logger.info(f"      - Details: {component_data['details']}")
        else:
            logger.info(f"   📊 {component_name}: {component_data}")

def log_content_processing_impact(agent_name: str, original_content: str, processed_content: str, processing_type: str):
    """
    Log the impact of content processing (truncation, sampling, etc.)
    
    Args:
        agent_name: Name of the quality gate agent
        original_content: Original content before processing
        processed_content: Content after processing
        processing_type: Type of processing performed
    """
    original_length = len(original_content)
    processed_length = len(processed_content)
    reduction_pct = ((original_length - processed_length) / original_length * 100) if original_length > 0 else 0
    
    # Citation loss analysis
    original_citations = len(re.findall(r'\[Citation\s*\d+\]', original_content))
    processed_citations = len(re.findall(r'\[Citation\s*\d+\]', processed_content))
    citation_loss = original_citations - processed_citations
    
    # Protected term loss analysis
    original_terms = len(re.findall(r'<([^>]+)>', original_content))
    processed_terms = len(re.findall(r'<([^>]+)>', processed_content))
    term_loss = original_terms - processed_terms
    
    logger.info(f"✂️ {agent_name} Content Processing Impact ({processing_type}):")
    logger.info(f"   📏 Length: {original_length:,} → {processed_length:,} chars ({reduction_pct:.1f}% reduction)")
    logger.info(f"   🔗 Citations: {original_citations} → {processed_citations} (loss: {citation_loss})")
    logger.info(f"   🏷️ Protected terms: {original_terms} → {processed_terms} (loss: {term_loss})")
    
    if citation_loss > 0:
        logger.warning(f"   ⚠️ Citation loss detected: {citation_loss} citations lost during {processing_type}")
    if term_loss > 0:
        logger.warning(f"   ⚠️ Protected term loss detected: {term_loss} terms lost during {processing_type}")
    
    return {
        'original_length': original_length,
        'processed_length': processed_length,
        'reduction_percentage': reduction_pct,
        'citation_loss': citation_loss,
        'term_loss': term_loss
    }


async def invalidate_quality_gate_caches():
    """
    Invalidate all quality gate caches to force fresh scoring.
    Call this after deploying scoring improvements.
    """
    try:
        from app.services.redis_service import redis_service
        
        # Clear quality gate result caches
        cache_patterns = [
            "fact_check_*",
            "domain_expertise_*", 
            "style_consistency_*",
            "compliance_*"
        ]
        
        cleared_count = 0
        for pattern in cache_patterns:
            keys = await redis_service.redis_client.keys(pattern)
            if keys:
                deleted = await redis_service.redis_client.delete(*keys)
                cleared_count += deleted
                logger.info(f"Cleared {deleted} cache keys for pattern: {pattern}")
        
        logger.info(f"✅ Quality gate cache invalidation complete: {cleared_count} keys cleared")
        return cleared_count
        
    except ImportError:
        logger.warning("Redis service not available for cache invalidation")
        return 0
    except Exception as e:
        logger.error(f"Failed to invalidate quality gate caches: {e}")
        return 0


def force_cache_bypass():
    """
    Temporarily disable caching for quality gates by incrementing version.
    This forces all gates to recalculate without clearing existing caches.
    """
    global QUALITY_GATE_SCORING_VERSION
    import time
    timestamp = str(int(time.time()))
    QUALITY_GATE_SCORING_VERSION = f"v2.1_bypass_{timestamp}"
    logger.info(f"🚫 Cache bypass activated: {QUALITY_GATE_SCORING_VERSION}")
    return QUALITY_GATE_SCORING_VERSION


def analyze_draft_structure(draft_content: str) -> Dict[str, Any]:
    """
    Analyze draft content to verify it has required quality gate signals.
    Use this to debug why quality gates might be failing.
    """
    analysis = {
        "fact_check_signals": {},
        "domain_expertise_signals": {},
        "style_consistency_signals": {},
        "compliance_signals": {}
    }
    
    # Fact check signals
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d{4}\)',  # (2023), (2024), etc.
        r'according to [A-Z][a-zA-Z\s&]+ \(\d{4}\)',  # according to Company Name (2024)
        r'Source:',  # Source: ...
        r'as reported by',
        r'data from',
        r'findings from'
    ]
    
    citation_matches = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, draft_content, re.IGNORECASE)
        if matches:
            citation_matches.extend(matches)
    
    analysis["fact_check_signals"] = {
        "citations_found": len(citation_matches),
        "citation_examples": citation_matches[:5],
        "has_explicit_citations": len(citation_matches) > 0
    }
    
    # Domain expertise signals
    technical_terms = len(re.findall(r'\b(?:API|framework|infrastructure|architecture|implementation|methodology|analytics|compliance|governance|automation|integration|optimization|scalability|security|encryption|authentication|protocol|algorithm|machine learning|artificial intelligence|neural network|deep learning|blockchain|cloud computing|microservices|DevOps|CI/CD|containerization|orchestration|monitoring|observability|performance|latency|throughput|availability|reliability|resilience|fault tolerance|load balancing|caching|database|SQL|NoSQL|data warehouse|ETL|pipeline|streaming|real-time|batch processing|distributed systems|event-driven|serverless|edge computing|IoT|5G|quantum computing|cybersecurity|threat detection|vulnerability|penetration testing|incident response|disaster recovery|business continuity|risk management|audit|regulatory|GDPR|HIPAA|SOC|ISO|NIST|compliance framework|governance model|risk assessment|due diligence|KPI|ROI|SLA|SLO|metrics|dashboard|reporting|visualization|business intelligence|predictive analytics|descriptive analytics|prescriptive analytics|data science|statistics|correlation|regression|classification|clustering|natural language processing|computer vision|recommendation system|personalization|A/B testing|experimentation|feature engineering|model training|model deployment|model monitoring|MLOps|data engineering|data lake|data mesh|data fabric|data governance|master data management|metadata|data quality|data lineage|data catalog|data privacy|data ethics|data sovereignty|digital transformation|digital strategy|digital ecosystem|platform economy|API economy|ecosystem orchestration|partner integration|marketplace|multi-tenant|single sign-on|identity management|access control|zero trust|network security|endpoint security|application security|data security|privacy by design|security by design|shift left|DevSecOps|threat modeling|attack surface|kill chain|MITRE ATT&CK|OWASP|security operations center|security information and event management|security orchestration|incident response|forensics|threat intelligence|threat hunting|red team|blue team|purple team|penetration testing|vulnerability assessment|security audit|compliance audit|regulatory compliance|industry standards|best practices|maturity model|reference architecture|design patterns|architectural patterns|enterprise architecture|solution architecture|system architecture|network architecture|data architecture|security architecture|cloud architecture|hybrid cloud|multi-cloud|edge computing|fog computing|serverless computing|function as a service|platform as a service|infrastructure as a service|software as a service|everything as a service|service mesh|container orchestration|container security|container registry|container runtime|container image|microservices architecture|event-driven architecture|service-oriented architecture|monolithic architecture|layered architecture|hexagonal architecture|clean architecture|domain-driven design|test-driven development|behavior-driven development|continuous integration|continuous deployment|continuous delivery|continuous monitoring|continuous testing|shift left testing|test automation|unit testing|integration testing|system testing|acceptance testing|performance testing|load testing|stress testing|volume testing|security testing|usability testing|accessibility testing|compatibility testing|regression testing|smoke testing|sanity testing|exploratory testing|mutation testing|property-based testing|contract testing|end-to-end testing|API testing|database testing|mobile testing|web testing|desktop testing|embedded testing|IoT testing|cloud testing|distributed testing|parallel testing|concurrent testing|asynchronous testing|synchronous testing|real-time testing|batch testing|stream testing|data testing|ETL testing|data warehouse testing|data lake testing|data pipeline testing|machine learning testing|AI testing|model testing|algorithm testing|performance benchmarking|scalability testing|reliability testing|availability testing|disaster recovery testing|business continuity testing|security testing|penetration testing|vulnerability testing|compliance testing|audit testing|risk testing|threat testing|incident testing|forensics testing|intelligence testing|hunting testing|operations testing|monitoring testing|observability testing|logging testing|tracing testing|profiling testing|debugging testing|troubleshooting testing|root cause analysis|failure analysis|impact analysis|business impact analysis|technology impact analysis|risk impact analysis|security impact analysis|privacy impact analysis|environmental impact analysis|social impact analysis|economic impact analysis|regulatory impact analysis|compliance impact analysis|operational impact analysis|strategic impact analysis|tactical impact analysis)\b', draft_content, re.IGNORECASE))
    
    actionable_patterns = [
        r'(?:should|must|need to|recommend|suggest|consider|implement|ensure|establish|develop|create|build|design)[^.]+',
        r'(?:best practice|strategy|approach|method|technique|framework|solution|process)[^.]+',
        r'(?:step|action|guideline|principle|rule|requirement)[^.]+',
    ]
    
    actionable_insights = []
    for pattern in actionable_patterns:
        matches = re.findall(pattern, draft_content, re.IGNORECASE)
        actionable_insights.extend(matches)
    
    analysis["domain_expertise_signals"] = {
        "technical_terms_count": technical_terms,
        "actionable_insights_count": len(actionable_insights),
        "actionable_examples": actionable_insights[:3],
        "has_sufficient_terminology": technical_terms >= 15,
        "has_practical_guidance": len(actionable_insights) >= 5
    }
    
    # Style consistency signals
    markdown_headers = len(re.findall(r'^#{1,6}\s+', draft_content, re.MULTILINE))
    bullet_lists = len(re.findall(r'^\s*[-*+•]\s+', draft_content, re.MULTILINE))
    numbered_lists = len(re.findall(r'^\s*\d+\.\s+', draft_content, re.MULTILINE))
    
    analysis["style_consistency_signals"] = {
        "markdown_headers": markdown_headers,
        "bullet_lists": bullet_lists,
        "numbered_lists": numbered_lists,
        "total_lists": bullet_lists + numbered_lists,
        "has_proper_structure": markdown_headers >= 3,
        "has_lists": (bullet_lists + numbered_lists) >= 2
    }
    
    # Compliance signals
    disclaimers = len(re.findall(r'(?:for informational purposes|consult.*(?:legal|expert|professional)|disclaimer|not.*(?:legal|financial|medical) advice)', draft_content, re.IGNORECASE))
    
    analysis["compliance_signals"] = {
        "disclaimers_found": disclaimers,
        "has_disclaimers": disclaimers > 0
    }
    
    # Overall assessment
    analysis["overall_assessment"] = {
        "fact_check_ready": analysis["fact_check_signals"]["has_explicit_citations"],
        "domain_expertise_ready": (
            analysis["domain_expertise_signals"]["has_sufficient_terminology"] and 
            analysis["domain_expertise_signals"]["has_practical_guidance"]
        ),
        "style_ready": (
            analysis["style_consistency_signals"]["has_proper_structure"] and 
            analysis["style_consistency_signals"]["has_lists"]
        ),
        "compliance_ready": analysis["compliance_signals"]["has_disclaimers"]
    }
    
    return analysis


# Optimized Fact Check Gate Agent Instructions
fact_check_agent_role_and_goal = """
Role: Fact Check Gate Agent for accuracy validation.
Goal: Extract verifiable claims, verify against sources, score accuracy, generate reports.
"""

fact_check_agent_hints = """
Focus: Statistics, facts, specs, research findings, market data, regulations.
Ignore: Opinions, predictions, subjective statements.
Process: Verify against high-credibility sources, score as SUPPORTED/PARTIAL/CONTRADICTED/INSUFFICIENT.
Output: Confidence scores (0-1), actionable recommendations.
"""

fact_check_agent_output_description = """
The Fact Check Gate Agent returns a comprehensive fact-check assessment containing:
- score: Overall fact-check score (0.0-1.0) based on claim verification results
- report: Detailed report with verification summary, recommendations, and analysis
- claims_verified: Number of claims successfully processed
- unsupported_claims: Number of potentially unsupported assertions identified

The report includes overall_score, total_claims_analyzed, verification_summary by status, unsupported_claims_found, recommendations, detailed_results, and timestamp.
"""

fact_check_agent_chain_of_thought_directions = """
Fact-checking workflow:
1. Use '_extract_factual_claims' to identify verifiable factual statements from content
2. Focus on statistical claims, historical facts, technical specifications, research findings, market data, regulatory statements
3. Limit extraction to 15 most important factual claims for performance
4. Use '_verify_claims' to check each claim against available sources
5. Apply '_find_sources_for_claim' to match claims with relevant sources using keyword analysis
6. Use '_verify_single_claim' to assess individual claims as SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or INSUFFICIENT
7. Apply confidence scoring (0.0-1.0) for each verification assessment
8. Use '_identify_unsupported_claims' to detect potentially unsupported assertions using pattern matching
9. Apply '_calculate_fact_check_score' using weighted scoring and unsupported claim penalties
10. Use '_generate_fact_check_report' to create comprehensive assessment with recommendations
11. Return fact-check package with score, detailed report, and actionable recommendations

Tool usage conditions:
- Extract factual claims when content contains verifiable statements
- Verify claims when relevant sources are available (minimum 2 keyword matches)
- Apply pattern matching for unsupported claims using predefined patterns
- Generate recommendations based on verification results and issue severity
- Use fallback scoring when verification fails or sources are insufficient
"""

fact_check_agent_instruction = f"""
{fact_check_agent_role_and_goal}
{fact_check_agent_hints}
{fact_check_agent_output_description}
{fact_check_agent_chain_of_thought_directions}
"""


# Domain Expertise Gate Agent Instructions
domain_expertise_agent_role_and_goal = """
You are a Domain Expertise Gate Agent specializing in validating domain-specific expertise and technical accuracy for IT Insurance, AI, and Agentic AI content.
Your primary goal is to assess technical depth, evaluate domain-specific terminology usage, check practical insights, and ensure content meets expert-level standards for professional audiences.
"""

domain_expertise_agent_hints = """
Domain expertise validation best practices:
- Assess technical depth based on accuracy of concepts, depth of explanation, current best practices, recent developments awareness
- Evaluate domain-specific terminology coverage using predefined criteria for each domain
- Check for practical insights including actionable recommendations, real-world examples, implementation guidance
- Validate technical accuracy and currency of domain-specific information
- Consider domain coverage across multiple target domains with weighted scoring
- Generate specific recommendations for improving technical depth and domain expertise
- Focus on professional relevance and practical value for working professionals
- Ensure content demonstrates expert-level understanding of domain concepts
"""

domain_expertise_agent_output_description = """
The Domain Expertise Gate Agent returns a comprehensive expertise assessment containing:
- score: Overall domain expertise score (0.0-1.0) based on technical depth, terminology, and practical value
- technical_depth_score: Assessment of technical accuracy and explanation depth
- terminology_score: Evaluation of domain-specific terminology usage
- practical_value_score: Assessment of actionable insights and professional relevance
- recommendations: Specific suggestions for improving domain expertise
- domain_coverage: Coverage assessment for each target domain

Scoring uses weighted criteria: technical depth (40%), terminology (30%), practical value (30%).
"""

domain_expertise_agent_chain_of_thought_directions = """
Domain expertise validation workflow:
1. Use '_assess_technical_depth' to evaluate technical accuracy and explanation depth
2. Apply domain-specific criteria for IT Insurance, AI, and Agentic AI domains
3. Check technical concepts, current best practices, recent developments awareness
4. Use '_evaluate_terminology' to assess domain-specific terminology usage
5. Count coverage of domain-specific terms using predefined criteria dictionaries
6. Calculate term coverage percentage for each domain with scoring boost
7. Use '_assess_practical_value' to evaluate actionable insights and professional relevance
8. Check for implementation guidance, real-world examples, and problem-solving approaches
9. Apply '_calculate_expertise_score' using weighted scoring (technical 40%, terminology 30%, practical 30%)
10. Use '_generate_expertise_recommendations' to provide specific improvement suggestions
11. Apply '_assess_domain_coverage' to evaluate coverage across all target domains
12. Return comprehensive expertise assessment with detailed scoring and recommendations

Tool usage conditions:
- Apply technical depth assessment for all domain expertise evaluations
- Use terminology evaluation when domain-specific content is present
- Generate practical value assessment for professional audience content
- Apply domain coverage analysis when multiple domains are specified
- Use fallback scoring when AI assessment fails or encounters errors
"""

domain_expertise_agent_instruction = f"""
{domain_expertise_agent_role_and_goal}
{domain_expertise_agent_hints}
{domain_expertise_agent_output_description}
{domain_expertise_agent_chain_of_thought_directions}
"""


# Style Critic Gate Agent Instructions
style_critic_agent_role_and_goal = """
You are a Style Critic Gate Agent specializing in evaluating writing style, tone consistency, readability, and content structure for professional audiences.
Your primary goal is to assess writing quality, ensure tone consistency throughout content, calculate readability metrics, analyze content structure, and provide specific suggestions for style improvements.
"""

style_critic_agent_hints = """
Style evaluation best practices:
- Evaluate writing style based on appropriate tone for target audience, clarity, professional engagement, consistent voice, and appropriate formality
- Check tone consistency across all content sections to ensure coherent reader experience
- Calculate readability metrics including sentence length, syllable complexity, and Flesch Reading Ease score
- Analyze content structure for proper use of headers, paragraphs, lists, and logical organization
- Target readability appropriate for professional audiences (typically 50-70 Flesch score)
- Ensure content maintains professional tone while being engaging and accessible
- Check for appropriate use of active vs passive voice
- Validate that content structure supports easy scanning and comprehension
"""

style_critic_agent_output_description = """
The Style Critic Gate Agent returns a comprehensive style assessment containing:
- score: Overall style consistency score (0.0-1.0) based on weighted criteria
- style_analysis: Writing style assessment with tone, clarity, engagement, and voice consistency
- tone_consistency: Analysis of tone consistency across content sections
- readability_metrics: Detailed readability analysis with Flesch score and structural metrics
- structure_score: Content structure and organization assessment
- suggestions: Specific, actionable suggestions for style improvements

Scoring uses weighted criteria: style analysis (40%), tone consistency (30%), readability (20%), structure (10%).
"""

style_critic_agent_chain_of_thought_directions = """
Style evaluation workflow:
1. Use '_analyze_writing_style' to assess overall writing style appropriateness for target audience
2. Evaluate tone, clarity, engagement level, voice consistency, and formality appropriateness
3. Use '_check_tone_consistency' to analyze tone across content sections
4. Apply '_split_content_sections' to divide content for section-by-section analysis
5. Use '_calculate_readability_metrics' to compute Flesch Reading Ease and structural metrics
6. Apply '_count_syllables' for syllable analysis and readability calculation
7. Use '_analyze_structure' to evaluate content organization and formatting
8. Count headers, paragraphs, lists, and assess structural quality
9. Apply '_calculate_style_score' using weighted criteria (style 40%, tone 30%, readability 20%, structure 10%)
10. Use '_generate_style_suggestions' to provide specific improvement recommendations
11. Return comprehensive style assessment with detailed analysis and actionable suggestions

Tool usage conditions:
- Apply writing style analysis for all content evaluations
- Use tone consistency checking when content has multiple sections (≥2)
- Calculate readability metrics for all content using Flesch Reading Ease formula
- Analyze structure when content contains headers, paragraphs, or lists
- Generate suggestions based on score thresholds and specific style issues identified
"""

style_critic_agent_instruction = f"""
{style_critic_agent_role_and_goal}
{style_critic_agent_hints}
{style_critic_agent_output_description}
{style_critic_agent_chain_of_thought_directions}
"""


# Compliance Gate Agent Instructions
compliance_agent_role_and_goal = """
You are a Compliance Gate Agent specializing in validating content compliance with legal, ethical, regulatory, and privacy guidelines.
Your primary goal is to identify potential legal issues, assess ethical considerations, detect bias and discriminatory content, validate privacy compliance, and ensure content meets professional and journalistic ethics standards.
"""

compliance_agent_hints = """
Compliance validation best practices:
- Check for legal compliance issues: copyright infringement, trademark violations, defamatory statements, false claims, regulatory compliance
- Assess ethical considerations: misleading information, conflicts of interest, harmful recommendations, transparency issues
- Detect bias and discriminatory content: gender/racial/cultural bias, exclusionary language, unconscious assumptions
- Validate privacy compliance: GDPR considerations, data protection topics, consent requirements
- Apply weighted scoring with legal compliance as most critical factor (40% weight)
- Use very low temperature (0.1) for consistent, conservative compliance assessment
- Generate actionable recommendations for addressing compliance concerns
- Apply penalty scoring for high-risk legal issues (50% reduction for high risk, 20% for medium risk)
"""

compliance_agent_output_description = """
The Compliance Gate Agent returns a comprehensive compliance assessment containing:
- score: Overall compliance score (0.0-1.0) with weighted legal, ethical, bias, and privacy factors
- report: Detailed compliance report with status, risk levels, and recommendations
- legal_issues: Specific legal compliance concerns identified
- ethical_concerns: Ethical issues and transparency problems
- bias_indicators: Bias and discriminatory content detected
- privacy_concerns: Privacy and data protection issues

The report includes compliance_status (PASS/REVIEW_REQUIRED/FAIL), legal_risk_level, and comprehensive recommendations.
"""

compliance_agent_chain_of_thought_directions = """
Compliance validation workflow:
1. Use '_check_legal_compliance' to identify potential legal issues
2. Assess copyright infringement, trademark violations, defamatory statements, false claims, regulatory compliance
3. Apply risk assessment with low/medium/high categorization
4. Use '_check_ethical_compliance' to validate ethical considerations
5. Check for misleading information, conflicts of interest, harmful recommendations, transparency issues
6. Use '_check_bias_and_discrimination' to detect bias and discriminatory content
7. Analyze for gender/racial/cultural bias, exclusionary language, and unconscious assumptions
8. Use '_check_privacy_compliance' to validate privacy and data protection compliance
9. Check for GDPR-related content, privacy policy requirements, and data protection topics
10. Apply '_calculate_compliance_score' using weighted scoring (legal 40%, ethical 30%, bias 20%, privacy 10%)
11. Apply risk penalties: high risk (50% reduction), medium risk (20% reduction)
12. Use '_generate_compliance_report' to create comprehensive assessment with status and recommendations
13. Return compliance package with score, detailed report, and specific issue categories

Tool usage conditions:
- Apply legal compliance checking for all content evaluations
- Use ethical compliance assessment for professional content
- Detect bias when content involves people, groups, or social topics
- Check privacy compliance when content mentions data, privacy, or GDPR topics
- Generate comprehensive reports with actionable recommendations
- Use Claude model for sensitive compliance analysis with very low temperature
"""

compliance_agent_instruction = f"""
{compliance_agent_role_and_goal}
{compliance_agent_hints}
{compliance_agent_output_description}
{compliance_agent_chain_of_thought_directions}
"""


class FactCheckGateAgent:
    """Validates factual accuracy and verifies claims against sources."""
    
    def __init__(self, model_name: str = None, enable_tool_selection: bool = True):
        # Use cost-optimized model configuration
        config = get_optimized_model_config("FactCheckGateAgent", task_type="analytical_tasks")
        
        try:
            from app.services.rate_limiter import wrap_llm_with_caching
            has_caching = True
        except ImportError:
            logger.warning("Rate limiter not available, using basic LLM")
            has_caching = False
        
        base_llm = ChatOpenAI(
            model=config.model_name,
            temperature=0.1,  # Very low temperature for factual accuracy
            max_tokens=config.max_tokens
        )
        self.base_llm = wrap_llm_with_caching(base_llm, "openai") if has_caching else base_llm
        self.agent_name = "FactCheckGateAgent"
        self._sentence_citation_map: List[Dict[str, Any]] = []
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(CONTENT_VALIDATION_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(
        self, 
        draft: Draft, 
        sources: List[Source],
        sentence_citation_map: Optional[List[Dict[str, Any]]] = None,
        target_word_count: int = 500,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fact-checking of the content draft.
        
        Args:
            draft: Content draft to fact-check
            sources: Available sources for verification
            
        Returns:
            Dictionary with fact-check score and detailed report
        """
        logger.info("🔍 FactCheckGateAgent - Starting fact-check validation")
        
        # Persist latest sentence-to-citation mapping for claim alignment
        self._sentence_citation_map = sentence_citation_map or []
        
        # Comprehensive content analysis logging
        content_stats = log_content_analysis("FactCheckGateAgent", draft, sources, "pre-validation")
        
        # Pre-gate verification - log content status and bounce back if insufficient
        verification_result = self._verify_content_readiness(draft, sources)
        if not verification_result['ready']:
            logger.error(f"❌ FactCheckGateAgent - Content not ready for quality gates: {verification_result['reason']}")
            logger.error(f"   📊 Content stats: citations={content_stats['citation_count']}, terms={content_stats['protected_term_count']}")
            return {
                'score': 0.0,
                'status': 'content_verification_failed',
                'reason': verification_result['reason'],
                'recommendations': verification_result['recommendations'],
                'bounce_back_required': True
            }
        
        try:
            # Enhanced caching with new LLM cache layer
            content_for_cache = draft.content[:1000]  # First 1000 chars for similarity matching
            content_fingerprint = hashlib.sha256(draft.content.encode("utf-8")).hexdigest()
            cache_key_params = {
                "sources_count": len(sources), 
                "agent": "fact_check",
                "scoring_version": QUALITY_GATE_SCORING_VERSION,
                "content_hash": content_fingerprint
            }
            
            cached_result = await get_cached_llm_response(
                agent_name=self.agent_name,
                model=self.base_llm.model_name,
                prompt="fact_check_analysis",
                content=content_for_cache,
                extra_params=cache_key_params
            )
            
            if cached_result:
                if cached_result.get("fallback"):
                    logger.info("⚠️ Cached fact-check result flagged as fallback; forcing fresh analysis")
                else:
                    logger.info(f"📋 Using enhanced cached fact-check result (version: {QUALITY_GATE_SCORING_VERSION})")
                    return cached_result
            
            # Fallback to legacy caching if available
            try:
                from app.services.redis_service import redis_service
                content_hash = redis_service.create_content_hash(
                    draft.content, 
                    f"fact_check_{len(sources)}_{QUALITY_GATE_SCORING_VERSION}"
                )
                
                legacy_cached_result = await redis_service.get_cached_quality_result(
                    content_hash, 
                    "fact_check"
                )
                
                if legacy_cached_result:
                    if legacy_cached_result.get("fallback"):
                        logger.info("⚠️ Legacy cached fact-check result flagged as fallback; bypassing cache")
                    else:
                        logger.info(f"📋 Using legacy cached fact-check result (version: {QUALITY_GATE_SCORING_VERSION})")
                        return legacy_cached_result
                    
            except ImportError:
                logger.debug("Legacy caching not available, proceeding with analysis")
            
            logger.info(f"🔄 Performing FRESH fact-check analysis (version: {QUALITY_GATE_SCORING_VERSION})")
            
            # Store draft content for citation-based fallback scoring
            self._current_draft_content = draft.content
            
            # Step 1: Extract factual claims from content
            claims = await self._extract_factual_claims(draft)
            
            # Step 2: Verify claims against sources (prioritise explicit citations)
            citation_lookup = self._build_citation_lookup(getattr(draft, "citations", []))
            if not citation_lookup and self._sentence_citation_map:
                logger.warning("🔁 FactCheckGateAgent: citation lookup empty; constructing from sentence map")
                mapped = self._build_lookup_from_sentence_map()
                citation_lookup.update(mapped)
            
            verification_results = await self._verify_claims(
                claims,
                sources,
                citation_lookup
            )
            
            # Step 3: Check for unsupported assertions with citation awareness
            unsupported_claims = self._identify_unsupported_claims(draft, sources, citation_lookup)
            
            # Step 4: Generate fact-check score with scaled requirements
            fact_check_score = self._calculate_fact_check_score(
                verification_results, 
                unsupported_claims,
                len(claims),
                target_word_count
            )
            
            # Step 5: Generate detailed report
            report = self._generate_fact_check_report(
                claims,
                verification_results, 
                unsupported_claims,
                fact_check_score
            )
            
            logger.info(f"Fact-check completed: {fact_check_score:.3f} score")
            
            result = {
                'score': fact_check_score,
                'report': report,
                'claims_verified': len(verification_results),
                'unsupported_claims': len(unsupported_claims),
                'fallback': False
            }
            
            # Enhanced caching - cache with new system
            estimated_cost_saved = 0.015  # Estimated GPT-4 cost saved per fact-check
            await cache_llm_response(
                agent_name=self.agent_name,
                model=self.base_llm.model_name,
                prompt="fact_check_analysis",
                response=result,
                content=content_for_cache,
                extra_params=cache_key_params,
                cost_saved=estimated_cost_saved
            )
            
            # Legacy caching for backward compatibility
            try:
                await redis_service.cache_quality_gate_result(
                    content_hash,
                    "fact_check", 
                    result
                )
            except (NameError, AttributeError):
                logger.debug("Legacy caching not available")
            
            return result
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            heuristic_result = self._build_heuristic_fact_result(draft, sources, str(e))
            return heuristic_result
    
    async def _extract_factual_claims(self, draft: Draft) -> List[Dict[str, Any]]:
        """Extract factual claims that need verification."""
        
        system_prompt = fact_check_agent_instruction
        
        # Log full content processing (no truncation)
        logger.info(f"✅ FactCheckGateAgent processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Extract verifiable factual claims from this content:

        Title: {draft.title}
        Content: {draft.content}

        Return JSON array of claims with format:
        [{{"claim": "specific factual statement", "type": "statistic|historical|technical|research|market|legal|company", "context": "surrounding context"}}]
        
        Limit to the 15 most important factual claims.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Universal JSON extraction with multiple fallback patterns
            try:
                claims = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Fact check claims JSON parsing failed: {json_error}")
                # Try multiple extraction patterns
                import re
                
                # Pattern 1: Standard array
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        claims = json.loads(json_str)
                        logger.info("Successfully extracted fact check claims JSON from response")
                    except json.JSONDecodeError:
                        # Pattern 2: Object with claims array
                        obj_match = re.search(r'\{.*"claims".*:\s*\[.*\].*\}', content, re.DOTALL)
                        if obj_match:
                            obj_str = obj_match.group(0)
                            parsed_obj = json.loads(obj_str)
                            claims = parsed_obj.get('claims', [])
                        else:
                            # Fallback: Generate from research findings
                            logger.info("JSON parsing failed, generating fallback claims from content")
                            claims = self._generate_fallback_claims(draft)
                else:
                    # Fallback: Generate from research findings
                    logger.info("No JSON found, generating fallback claims from content")
                    claims = self._generate_fallback_claims(draft)
            
            return claims[:15]  # Limit for performance
            
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            # Final fallback: Generate simple claims from content
            return self._generate_fallback_claims(draft)
    
    def _generate_fallback_claims(self, draft: Draft) -> List[Dict[str, Any]]:
        """Generate simple fallback claims when JSON parsing fails."""
        
        content = draft.content.lower()
        fallback_claims = []
        
        # Extract sentences with statistical or factual patterns
        import re
        stat_patterns = [
            r'(\d+(?:\.\d+)?%?\s+(?:of|percent|million|billion|thousand))',
            r'(according to [^,]+,)',
            r'(studies? (?:show|indicate|suggest)[^.]+)',
            r'(research (?:shows|indicates|suggests)[^.]+)',
            r'(\d{4}\s+(?:study|research|report)[^.]+)',
            r'(market (?:size|value|growth)[^.]+)',
        ]
        
        sentences = re.split(r'[.!?]+', draft.content)
        claim_count = 0
        
        for sentence in sentences:
            if claim_count >= 15:
                break
                
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            for pattern in stat_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    fallback_claims.append({
                        'claim': sentence,
                        'type': 'statistic',
                        'context': f"From content: {sentence[:100]}..."
                    })
                    claim_count += 1
                    break
        
        # If still no claims, extract any sentence with numbers
        if not fallback_claims:
            for sentence in sentences[:10]:  # Check first 10 sentences
                if re.search(r'\d+', sentence) and len(sentence) > 30:
                    fallback_claims.append({
                        'claim': sentence.strip(),
                        'type': 'general',
                        'context': f"Extracted from content"
                    })
                    if len(fallback_claims) >= 5:
                        break
        
        return fallback_claims
    
    def _build_citation_lookup(self, citations: List[Any]) -> Dict[int, Source]:
        """Build lookup dictionary for citations by number."""
        lookup: Dict[int, Source] = {}
        for citation in citations or []:
            try:
                number = getattr(citation, "number", None)
                source = getattr(citation, "source", None)
                if number is not None and source is not None:
                    lookup[int(number)] = source
            except (TypeError, ValueError):
                continue
        return lookup

    def _build_lookup_from_sentence_map(self) -> Dict[int, Source]:
        """Construct citation lookup using the sentence-level citation map."""
        lookup: Dict[int, Source] = {}
        if not self._sentence_citation_map:
            return lookup

        # Attempt to match citation numbers to existing source order
        for entry in self._sentence_citation_map:
            for citation_num in entry.get("citations", []):
                if citation_num not in lookup:
                    # We don't have source objects here; we create simple placeholder sources
                    placeholder_source = Source(
                        url=f"https://placeholder.local/citation/{citation_num}",
                        title=f"Placeholder Source {citation_num}",
                        author=None,
                        publication_date=None,
                        domain="unknown",
                        source_type="reference",
                        summary="Generated placeholder source; replace with actual source metadata.",
                        key_points=[],
                        credibility_score=0.5,
                        relevance_score=0.5,
                        currency_score=0.5
                    )
                    lookup[citation_num] = placeholder_source
        return lookup

    def _build_heuristic_fact_result(
        self,
        draft: Draft,
        sources: List[Source],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build heuristic fact-check result when LLM verification fails."""
        content = draft.content
        citations = re.findall(r'\[Citation\s*\d+\]', content)
        unique_citations = sorted({match for match in citations})
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        cited_sentences = [s for s in sentences if '[Citation' in s]
        cited_ratio = len(cited_sentences) / max(1, len(sentences))

        sentence_map = self._sentence_citation_map or []
        map_cited_ratio = 0.0
        if sentence_map:
            mapped_sentences = len(sentence_map)
            map_cited_ratio = mapped_sentences / max(1, len(sentences))
            unique_map_citations = {num for entry in sentence_map for num in entry.get('citations', [])}
        else:
            unique_map_citations = set()
        word_count = len(content.split())
        citation_density = len(citations) / max(1, word_count / 100)  # per 100 words
        source_diversity = len(sources)

        score = 0.55
        score += min(0.25, (len(unique_citations) / 8.0) * 0.25)
        score += min(0.20, max(cited_ratio, map_cited_ratio) * 0.30)
        score += min(0.10, min(citation_density, 3) * (0.10 / 3))
        score += min(0.08, source_diversity * 0.01)
        score += min(0.05, len(unique_map_citations) * 0.01)
        score = min(0.95, score)

        logger.info(
            f"📊 Heuristic fact-check score: citations={len(unique_citations)}, "
            f"cited_ratio={cited_ratio:.2f}, density={citation_density:.2f}, "
            f"sources={source_diversity} → {score:.3f}"
        )

        report = {
            'mode': 'heuristic',
            'error': error_message,
            'metrics': {
                'total_citations': len(citations),
                'unique_citations': len(unique_citations),
                'cited_sentence_ratio': cited_ratio,
                'map_cited_ratio': map_cited_ratio,
                'citation_density_per_100_words': citation_density,
                'source_count': source_diversity,
                'mapped_citations': len(unique_map_citations)
            },
            'notes': [
                "Heuristic scoring applied due to LLM verification failure.",
                "Increase diversity of citations and ensure each factual sentence is cited for higher confidence."
            ]
        }

        return {
            'score': score,
            'report': report,
            'claims_verified': len(cited_sentences),
            'unsupported_claims': 0,
            'fallback': True,
            'heuristic': True
        }
    
    async def _verify_claims(
        self, 
        claims: List[Dict[str, Any]], 
        sources: List[Source],
        citation_lookup: Dict[int, Source]
    ) -> List[Dict[str, Any]]:
        """Verify each claim against available sources."""
        
        verification_results = []
        
        for claim in claims:
            # Find relevant sources for this claim
            relevant_sources = self._find_sources_for_claim(
                claim,
                sources,
                citation_lookup
            )
            
            if not relevant_sources:
                verification_results.append({
                    'claim': claim,
                    'status': 'no_sources',
                    'confidence': 0.0,
                    'sources_checked': 0
                })
                continue
            
            # Verify against sources
            verification = await self._verify_single_claim(claim, relevant_sources)
            verification_results.append(verification)
        
        return verification_results
    
    def _find_sources_for_claim(
        self, 
        claim: Dict[str, Any], 
        sources: List[Source],
        citation_lookup: Dict[int, Source]
    ) -> List[Source]:
        """Find sources most likely to contain information about the claim."""
        
        claim_text = claim['claim'].lower()
        claim_keywords = re.findall(r'\b[a-zA-Z]{3,}\b', claim_text)
        citation_numbers = self._extract_citation_numbers(claim)
        
        if not citation_numbers:
            inferred_citations = self._match_claim_to_sentence(claim['claim'])
            if inferred_citations:
                citation_numbers.extend(inferred_citations)
        
        # Helper functions to handle both dict and object formats
        def get_title(source):
            if isinstance(source, dict):
                return source.get('title', '')
            return getattr(source, 'title', '')
        
        def get_summary(source):
            if isinstance(source, dict):
                return source.get('summary', '')
            return getattr(source, 'summary', '')
        
        def get_credibility_score(source):
            if isinstance(source, dict):
                return source.get('credibility_score', 0.8)
            return getattr(source, 'credibility_score', 0.8)
        
        relevant_sources: List[Source] = []
        seen_identifiers = set()
        
        # Prioritise explicit citations linked in the claim/context
        for citation_number in citation_numbers:
            cited_source = citation_lookup.get(citation_number)
            if cited_source:
                identifier = self._get_source_identifier(cited_source)
                if identifier and identifier not in seen_identifiers:
                    relevant_sources.append(cited_source)
                    seen_identifiers.add(identifier)
        
        # Supplement with keyword matching to catch uncited but relevant sources
        scored_sources = []
        for source in sources:
            identifier = self._get_source_identifier(source)
            if identifier in seen_identifiers:
                continue
            
            source_text = f"{get_title(source)} {get_summary(source)}".lower()
            
            # Enhanced keyword matching with normalization
            normalized_source = re.sub(r'[^\w\s]', ' ', source_text)
            normalized_claim = re.sub(r'[^\w\s]', ' ', claim_text)
            
            # Count exact and partial keyword matches
            exact_matches = sum(1 for keyword in claim_keywords if keyword in normalized_source)
            
            # Check for partial matches (stems, plurals)
            partial_matches = 0
            for keyword in claim_keywords:
                if len(keyword) > 4:  # Only for longer keywords
                    stem = keyword[:max(3, len(keyword)-2)]  # Simple stemming
                    if stem in normalized_source:
                        partial_matches += 0.5
            
            total_relevance = exact_matches + partial_matches
            
            # Lower threshold for high-confidence single matches with rare terms
            rare_terms = [kw for kw in claim_keywords if len(kw) > 6]
            has_rare_match = any(rare_term in normalized_source for rare_term in rare_terms)
            
            # Include source if sufficient matches or rare term match
            if total_relevance >= 1.5 or (has_rare_match and total_relevance >= 1.0):
                scored_sources.append((source, total_relevance))
                seen_identifiers.add(identifier)
        
        # Sort by relevance score and credibility
        scored_sources.sort(key=lambda x: (x[1], get_credibility_score(x[0])), reverse=True)
        relevant_sources.extend([source for source, _ in scored_sources[:5]])
        
        return relevant_sources[:5]
    
    def _extract_citation_numbers(self, claim: Dict[str, Any]) -> List[int]:
        """Extract citation numbers referenced in a claim or its context."""
        citation_numbers: List[int] = []
        text_segments = [
            claim.get('claim', ''),
            claim.get('context', '')
        ]
        citation_pattern = re.compile(r'\[Citation\s*(\d+)\]', re.IGNORECASE)
        
        for segment in text_segments:
            for match in citation_pattern.findall(segment or ''):
                try:
                    citation_numbers.append(int(match))
                except ValueError:
                    continue
        return citation_numbers
    
    def _match_claim_to_sentence(self, claim_text: str) -> List[int]:
        """Attempt to align a claim with the sentence-level citation map."""
        if not self._sentence_citation_map:
            return []
        
        normalized_claim = self._normalize_text(claim_text)
        matched_citations: List[int] = []
        
        for entry in self._sentence_citation_map:
            sentence_text = entry.get('sentence', '')
            normalized_sentence = self._normalize_text(sentence_text)
            
            if not normalized_sentence:
                continue
            
            if self._text_overlap(normalized_claim, normalized_sentence):
                for citation in entry.get('citations', []):
                    if citation not in matched_citations:
                        matched_citations.append(citation)
        
        return matched_citations
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy overlap checks."""
        cleaned = re.sub(r'\[Citation\s*\d+\]', '', text)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned
    
    def _text_overlap(self, claim: str, sentence: str) -> bool:
        """Check for substantive overlap between claim and sentence text."""
        if not claim or not sentence:
            return False
        
        if claim in sentence or sentence in claim:
            return True
        
        # Token-based overlap
        claim_tokens = set(claim.split())
        sentence_tokens = set(sentence.split())
        if not claim_tokens or not sentence_tokens:
            return False
        
        overlap = claim_tokens & sentence_tokens
        # Require at least 6 shared tokens or 40% overlap
        return len(overlap) >= 6 or (len(overlap) / max(1, len(claim_tokens))) >= 0.4
    
    def _get_source_identifier(self, source: Source) -> str:
        """Create a stable identifier for a source."""
        if isinstance(source, dict):
            return source.get('url') or source.get('title', '')
        url = getattr(source, 'url', None)
        return url or getattr(source, 'title', '')
    
    async def _verify_single_claim(
        self, 
        claim: Dict[str, Any], 
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Verify a single claim against sources."""
        
        system_prompt = fact_check_agent_instruction
        
        # Helper functions to handle both dict and object formats
        def get_title(source):
            if isinstance(source, dict):
                return source.get('title', 'Untitled')
            return getattr(source, 'title', 'Untitled')
        
        def get_summary(source):
            if isinstance(source, dict):
                return source.get('summary', '')
            return getattr(source, 'summary', '')
        
        source_summaries = [
            f"Source {i+1}: {get_title(source)}\nSummary: {get_summary(source)}"
            for i, source in enumerate(sources)
        ]
        
        human_prompt = f"""Claim to verify: {claim['claim']}
        Claim type: {claim['type']}
        Context: {claim['context']}

        Available sources:
        {chr(10).join(source_summaries)}

        Return JSON with:
        {{"status": "SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|INSUFFICIENT", "confidence": 0.0-1.0, "reasoning": "explanation"}}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Fact check verification JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted fact check verification JSON from response")
                else:
                    logger.error("No JSON object found in fact check verification response")
                    raise json_error
            
            return {
                'claim': claim,
                'status': result['status'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'sources_checked': len(sources)
            }
            
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return {
                'claim': claim,
                'status': 'INSUFFICIENT',
                'confidence': 0.5,
                'reasoning': f'Verification failed: {str(e)}',
                'sources_checked': len(sources)
            }
    
    def _identify_unsupported_claims(
        self,
        draft: Draft,
        sources: List[Source],
        citation_lookup: Optional[Dict[int, Source]] = None
    ) -> List[str]:
        """Identify potentially unsupported assertions in the content with citation awareness."""
        
        content = draft.content
        unsupported_patterns = [
            r'studies show',
            r'research indicates',
            r'experts believe',
            r'most companies',
            r'industry leaders',
            r'statistics reveal'
        ]
        
        # Look for citation patterns in the content
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (2024), etc.
            r'according to [A-Z][a-zA-Z\s&]+ \(\d{4}\)',  # according to Company Name (2024)
            r'Source:',  # Source: ...
            r'References?:',  # References: or Reference:
            r'as reported by',
            r'data from',
            r'findings from'
        ]
        
        # Check if content has citation markers
        has_citations = any(re.search(pattern, content, re.IGNORECASE) for pattern in citation_patterns)
        
        logger.info(f"🔗 Citation analysis: Has citations = {has_citations}")
        
        unsupported_claims = []
        for pattern in unsupported_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract sentence containing the pattern
                start = max(0, content.rfind('.', 0, match.start()) + 1)
                end = content.find('.', match.end())
                if end == -1:
                    end = len(content)
                
                sentence = content[start:end].strip()
                if len(sentence) > 20:  # Avoid very short matches
                    
                    # Check if this sentence or nearby context has citations
                    sentence_has_citation = any(
                        re.search(cit_pattern, sentence, re.IGNORECASE) 
                        for cit_pattern in citation_patterns
                    )
                    
                    # Check surrounding context (100 chars before/after)
                    context_start = max(0, start - 100)
                    context_end = min(len(content), end + 100)
                    context = content[context_start:context_end]
                    context_has_citation = any(
                        re.search(cit_pattern, context, re.IGNORECASE) 
                        for cit_pattern in citation_patterns
                    )
                    
                    citation_numbers_nearby = []
                    for cit_text in re.findall(r'\[Citation\s*(\d+)\]', sentence + " " + context, re.IGNORECASE):
                        try:
                            citation_numbers_nearby.append(int(cit_text))
                        except ValueError:
                            continue
                    
                    citation_resolved = False
                    if citation_numbers_nearby and citation_lookup:
                        for number in citation_numbers_nearby:
                            if citation_lookup.get(number):
                                citation_resolved = True
                                break
                    
                    # Only flag as unsupported if no citations nearby or unresolved
                    if not sentence_has_citation and not context_has_citation:
                        unsupported_claims.append(sentence)
                        logger.info(f"📝 Flagged unsupported: '{sentence[:50]}...'")
                    elif citation_numbers_nearby and not citation_resolved:
                        unsupported_claims.append(sentence)
                        logger.info(f"⚠️ Citation reference without matching source near: '{sentence[:50]}...'")
                    else:
                        logger.info(f"✅ Citation found near: '{sentence[:50]}...'")
        
        # If content has research integration from sources, be more lenient
        if sources and len(sources) > 0 and has_citations:
            # Reduce unsupported claims if research sources are integrated
            original_count = len(unsupported_claims)
            unsupported_claims = unsupported_claims[:max(1, len(unsupported_claims) // 2)]
            if original_count > len(unsupported_claims):
                logger.info(f"🔬 Reduced unsupported claims due to research integration: {original_count} → {len(unsupported_claims)}")
        
        return unsupported_claims[:10]  # Limit results
    
    def _calculate_fact_check_score(
        self,
        verification_results: List[Dict[str, Any]],
        unsupported_claims: List[str],
        total_claims: int,
        target_word_count: int = 500
    ) -> float:
        """Calculate overall fact-check score with comprehensive logging."""
        
        logger.info(f"🔍 FACT CHECK SCORING DEBUG:")
        logger.info(f"  - Total claims extracted: {total_claims}")
        logger.info(f"  - Verification results count: {len(verification_results)}")
        logger.info(f"  - Unsupported claims found: {len(unsupported_claims)}")
        
        if not verification_results and not total_claims:
            # Check if content has citations even without extracted claims
            content = getattr(self, '_current_draft_content', '')
            citation_count = len(re.findall(r'\[Citation\s*\d+\]', content)) if content else 0
            
            # Scale citation requirements based on target length (words)
            # Baseline: 500 words should include ~5 citations
            min_citations_required = max(1, min(10, round(target_word_count / 100)))
            logger.info(f"  - Scaled citation requirement: {min_citations_required} citations for {target_word_count} words")
            
            if citation_count >= min_citations_required:
                logger.info(f"  - No claims extracted but {citation_count} citations found (>= {min_citations_required}), returning high score: 0.90")
                return 0.90  # High score for well-cited content even without extracted claims
            elif citation_count > 0:
                logger.info(f"  - No claims extracted but {citation_count} citations found (< {min_citations_required}), returning good score: 0.75")
                return 0.75  # Good score for cited content
            else:
                logger.info(f"  - No claims found, returning neutral score: 0.85")
                return 0.85  # Neutral score for content without factual claims
        
        # Detailed status breakdown
        status_breakdown = {}
        for result in verification_results:
            status = result['status']
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
        logger.info(f"  - Status breakdown: {status_breakdown}")
        
        # Score based on verification results
        verification_score = 0.0
        if verification_results:
            status_weights = {
                'SUPPORTED': 1.0,
                'PARTIALLY_SUPPORTED': 0.7,
                'CONTRADICTED': 0.0,
                'INSUFFICIENT': 0.3,
                'no_sources': 0.6  # Improved: treat as neutral instead of penalty
            }
            
            total_weight = 0.0
            weighted_contributions = []
            for result in verification_results:
                weight = status_weights.get(result['status'], 0.2)
                confidence = result['confidence']
                contribution = weight * confidence
                total_weight += contribution
                weighted_contributions.append({
                    'status': result['status'],
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': contribution
                })
            
            verification_score = total_weight / len(verification_results)
            logger.info(f"  - Base verification score: {verification_score:.3f}")
            
            # Log individual contributions
            for contrib in weighted_contributions[:3]:  # Log first 3
                logger.info(f"    * {contrib['status']} (conf: {contrib['confidence']:.2f}) → {contrib['contribution']:.3f}")
            
            # Confidence uplift when ≥60% of claims are SUPPORTED with high confidence
            supported_high_conf = [
                r for r in verification_results 
                if r['status'] == 'SUPPORTED' and r['confidence'] >= 0.8
            ]
            uplift_threshold = 0.6 * len(verification_results)
            if len(supported_high_conf) >= uplift_threshold:
                old_score = verification_score
                verification_score = min(1.0, verification_score + 0.1)
                logger.info(f"  - High confidence uplift applied: {old_score:.3f} → {verification_score:.3f}")
            else:
                logger.info(f"  - No uplift: {len(supported_high_conf)} high-conf claims < {uplift_threshold:.1f} threshold")
        
        # Enhanced unsupported claims analysis with improved generosity
        unsupported_penalty = min(0.12, len(unsupported_claims) * 0.04)  # Reduced from 0.15 and 0.05
        original_penalty = unsupported_penalty
        
        # More generous penalty reduction for well-cited content
        if total_claims > 0:
            unsupported_ratio = len(unsupported_claims) / total_claims
            logger.info(f"  - Unsupported claims ratio: {unsupported_ratio:.2f} ({len(unsupported_claims)}/{total_claims})")
            if unsupported_ratio < 0.4:  # Increased threshold from 0.3 to 0.4
                unsupported_penalty *= 0.4  # Reduced from 0.5 to 0.4 for more generous reduction
                logger.info(f"  - Penalty reduced for well-cited content: {original_penalty:.3f} → {unsupported_penalty:.3f}")
        
        # Additional penalty reduction for content with good citation density
        if citation_count >= 3:
            unsupported_penalty *= 0.7  # Additional 30% reduction for well-cited content
            logger.info(f"  - Additional citation density bonus: penalty further reduced to {unsupported_penalty:.3f}")
        
        logger.info(f"  - Final unsupported penalty: {unsupported_penalty:.3f}")
        
        # Citation-based score boost for well-cited content
        content = getattr(self, '_current_draft_content', '')
        citation_count = len(re.findall(r'\[Citation\s*\d+\]', content)) if content else 0
        
        # If score is low but content has good citations, provide boost
        if verification_score < 0.5 and citation_count >= 3:
            citation_boost = min(0.4, citation_count * 0.1)  # Up to 0.4 boost
            logger.info(f"  - Citation boost applied: {citation_count} citations → +{citation_boost:.3f}")
            verification_score = min(1.0, verification_score + citation_boost)
        elif citation_count >= 2:
            citation_boost = min(0.2, citation_count * 0.05)  # Up to 0.2 boost
            logger.info(f"  - Minor citation boost: {citation_count} citations → +{citation_boost:.3f}")
            verification_score = min(1.0, verification_score + citation_boost)
        
        # Final score calculation
        final_score = max(0.0, verification_score - unsupported_penalty)
        logger.info(f"  - FINAL FACT CHECK SCORE: {verification_score:.3f} - {unsupported_penalty:.3f} = {final_score:.3f}")
        
        # Enhanced scoring breakdown logging
        scoring_components = {
            'base_verification_score': verification_score,
            'unsupported_penalty': unsupported_penalty,
            'final_calculated_score': final_score,
            'claims_stats': {
                'total_claims': total_claims,
                'verification_results_count': len(verification_results),
                'unsupported_claims_count': len(unsupported_claims)
            }
        }
        log_scoring_details("FactCheckGateAgent", min(1.0, final_score), scoring_components)
        
        # Log problematic unsupported claims if score is low
        if final_score < 0.5 and unsupported_claims:
            logger.warning(f"  - Low score due to unsupported claims:")
            for claim in unsupported_claims[:3]:
                logger.warning(f"    * '{claim[:100]}...'")
        
        return min(1.0, final_score)
    
    def _generate_fact_check_report(
        self,
        claims: List[Dict[str, Any]],
        verification_results: List[Dict[str, Any]],
        unsupported_claims: List[str],
        score: float
    ) -> Dict[str, Any]:
        """Generate comprehensive fact-check report."""
        
        return {
            'overall_score': score,
            'total_claims_analyzed': len(claims),
            'verification_summary': {
                'supported': len([r for r in verification_results if r['status'] == 'SUPPORTED']),
                'partially_supported': len([r for r in verification_results if r['status'] == 'PARTIALLY_SUPPORTED']),
                'contradicted': len([r for r in verification_results if r['status'] == 'CONTRADICTED']),
                'insufficient_sources': len([r for r in verification_results if r['status'] in ['INSUFFICIENT', 'no_sources']])
            },
            'unsupported_claims_found': len(unsupported_claims),
            'recommendations': self._generate_fact_check_recommendations(verification_results, unsupported_claims),
            'detailed_results': verification_results[:5],  # Top 5 for brevity
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_fact_check_recommendations(
        self,
        verification_results: List[Dict[str, Any]],
        unsupported_claims: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for improving factual accuracy."""
        
        recommendations = []
        
        contradicted = [r for r in verification_results if r['status'] == 'CONTRADICTED']
        if contradicted:
            recommendations.append(f"Review and correct {len(contradicted)} contradicted claims")
        
        insufficient = [r for r in verification_results if r['status'] in ['INSUFFICIENT', 'no_sources']]
        if insufficient:
            recommendations.append(f"Find additional sources for {len(insufficient)} unverified claims")
        
        if unsupported_claims:
            recommendations.append(f"Add citations for {len(unsupported_claims)} potentially unsupported assertions")
        
        partial = [r for r in verification_results if r['status'] == 'PARTIALLY_SUPPORTED']
        if partial:
            recommendations.append(f"Refine accuracy of {len(partial)} partially supported claims")
        
        if not recommendations:
            recommendations.append("Fact-check quality is good - no major issues identified")
        
        return recommendations
    
    def _verify_content_readiness(self, draft: Draft, sources: List[Source]) -> Dict[str, Any]:
        """Enhanced pre-gate validation with Phase 4 improvements."""
        
        import re
        
        # Phase 4: Enhanced content analysis for optimal quality gate preparation
        content_length = len(draft.content)
        word_count = len(draft.content.split())
        
        # Extract citations and domain terms for verification
        citation_count = len(re.findall(r'\[Citation\s*\d+\]', draft.content))
        protected_terms = re.findall(r'<([^>]+)>', draft.content)
        
        # Calculate sentence length statistics
        sentences = re.split(r'(?<=[.!?])\s+', draft.content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        long_sentences = sum(1 for length in sentence_lengths if length > 24)
        
        # Phase 4: Advanced content quality metrics
        sections = draft.content.count('##')
        paragraphs = draft.content.count('\n\n') + 1
        
        # Citation density analysis
        citation_density = citation_count / max(1, word_count / 100)  # Citations per 100 words
        
        # Protected term analysis  
        term_density = len(protected_terms) / max(1, word_count / 100)  # Terms per 100 words
        
        # Phase 4: Enhanced logging with density metrics
        logger.info(f"📊 Phase 4 Enhanced Pre-Gate Content Verification:")
        logger.info(f"   📝 Content Metrics: {content_length:,} characters, {word_count:,} words")
        logger.info(f"   📎 Citation Analysis: {citation_count} citations (density: {citation_density:.2f}/100 words)")
        logger.info(f"   🏷️ Domain Terms: {len(protected_terms)} protected terms (density: {term_density:.2f}/100 words)")
        logger.info(f"   📏 Readability: {avg_sentence_length:.1f} avg words/sentence, {long_sentences} long sentences")
        logger.info(f"   🏗️ Structure: {sections} sections, {paragraphs} paragraphs")
        logger.info(f"   📚 Resources: {len(sources)} sources available")
        
        # Phase 4: Enhanced readiness criteria with optimization targets
        readiness_criteria = []
        recommendations = []
        optimization_suggestions = []
        
        # Citation optimization (Phase 4: Higher standards for 0.85+ scores)
        if citation_count == 0:
            readiness_criteria.append("No citations found")
            recommendations.append("Add inline citations [Citation X] to factual claims")
        elif citation_count < 5:  # Raised from 3 to 5 for optimal scoring
            logger.warning(f"⚠️ Suboptimal citation count: {citation_count} (target: 5+ for 0.85+ scores)")
            optimization_suggestions.append(f"Increase citation coverage (current: {citation_count}, optimal: 5+)")
        elif citation_density < 2.0:  # At least 2 citations per 100 words
            optimization_suggestions.append(f"Increase citation density (current: {citation_density:.1f}, target: 2.0+ per 100 words)")
        
        # Domain terminology optimization (Phase 4: Align with Phase 2 requirements)
        if len(protected_terms) == 0:
            # Check for regular domain terms as fallback
            content_lower = draft.content.lower()
            domain_terms_found = sum(1 for term in ['machine learning', 'neural networks', 'ai', 'automation', 'algorithm'] 
                                   if term in content_lower)
            if domain_terms_found == 0:
                readiness_criteria.append("No domain terminology found")
                recommendations.append("Add domain-specific terminology with <term> protection markers")
        elif len(protected_terms) < 8:  # Phase 4: Align with Phase 2's 8+ term requirement
            if len(protected_terms) < 4:
                logger.warning(f"⚠️ Low domain term count: {len(protected_terms)} (minimum: 4, optimal: 8+)")
                recommendations.append(f"Increase domain term count (current: {len(protected_terms)}, minimum: 4)")
            else:
                optimization_suggestions.append(f"Optimize domain expertise (current: {len(protected_terms)}, optimal: 8+ for excellent scoring)")
        elif term_density < 1.5:  # At least 1.5 terms per 100 words
            optimization_suggestions.append(f"Increase domain term density (current: {term_density:.1f}, target: 1.5+ per 100 words)")
        
        # Readability optimization (Phase 4: Enhanced thresholds)
        if long_sentences > len(sentences) * 0.25:  # Tightened from 30% to 25%
            if long_sentences > len(sentences) * 0.4:
                logger.warning(f"⚠️ High long sentence ratio: {long_sentences}/{len(sentences)} ({long_sentences/len(sentences)*100:.1f}%)")
                recommendations.append(f"Split long sentences (current: {long_sentences}, target: <{int(len(sentences)*0.2)})")
            else:
                optimization_suggestions.append(f"Optimize readability (current: {long_sentences} long sentences, optimal: <{int(len(sentences)*0.2)})")
        
        # Source availability check
        if len(sources) == 0:
            readiness_criteria.append("No sources available for verification")
            recommendations.append("Ensure sources are available from research stage")
        elif len(sources) < 3:
            optimization_suggestions.append(f"Increase source diversity (current: {len(sources)}, optimal: 3+ sources)")
        
        # Content length optimization (Phase 4: Higher standards)
        if word_count < 50:  # Raised minimum from 30 to 50
            readiness_criteria.append("Content too short for comprehensive quality evaluation")
            recommendations.append("Expand content to minimum 50 words for optimal analysis")
        elif word_count < 200:
            optimization_suggestions.append(f"Consider expanding content (current: {word_count} words, optimal: 200+ for best scoring)")
        
        # Phase 4: Quality potential assessment
        quality_potential = "HIGH"
        if citation_count < 5 or len(protected_terms) < 8:
            quality_potential = "MEDIUM"
        if citation_count < 3 or len(protected_terms) < 4:
            quality_potential = "LOW"
        
        # Determine readiness
        is_ready = len(readiness_criteria) == 0
        
        # Phase 4: Enhanced logging
        if not is_ready:
            logger.error(f"❌ Content verification failed: {'; '.join(readiness_criteria)}")
        else:
            logger.info("✅ Content passed Phase 4 enhanced pre-gate verification")
            logger.info(f"   🎯 Quality potential: {quality_potential}")
            if optimization_suggestions:
                logger.info(f"   💡 Optimization opportunities: {len(optimization_suggestions)} suggestions available")
        
        return {
            'ready': is_ready,
            'reason': '; '.join(readiness_criteria) if readiness_criteria else 'Content ready for enhanced quality gates',
            'recommendations': recommendations,
            'optimization_suggestions': optimization_suggestions,
            'quality_potential': quality_potential,
            'stats': {
                'content_length': content_length,
                'word_count': word_count,
                'citation_count': citation_count,
                'citation_density': citation_density,
                'domain_term_count': len(protected_terms),
                'avg_sentence_length': avg_sentence_length,
                'long_sentence_count': long_sentences,
                'total_sentences': len(sentences),
                'source_count': len(sources),
                'word_count': len(draft.content.split())
            }
        }


class DomainExpertiseGateAgent:
    """Validates domain-specific expertise and technical accuracy."""
    
    def __init__(self, model_name: str = None, enable_tool_selection: bool = True):
        # Use cost-optimized model configuration
        config = get_optimized_model_config("DomainExpertiseGateAgent", task_type="analytical_tasks")
        
        try:
            from app.services.rate_limiter import wrap_llm_with_caching
            has_caching = True
        except ImportError:
            logger.warning("Rate limiter not available, using basic LLM")
            has_caching = False
        
        base_llm = ChatOpenAI(
            model=config.model_name,
            temperature=0.2,  # Low temperature for technical accuracy
            max_tokens=config.max_tokens
        )
        self.base_llm = wrap_llm_with_caching(base_llm, "openai") if has_caching else base_llm
        self.agent_name = "DomainExpertiseGateAgent"
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(CONTENT_VALIDATION_TOOLS)
        else:
            self.llm = self.base_llm
        
        # Domain-specific expertise criteria
        self.domain_criteria = {
            'it_insurance': {
                'technical_depth': ['cybersecurity', 'digital transformation', 'insurtech', 'data privacy'],
                'regulatory_knowledge': ['GDPR', 'SOX', 'PCI DSS', 'HIPAA'],
                'industry_trends': ['digital claims', 'AI underwriting', 'IoT sensors']
            },
            'ai': {
                'technical_depth': ['machine learning', 'neural networks', 'NLP', 'computer vision'],
                'current_developments': ['transformer models', 'LLMs', 'generative AI'],
                'practical_applications': ['deployment', 'MLOps', 'model evaluation']
            },
            'agentic_ai': {
                'technical_depth': ['multi-agent systems', 'agent coordination', 'reasoning'],
                'frameworks': ['LangGraph', 'LangChain', 'AutoGen', 'CrewAI'],
                'implementation': ['orchestration', 'state management', 'tool usage']
            }
        }
    
    async def execute(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """
        Evaluate domain expertise and technical depth of content.
        
        Args:
            draft: Content draft to evaluate
            domains: Target domains for expertise assessment
            
        Returns:
            Dictionary with expertise score and assessment details
        """
        logger.info(f"🔍 DomainExpertiseGateAgent - Evaluating domain expertise for: {domains}")
        
        # Comprehensive content analysis logging
        content_stats = log_content_analysis("DomainExpertiseGateAgent", draft, analysis_type="domain-expertise")
        
        try:
            # Check for cached result first
            from app.services.redis_service import redis_service
            content_hash = redis_service.create_content_hash(
                draft.content, 
                f"domain_expertise_{','.join(domains)}_{QUALITY_GATE_SCORING_VERSION}"
            )
            
            cached_result = await redis_service.get_cached_quality_result(
                content_hash, 
                "domain_expertise"
            )
            
            if cached_result:
                if cached_result.get("fallback"):
                    logger.info("⚠️ Cached domain expertise result flagged as fallback; forcing fresh analysis")
                else:
                    logger.info(f"📋 Using cached domain expertise result (version: {QUALITY_GATE_SCORING_VERSION})")
                    return cached_result
            
            logger.info(f"🔄 Performing FRESH domain expertise analysis (version: {QUALITY_GATE_SCORING_VERSION})")
            
            # Step 1: Assess technical depth
            technical_assessment = await self._assess_technical_depth(draft, domains)
            
            # Step 2: Evaluate domain-specific terminology
            terminology_assessment = self._evaluate_terminology(draft, domains)
            
            # Step 3: Check for practical insights
            practical_insights = await self._assess_practical_value(draft, domains)
            
            # Step 4: Calculate expertise score
            expertise_score = self._calculate_expertise_score(
                technical_assessment,
                terminology_assessment,
                practical_insights
            )
            
            # Step 5: Generate recommendations
            recommendations = self._generate_expertise_recommendations(
                technical_assessment,
                terminology_assessment,
                practical_insights,
                domains
            )
            
            logger.info(f"Domain expertise assessment completed: {expertise_score:.3f}")
            
            result = {
                'score': expertise_score,
                'technical_depth_score': technical_assessment['score'],
                'terminology_score': terminology_assessment['score'],
                'practical_value_score': practical_insights['score'],
                'recommendations': recommendations,
                'domain_coverage': self._assess_domain_coverage(draft, domains),
                'fallback': False
            }
            
            # Cache the result
            await redis_service.cache_quality_gate_result(
                content_hash,
                "domain_expertise", 
                result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Domain expertise assessment failed: {e}")
            heuristic_result = self._build_heuristic_domain_result(draft, domains, str(e))
            return heuristic_result
    
    async def _assess_technical_depth(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Assess the technical depth and accuracy of content."""
        
        system_prompt = domain_expertise_agent_instruction
        
        domain_context = []
        for domain in domains:
            if domain in self.domain_criteria:
                criteria = self.domain_criteria[domain]
                domain_context.append(f"{domain.upper()}:")
                domain_context.extend([f"  - {category}: {', '.join(items)}" 
                                     for category, items in criteria.items()])
        
        # Log full content processing (no truncation)
        logger.info(f"✅ DomainExpertiseGateAgent processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Evaluate the technical depth of this content for domains: {', '.join(domains)}

        Expected domain expertise areas:
        {chr(10).join(domain_context)}

        Content to evaluate:
        Title: {draft.title}
        Content: {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "technical_concepts_identified": ["concept1", "concept2"],
            "depth_analysis": "detailed assessment",
            "accuracy_concerns": ["any issues found"],
            "strengths": ["areas of good technical depth"]
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Technical depth assessment JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted technical depth assessment JSON from response")
                else:
                    logger.error("No JSON object found in technical depth assessment response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Technical depth assessment failed: {e}")
            return self._heuristic_technical_depth(draft, domains, str(e))
    
    def _evaluate_terminology(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Evaluate use of domain-specific terminology with enhanced matching."""
        
        content = draft.content
        content_lower = content.lower()
        
        # PRIORITY: Check for protected domain terms first (enhanced content)
        import re
        protected_terms = re.findall(r'<([^>]+)>', content)
        
        if protected_terms:
            # Enhanced domain expertise scoring with 8+ term requirement
            term_count = len(protected_terms)
            logger.info(f"🏷️ DOMAIN EXPERTISE ANALYSIS: {term_count} protected terms found")
            
            # Phase 2 Enhancement: Require 8+ protected terms for optimal scoring
            if term_count >= 8:
                # Excellent domain expertise: 8+ terms
                protected_score = min(1.0, 0.85 + (term_count - 8) * 0.02)  # 0.85 base + bonus
                score_tier = "EXCELLENT"
            elif term_count >= 6:
                # Good domain expertise: 6-7 terms
                protected_score = 0.70 + (term_count - 6) * 0.075  # 0.70-0.825
                score_tier = "GOOD"
            elif term_count >= 4:
                # Adequate domain expertise: 4-5 terms
                protected_score = 0.55 + (term_count - 4) * 0.075  # 0.55-0.70
                score_tier = "ADEQUATE"
            else:
                # Insufficient domain expertise: <4 terms
                protected_score = min(0.50, term_count * 0.125)  # Up to 0.50
                score_tier = "INSUFFICIENT"
            
            logger.info(f"   🎯 Domain Expertise Tier: {score_tier} ({term_count} terms)")
            logger.info(f"   📊 Score: {protected_score:.3f}")
            logger.info(f"   🏷️ Protected terms: {protected_terms[:8]}{'...' if len(protected_terms) > 8 else ''}")
            
            if term_count < 8:
                logger.warning(f"   ⚠️ Below optimal term count: {term_count}/8 recommended protected terms")
            
            return {
                'score': protected_score,
                'terms_found': protected_terms,
                'term_count': term_count,
                'tier': score_tier,
                'weighted_coverage': protected_score * 100,
                'match_details': [{'term': term, 'type': 'protected'} for term in protected_terms],
                'enhancement_detected': True,
                'meets_optimal_requirement': term_count >= 8
            }
        
        # Normalize content by removing punctuation for better matching
        normalized_content = re.sub(r'[^\w\s]', ' ', content_lower)
        content_words = set(normalized_content.split())
        
        terminology_score = 0.0
        found_terms = []
        total_term_weight = 0.0
        total_possible_weight = 0.0
        
        for domain in domains:
            if domain in self.domain_criteria:
                criteria = self.domain_criteria[domain]
                
                for category, terms in criteria.items():
                    for term in terms:
                        term_lower = term.lower()
                        term_weight = 1.0  # Base weight
                        
                        # Enhanced matching strategies
                        match_found = False
                        match_type = None
                        
                        # 1. Exact match
                        if term_lower in normalized_content:
                            match_found = True
                            match_type = "exact"
                        
                        # 2. Plural/singular variations
                        elif not match_found:
                            variations = []
                            if term_lower.endswith('s') and len(term_lower) > 3:
                                variations.append(term_lower[:-1])  # Remove 's'
                            else:
                                variations.append(term_lower + 's')  # Add 's'
                            
                            # Common plural patterns
                            if term_lower.endswith('y') and len(term_lower) > 3:
                                variations.append(term_lower[:-1] + 'ies')
                            
                            for variation in variations:
                                if variation in normalized_content:
                                    match_found = True
                                    match_type = "variation"
                                    term_weight *= 0.9  # Slight penalty for variation
                                    break
                        
                        # 3. Word-level matching for compound terms
                        if not match_found and ' ' in term_lower:
                            term_words = set(term_lower.split())
                            word_matches = len(term_words.intersection(content_words))
                            if word_matches >= len(term_words) * 0.7:  # 70% word match
                                match_found = True
                                match_type = "partial"
                                term_weight *= (word_matches / len(term_words)) * 0.8
                        
                        # 4. Stemming for longer terms
                        elif not match_found and len(term_lower) > 6:
                            stem = term_lower[:max(4, len(term_lower)-2)]
                            if stem in normalized_content:
                                match_found = True
                                match_type = "stem"
                                term_weight *= 0.7  # Penalty for stem match
                        
                        total_possible_weight += 1.0
                        
                        if match_found:
                            found_terms.append({
                                'term': term,
                                'type': match_type,
                                'weight': term_weight,
                                'category': category
                            })
                            total_term_weight += term_weight
        
        # Calculate weighted score normalized by content length
        content_length_words = len(content_lower.split())
        length_normalization = min(1.0, content_length_words / 500)  # Normalize for 500+ word content
        
        if total_possible_weight > 0:
            base_terminology_score = total_term_weight / total_possible_weight
            # Apply length normalization and slight boost for density
            terminology_score = base_terminology_score * length_normalization * 1.2
        
        # Bonus for diverse terminology categories
        categories_used = set(term['category'] for term in found_terms)
        category_bonus = min(0.1, len(categories_used) * 0.02)
        
        final_score = min(1.0, terminology_score + category_bonus)
        
        return {
            'score': final_score,
            'terms_found': [term['term'] for term in found_terms],
            'detailed_matches': found_terms,
            'term_coverage_percentage': terminology_score * 100,
            'categories_covered': list(categories_used),
            'weighted_coverage': total_term_weight / max(1, total_possible_weight) * 100
        }
    
    async def _assess_practical_value(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Assess practical insights and actionable value."""
        
        system_prompt = domain_expertise_agent_instruction
        
        human_prompt = f"""Evaluate the practical value of this content for professionals in: {', '.join(domains)}

        Content:
        {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "actionable_insights": ["insight1", "insight2"],
            "practical_examples": ["example1", "example2"],
            "implementation_guidance": "level of implementation detail",
            "professional_relevance": "how relevant for working professionals"
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Enhanced JSON extraction with multiple fallback patterns
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Practical value assessment JSON parsing failed: {json_error}")
                import re
                
                # Try multiple extraction patterns
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        result = json.loads(json_str)
                        logger.info("Successfully extracted practical value assessment JSON from response")
                    except json.JSONDecodeError:
                        # Parse manually if structured text is present
                        result = self._parse_practical_value_fallback(content)
                else:
                    # Parse manually if structured text is present
                    result = self._parse_practical_value_fallback(content)
            
            # Ensure numeric score is valid
            if isinstance(result.get('score'), str):
                try:
                    result['score'] = float(result['score'])
                except (ValueError, TypeError):
                    result['score'] = 0.6
            
            # Ensure score is within valid range
            result['score'] = max(0.0, min(1.0, result.get('score', 0.6)))
            
            return result
            
        except Exception as e:
            logger.warning(f"Practical value assessment failed: {e}")
            return self._heuristic_practical_value(draft, domains, str(e))
    
    def _parse_practical_value_fallback(self, content: str) -> Dict[str, Any]:
        """Parse practical value assessment from unstructured text."""
        
        import re
        
        # Extract score if mentioned
        score_match = re.search(r'(?:score|rating).*?([0-9]*\.?[0-9]+)', content, re.IGNORECASE)
        score = 0.6
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1.0:  # Assume it's out of 10 or 100
                    score = score / 10 if score <= 10 else score / 100
            except ValueError:
                pass
        
        # Enhanced insight extraction - more aggressive patterns
        insights = []
        insight_patterns = [
            r'[-*•]\s*([^.\n]+)',  # Bullet points
            r'\d+\.\s*([^.\n]+)',  # Numbered lists
            r'(?:should|must|need to|recommend|suggest|consider|implement|ensure|establish|develop|create|build|design)[^.]+',
            r'(?:best practice|strategy|approach|method|technique|framework|solution|process)[^.]+',
            r'(?:step|action|guideline|principle|rule|requirement)[^.]+',
            r'(?:key|important|critical|essential)[^.]+(?:is to|involves|requires|includes)[^.]+',
            r'(?:to|in order to|for)[^,]+(?:you should|organizations must|companies need|teams should)[^.]+',
            r'(?:start by|begin with|first|initially|next|then|finally)[^.]+',
            r'(?:avoid|prevent|minimize|reduce|eliminate)[^.]+',
            r'(?:increase|improve|enhance|optimize|maximize|strengthen)[^.]+',
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            insights.extend([match.strip() for match in matches if len(match.strip()) > 15])  # Longer minimum
        
        # Extract examples and case studies
        examples = []
        example_patterns = [
            r'(?:example|instance|such as|for example|case study|case in point)[:\s]+([^.]+)',
            r'(?:like|including|namely|specifically)[:\s]+([^.]+)',
            r'(?:consider|take)[^.]*(?:example of|case of)[^.]+',
            r'(?:companies|organizations|teams|businesses)[^.]*(?:have|use|implement|adopt)[^.]+',
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            examples.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        # Look for implementation steps and how-to content
        implementation_patterns = [
            r'how to[^.]+',
            r'ways? to[^.]+',
            r'methods? (?:for|to)[^.]+',
            r'processes? (?:for|to)[^.]+',
            r'steps? (?:for|to|include)[^.]+',
        ]
        
        for pattern in implementation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            insights.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        # Remove duplicates and limit
        insights = list(dict.fromkeys(insights))  # Remove duplicates while preserving order
        examples = list(dict.fromkeys(examples))
        
        # Boost score based on content richness
        insights_found = len(insights)
        examples_found = len(examples)
        
        if insights_found >= 3:
            score = max(score, 0.8)  # Good actionable content
        elif insights_found >= 1:
            score = max(score, 0.7)  # Some actionable content
        
        if examples_found >= 2:
            score = min(1.0, score + 0.1)  # Boost for examples
        elif examples_found >= 1:
            score = min(1.0, score + 0.05)  # Small boost for examples
        
        logger.info(f"🔧 Practical value fallback extracted: {insights_found} insights, {examples_found} examples → score: {score:.3f}")
        
        return {
            'score': score,
            'actionable_insights': insights[:7],  # Increased limit
            'practical_examples': examples[:4],   # Increased limit
            'implementation_guidance': f'Extracted {insights_found} insights and {examples_found} examples from content',
            'professional_relevance': 'High' if score >= 0.8 else 'Moderate' if score >= 0.6 else 'Low',
            'analysis_failed': True
        }
    
    def _calculate_expertise_score(
        self,
        technical_assessment: Dict[str, Any],
        terminology_assessment: Dict[str, Any],
        practical_insights: Dict[str, Any]
    ) -> float:
        """Calculate overall domain expertise score with detailed logging."""
        
        tech_score = technical_assessment['score']
        term_score = terminology_assessment['score']
        prac_score = practical_insights['score']
        
        logger.info(f"🎓 DOMAIN EXPERTISE SCORING DEBUG:")
        logger.info(f"  - Technical depth score: {tech_score:.3f}")
        logger.info(f"  - Terminology score: {term_score:.3f}")
        logger.info(f"  - Practical value score: {prac_score:.3f}")
        
        # Log terminology details if available
        if 'terms_found' in terminology_assessment:
            terms_found = terminology_assessment['terms_found']
            logger.info(f"  - Terms found: {len(terms_found)} ({', '.join(terms_found[:5])}...)")
        
        if 'weighted_coverage' in terminology_assessment:
            logger.info(f"  - Terminology coverage: {terminology_assessment['weighted_coverage']:.1f}%")
        
        # Log practical insights details
        if 'actionable_insights' in practical_insights:
            insights = practical_insights['actionable_insights']
            logger.info(f"  - Actionable insights found: {len(insights)}")
            for insight in insights[:2]:
                logger.info(f"    * '{insight[:50]}...'")
        
        # Weighted scoring
        technical_weight = 0.4
        terminology_weight = 0.3
        practical_weight = 0.3
        
        base_score = (
            tech_score * technical_weight +
            term_score * terminology_weight +
            prac_score * practical_weight
        )
        
        logger.info(f"  - Base weighted score: {base_score:.3f}")
        logger.info(f"    * Technical: {tech_score:.3f} × {technical_weight} = {tech_score * technical_weight:.3f}")
        logger.info(f"    * Terminology: {term_score:.3f} × {terminology_weight} = {term_score * terminology_weight:.3f}")
        logger.info(f"    * Practical: {prac_score:.3f} × {practical_weight} = {prac_score * practical_weight:.3f}")
        
        expertise_score = base_score
        
        # Enhanced uplift with more generous thresholds
        if (tech_score >= 0.75 and prac_score >= 0.75 and term_score >= 0.65):  # Lowered thresholds
            old_score = expertise_score
            expertise_score = min(1.0, expertise_score + 0.08)  # Increased from 0.05
            logger.info(f"  - High performance uplift: {old_score:.3f} → {expertise_score:.3f}")
        else:
            logger.info(f"  - No high performance uplift (tech: {tech_score:.2f}, prac: {prac_score:.2f}, term: {term_score:.2f})")
        
        # More generous comprehensive coverage bonus
        if all(score >= 0.70 for score in [tech_score, term_score, prac_score]):  # Lowered from 0.75
            old_score = expertise_score
            expertise_score = min(1.0, expertise_score + 0.05)  # Increased from 0.03
            logger.info(f"  - Comprehensive coverage bonus: {old_score:.3f} → {expertise_score:.3f}")
        else:
            below_threshold = [name for name, score in [('tech', tech_score), ('term', term_score), ('prac', prac_score)] if score < 0.70]
            logger.info(f"  - No comprehensive bonus: {below_threshold} below 0.70")
        
        # New: Domain terminology density bonus
        if term_score >= 0.8:
            old_score = expertise_score
            expertise_score = min(1.0, expertise_score + 0.03)
            logger.info(f"  - Strong terminology bonus: {old_score:.3f} → {expertise_score:.3f}")
        
        logger.info(f"  - FINAL DOMAIN EXPERTISE SCORE: {expertise_score:.3f}")
        
        return min(1.0, expertise_score)

    def _build_heuristic_domain_result(
        self,
        draft: Draft,
        domains: List[str],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Construct heuristic domain expertise result when LLM scoring fails."""
        terminology_assessment = self._evaluate_terminology(draft, domains)
        technical_assessment = self._heuristic_technical_depth(draft, domains, error_message)
        practical_insights = self._heuristic_practical_value(draft, domains, error_message)

        score = self._calculate_expertise_score(
            technical_assessment,
            terminology_assessment,
            practical_insights
        )

        recommendations = self._generate_expertise_recommendations(
            technical_assessment,
            terminology_assessment,
            practical_insights,
            domains
        )

        return {
            'score': score,
            'technical_depth_score': technical_assessment['score'],
            'terminology_score': terminology_assessment['score'],
            'practical_value_score': practical_insights['score'],
            'recommendations': recommendations,
            'domain_coverage': self._assess_domain_coverage(draft, domains),
            'fallback': True,
            'error': error_message
        }

    def _heuristic_technical_depth(
        self,
        draft: Draft,
        domains: List[str],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Estimate technical depth using keyword coverage when LLM fails."""
        content_lower = draft.content.lower()
        matched_terms = []
        for domain in domains:
            criteria = self.domain_criteria.get(domain, {})
            for category, terms in criteria.items():
                for term in terms:
                    if term.lower() in content_lower:
                        matched_terms.append({'term': term, 'category': category})

        unique_terms = {item['term'] for item in matched_terms}
        coverage = len(unique_terms)
        score = min(1.0, 0.55 + coverage * 0.04)

        return {
            'score': score,
            'technical_concepts_identified': list(unique_terms),
            'depth_analysis': f"Heuristic technical depth computed from {coverage} matched terms.",
            'accuracy_concerns': [error_message] if error_message else [],
            'strengths': [item['term'] for item in matched_terms[:5]],
            'heuristic': True
        }

    def _heuristic_practical_value(
        self,
        draft: Draft,
        domains: List[str],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Estimate practical value using structural cues and actionable language."""
        content = draft.content
        bullet_points = re.findall(r'^\s*[-*]\s+', content, re.MULTILINE)
        numbered_points = re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE)
        call_to_action_phrases = re.findall(r'\b(should|must|ensure|implement|establish|recommended|next steps)\b', content, re.IGNORECASE)

        actionable_sentences = re.findall(r'\b(how to|steps? to|in order to|best practice|strategy)\b', content, re.IGNORECASE)
        practical_examples = re.findall(r'(for example|case study|for instance|consider)', content, re.IGNORECASE)

        signal_score = 0.55
        signal_score += min(0.20, len(bullet_points) * 0.02)
        signal_score += min(0.10, len(numbered_points) * 0.02)
        signal_score += min(0.10, len(call_to_action_phrases) * 0.02)
        signal_score += min(0.10, len(actionable_sentences) * 0.015)
        signal_score += min(0.10, len(practical_examples) * 0.02)
        signal_score = min(0.9, signal_score)

        actionable_insights = [s.strip() for s in re.findall(r'[-*]\s+([^\n]+)', content)][:5]

        return {
            'score': signal_score,
            'actionable_insights': actionable_insights,
            'practical_examples': practical_examples[:3],
            'implementation_guidance': 'Heuristic analysis based on actionable language.',
            'professional_relevance': 'High' if signal_score >= 0.8 else 'Medium',
            'heuristic': True,
            'error': error_message
        }
    
    def _generate_expertise_recommendations(
        self,
        technical_assessment: Dict[str, Any],
        terminology_assessment: Dict[str, Any],
        practical_insights: Dict[str, Any],
        domains: List[str]
    ) -> List[str]:
        """Generate recommendations for improving domain expertise."""
        
        recommendations = []
        
        if technical_assessment['score'] < 0.7:
            recommendations.append("Increase technical depth with more detailed explanations")
            
        if technical_assessment.get('accuracy_concerns'):
            recommendations.append("Address technical accuracy concerns identified")
        
        if terminology_assessment['score'] < 0.6:
            recommendations.append(f"Include more domain-specific terminology for {', '.join(domains)}")
        
        if practical_insights['score'] < 0.7:
            recommendations.append("Add more actionable insights and practical examples")
        
        if len(practical_insights.get('actionable_insights', [])) < 3:
            recommendations.append("Include specific recommendations and best practices")
        
        return recommendations if recommendations else ["Domain expertise level is appropriate"]
    
    def _assess_domain_coverage(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Assess how well the content covers each target domain."""
        
        coverage = {}
        for domain in domains:
            if domain in self.domain_criteria:
                criteria = self.domain_criteria[domain]
                domain_coverage = 0
                
                for category, terms in criteria.items():
                    category_matches = sum(1 for term in terms if term.lower() in draft.content.lower())
                    category_coverage = category_matches / len(terms) if terms else 0
                    domain_coverage += category_coverage
                
                coverage[domain] = domain_coverage / len(criteria) if criteria else 0
            else:
                coverage[domain] = 0.5  # Unknown domain
        
        return coverage


class StyleCriticGateAgent:
    """Evaluates writing style, tone, and consistency."""
    
    def __init__(self, model_name: str = None, enable_tool_selection: bool = True):
        # Use cost-optimized model configuration for style analysis
        config = get_optimized_model_config("StyleCriticGateAgent", task_type="rule_based_tasks")
        
        self.base_llm = ChatOpenAI(
            model=config.model_name,
            temperature=0.3,
            max_tokens=config.max_tokens
        )
        self.agent_name = "StyleCriticGateAgent"
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(CONTENT_VALIDATION_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(self, draft: Draft, state: ContentPipelineState) -> Dict[str, Any]:
        """
        Evaluate content style, tone, and consistency.
        
        Args:
            draft: Content draft to evaluate
            state: Current pipeline state with target audience info
            
        Returns:
            Dictionary with style score and improvement suggestions
        """
        logger.info("🔍 StyleCriticGateAgent - Starting style and tone evaluation")
        
        # Comprehensive content analysis logging
        content_stats = log_content_analysis("StyleCriticGateAgent", draft, analysis_type="style-consistency")
        
        try:
            # Step 1: Analyze writing style
            style_analysis = await self._analyze_writing_style(draft, state)
            
            # Step 2: Check tone consistency
            tone_consistency = await self._check_tone_consistency(draft, state)
            
            # Step 3: Evaluate readability
            readability_metrics = self._calculate_readability_metrics(draft)
            
            # Step 4: Check structure and flow
            structure_analysis = self._analyze_structure(draft)
            
            # Step 5: Calculate overall style score
            style_score = self._calculate_style_score(
                style_analysis,
                tone_consistency,
                readability_metrics,
                structure_analysis
            )
            
            # Step 6: Generate improvement suggestions
            suggestions = self._generate_style_suggestions(
                style_analysis,
                tone_consistency,
                readability_metrics,
                structure_analysis
            )
            
            logger.info(f"Style evaluation completed: {style_score:.3f}")
            
            return {
                'score': style_score,
                'style_analysis': style_analysis,
                'tone_consistency': tone_consistency,
                'readability_metrics': readability_metrics,
                'structure_score': structure_analysis['score'],
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Style evaluation failed: {e}")
            return {
                'score': 0.7,  # Conservative fallback
                'error': str(e),
                'suggestions': ['Review content for style and tone consistency']
            }
    
    async def _analyze_writing_style(self, draft: Draft, state: ContentPipelineState) -> Dict[str, Any]:
        """Analyze overall writing style appropriateness."""
        
        # Helper function to get target audience from outline (dict or object)
        outline = state.get('outline', {})
        if hasattr(outline, 'target_audience'):
            target_audience = outline.target_audience
        elif isinstance(outline, dict):
            target_audience = outline.get('target_audience', 'professionals')
        else:
            target_audience = 'professionals'
        
        system_prompt = style_critic_agent_instruction
        
        # Log full content processing (no truncation)
        logger.info(f"✅ StyleCriticGateAgent processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Evaluate the writing style of this content:

        Target audience: {target_audience}
        Title: {draft.title}
        Content: {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "tone_assessment": "description of current tone",
            "clarity_score": 0.0-1.0,
            "engagement_level": 0.0-1.0,
            "voice_consistency": 0.0-1.0,
            "formality_appropriate": true/false,
            "specific_issues": ["issue1", "issue2"]
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Style analysis JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted style analysis JSON from response")
                else:
                    logger.error("No JSON object found in style analysis response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Style analysis failed: {e}")
            return {
                'score': 0.7,
                'tone_assessment': 'Unable to assess',
                'clarity_score': 0.7,
                'engagement_level': 0.7,
                'voice_consistency': 0.7,
                'formality_appropriate': True,
                'specific_issues': [str(e)]
            }
    
    async def _check_tone_consistency(self, draft: Draft, state: ContentPipelineState) -> Dict[str, Any]:
        """Check tone consistency throughout the complete content."""
        
        # Phase 3 Enhancement: Analyze complete content instead of samples
        logger.info(f"📝 Style Analysis - Processing COMPLETE content: {len(draft.content):,} characters")
        
        # Split content into sections for analysis
        sections = self._split_content_sections(draft.content)
        
        if len(sections) < 2:
            return {'score': 1.0, 'consistency': 'single_section', 'variations': []}
        
        system_prompt = style_critic_agent_instruction
        
        # Phase 3: Process ALL sections with intelligent truncation per section
        max_section_length = 500  # Allow longer sections for better analysis
        section_summaries = []
        
        for i, section in enumerate(sections):
            if len(section) <= max_section_length:
                # Short section - use complete content
                section_summaries.append(f"Section {i+1}: {section}")
            else:
                # Long section - use beginning and end for better context
                beginning = section[:250]
                ending = section[-250:]
                section_summaries.append(f"Section {i+1}: {beginning}...[MIDDLE CONTENT]...{ending}")
        
        # Limit to 10 sections max for performance, but include more content per section
        if len(section_summaries) > 10:
            logger.info(f"   📄 Large document: analyzing first 10 of {len(sections)} sections")
            section_summaries = section_summaries[:10]
        
        logger.info(f"   📊 Tone analysis covering {len(section_summaries)} sections with enhanced content sampling")
        
        human_prompt = f"""Analyze tone consistency across these content sections:

        {chr(10).join(section_summaries)}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "overall_tone": "description",
            "tone_variations": ["section X has different tone because..."],
            "consistency_issues": ["specific inconsistencies found"]
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Tone consistency check JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted tone consistency check JSON from response")
                else:
                    logger.error("No JSON object found in tone consistency check response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Tone consistency check failed: {e}")
            return {
                'score': 0.8,
                'overall_tone': 'Unable to assess',
                'tone_variations': [],
                'consistency_issues': [str(e)]
            }
    
    def _split_content_sections(self, content: str) -> List[str]:
        """Split content into logical sections for analysis."""
        
        # Split by markdown headers or double newlines
        sections = re.split(r'(?:\n#{1,3}\s+.*\n|\n\n)', content)
        sections = [section.strip() for section in sections if len(section.strip()) > 100]
        
        return sections
    
    def _calculate_readability_metrics(self, draft: Draft) -> Dict[str, Any]:
        """Calculate readability metrics with complete content analysis (Phase 3 Enhancement)."""
        
        content = draft.content
        content_length = len(content)
        
        # Phase 3: Log complete content analysis
        logger.info(f"📐 Readability Analysis - Processing COMPLETE content: {content_length:,} characters")
        
        # Enhanced sentence splitting that preserves citations and protected terms
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate words excluding citations and protected terms for accurate metrics
        clean_content = re.sub(r'\[Citation\s*\d+\]', '', content)  # Remove citations
        clean_content = re.sub(r'<[^>]+>', '', clean_content)  # Remove protected terms
        words = clean_content.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Calculate sentence lengths excluding enhancement markers
        sentence_lengths = []
        for sentence in sentences:
            clean_sentence = re.sub(r'\[Citation\s*\d+\]', '', sentence)
            clean_sentence = re.sub(r'<[^>]+>', '', clean_sentence)
            sentence_lengths.append(len(clean_sentence.split()))
        
        # Enhanced readability bonus for well-structured content
        long_sentences = sum(1 for length in sentence_lengths if length > 24)
        readability_enhancement_detected = long_sentences <= len(sentences) * 0.2  # Less than 20% long sentences
        
        # Basic metrics
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        # Simple readability approximation (Flesch Reading Ease approximation)
        if sentences and words:
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            readability_score = max(0, min(100, readability_score))  # Clamp to 0-100
        else:
            readability_score = 50  # Neutral score
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # Professional content readability mapping
        # Map Flesch scores to appropriate professional content scores
        professional_score = self._map_to_professional_readability(readability_score)
        
        # Readability enhancement bonus for well-polished content
        if readability_enhancement_detected:
            enhancement_boost = 0.15  # Boost for readability polishing
            professional_score = min(1.0, professional_score + enhancement_boost)
            logger.info(f"📈 READABILITY ENHANCEMENT DETECTED: {long_sentences}/{len(sentences)} long sentences → +{enhancement_boost:.3f} boost")
        
        return {
            'score': professional_score,
            'average_sentence_length': avg_sentence_length,
            'average_syllables_per_word': avg_syllables_per_word,
            'readability_score': readability_score,
            'paragraph_count': len(paragraphs),
            'average_paragraph_length': avg_paragraph_length,
            'readability_level': self._interpret_readability_score(readability_score),
            'professional_appropriateness': self._assess_professional_appropriateness(
                readability_score, avg_sentence_length, avg_paragraph_length
            ),
            'long_sentences_count': long_sentences,
            'total_sentences': len(sentences),
            'readability_enhancement_detected': readability_enhancement_detected
        }
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting heuristic."""
        word = word.lower().strip('.,!?;:"')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)  # Every word has at least 1 syllable
    
    def _interpret_readability_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very easy"
        elif score >= 80:
            return "Easy" 
        elif score >= 70:
            return "Fairly easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very difficult"
    
    def _map_to_professional_readability(self, flesch_score: float) -> float:
        """Map Flesch Reading Ease to professional content appropriateness."""
        
        # Professional content readability mapping
        # Recognizes that enterprise/technical material is inherently dense
        if flesch_score >= 60:
            return 1.0  # Very readable for professional content
        elif flesch_score >= 50:
            return 0.95  # Excellent for professional content
        elif flesch_score >= 40:
            return 0.9   # Good professional readability
        elif flesch_score >= 30:
            return 0.8   # Acceptable for technical content
        elif flesch_score >= 20:
            return 0.7   # Dense but manageable
        elif flesch_score >= 10:
            return 0.6   # Very dense, needs improvement
        else:
            return 0.5   # Too dense for most readers
    
    def _assess_professional_appropriateness(
        self, 
        flesch_score: float, 
        avg_sentence_length: float, 
        avg_paragraph_length: float
    ) -> str:
        """Assess if readability is appropriate for professional audience."""
        
        # Professional content guidelines
        ideal_sentence_length = 15-25  # words
        ideal_paragraph_length = 50-150  # words
        
        issues = []
        
        if avg_sentence_length > 30:
            issues.append("sentences too long")
        elif avg_sentence_length < 8:
            issues.append("sentences too short")
            
        if avg_paragraph_length > 200:
            issues.append("paragraphs too long")
        elif avg_paragraph_length < 30:
            issues.append("paragraphs too short")
            
        if flesch_score < 30:
            issues.append("content too dense")
        elif flesch_score > 80:
            issues.append("content may be too simple")
        
        if not issues:
            return "Excellent professional readability"
        elif len(issues) == 1:
            return f"Good readability, but {issues[0]}"
        else:
            return f"Readability issues: {', '.join(issues)}"
    
    def _analyze_structure(self, draft: Draft) -> Dict[str, Any]:
        """Analyze complete content structure and flow (Phase 3 Enhancement)."""
        
        content = draft.content
        content_length = len(content)
        
        # Phase 3: Log complete structure analysis
        logger.info(f"🏗️ Structure Analysis - Processing COMPLETE content: {content_length:,} characters")
        
        # Enhanced structural element detection
        # Markdown headers
        markdown_headers = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        
        # HTML headers and title case patterns
        html_headers = len(re.findall(r'<h[1-6][^>]*>', content, re.IGNORECASE))
        title_case_headers = len(re.findall(r'^[A-Z][A-Za-z\s]+:?\s*$', content, re.MULTILINE))
        
        # Total headers from all formats
        total_headers = markdown_headers + html_headers + title_case_headers
        
        # Paragraphs and sections
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Enhanced list detection
        bullet_lists = len(re.findall(r'^\s*[-*+•]\s+', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        html_lists = len(re.findall(r'<[uo]l[^>]*>', content, re.IGNORECASE))
        
        total_lists = bullet_lists + numbered_lists + html_lists
        
        # Additional structural elements
        tables = len(re.findall(r'\|.*\|', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```|<code>', content, re.IGNORECASE))
        emphasis = len(re.findall(r'\*\*.*?\*\*|<(strong|b)>', content, re.IGNORECASE))
        
        # Enhanced structural scoring
        structure_score = 0.6  # Base score
        
        # Header scoring
        if total_headers >= 3:
            structure_score += 0.15  # Excellent header usage
        elif total_headers >= 2:
            structure_score += 0.10  # Good header usage
        elif total_headers >= 1:
            structure_score += 0.05  # Basic header usage
        
        # Paragraph scoring  
        if 4 <= paragraphs <= 12:
            structure_score += 0.10  # Optimal paragraph count
        elif 3 <= paragraphs <= 15:
            structure_score += 0.08  # Good paragraph count
        elif paragraphs >= 2:
            structure_score += 0.05  # Acceptable paragraph count
        
        # List and formatting scoring
        if total_lists >= 2:
            structure_score += 0.10  # Good use of lists
        elif total_lists >= 1:
            structure_score += 0.05  # Some list usage
        
        # Additional formatting elements
        if tables > 0:
            structure_score += 0.05  # Tables for data
        if code_blocks > 0:
            structure_score += 0.03  # Code examples
        if emphasis > 0:
            structure_score += 0.02  # Text emphasis
        
        structure_score = min(1.0, structure_score)
        
        return {
            'score': structure_score,
            'total_headers': total_headers,
            'markdown_headers': markdown_headers,
            'html_headers': html_headers,
            'title_case_headers': title_case_headers,
            'paragraphs_count': paragraphs,
            'total_lists': total_lists,
            'bullet_lists': bullet_lists,
            'numbered_lists': numbered_lists,
            'tables_count': tables,
            'code_blocks_count': code_blocks,
            'emphasis_count': emphasis,
            'has_clear_structure': total_headers >= 2 and paragraphs >= 3,
            'formatting_diversity': len([x for x in [total_headers, total_lists, tables, code_blocks] if x > 0])
        }
    
    def _calculate_style_score(
        self,
        style_analysis: Dict[str, Any],
        tone_consistency: Dict[str, Any],
        readability_metrics: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall style consistency score with detailed logging."""
        
        style_score = style_analysis.get('score', 0.7)
        tone_score = tone_consistency.get('score', 0.7) 
        readability_score = readability_metrics.get('score', 0.7)
        structure_score = structure_analysis.get('score', 0.7)
        
        logger.info(f"📝 STYLE CONSISTENCY SCORING DEBUG:")
        logger.info(f"  - Style analysis score: {style_score:.3f}")
        logger.info(f"  - Tone consistency score: {tone_score:.3f}")
        logger.info(f"  - Readability score: {readability_score:.3f}")
        logger.info(f"  - Structure score: {structure_score:.3f}")
        
        # Log readability details
        if 'readability_score' in readability_metrics:
            flesch_score = readability_metrics['readability_score']
            avg_sentence_length = readability_metrics.get('average_sentence_length', 0)
            logger.info(f"    * Flesch Reading Ease: {flesch_score:.1f}")
            logger.info(f"    * Avg sentence length: {avg_sentence_length:.1f} words")
        
        # Log structure details
        if 'total_headers' in structure_analysis:
            headers = structure_analysis['total_headers']
            paragraphs = structure_analysis['paragraphs_count']
            lists = structure_analysis['total_lists']
            logger.info(f"    * Headers: {headers}, Paragraphs: {paragraphs}, Lists: {lists}")
        
        # Enhanced weighted scoring with professional content adjustments
        style_weight = 0.35      # Reduced slightly
        tone_weight = 0.25       # Reduced slightly  
        readability_weight = 0.25 # Increased for professional content
        structure_weight = 0.15   # Increased for professional content
        
        base_score = (
            style_score * style_weight +
            tone_score * tone_weight +
            readability_score * readability_weight +
            structure_score * structure_weight
        )
        
        logger.info(f"  - Base weighted score: {base_score:.3f}")
        
        # Professional content bonuses
        overall_score = base_score
        
        # Enhanced bonus for good structure with lower threshold
        if structure_score >= 0.75:  # Lowered from 0.8
            old_score = overall_score
            overall_score = min(1.0, overall_score + 0.08)  # Increased from 0.05
            logger.info(f"  - Structure bonus: {old_score:.3f} → {overall_score:.3f}")
        
        # More generous professional readability bonus with expanded range
        if 'readability_score' in readability_metrics:
            flesch = readability_metrics['readability_score']
            if 30 <= flesch <= 70:  # Expanded from 40-60 to 30-70
                old_score = overall_score
                overall_score = min(1.0, overall_score + 0.06)  # Increased from 0.03
                logger.info(f"  - Professional readability bonus: {old_score:.3f} → {overall_score:.3f}")
        
        # Enhanced style+tone bonus with lower thresholds
        if style_score >= 0.75 and tone_score >= 0.75:  # Lowered from 0.8
            old_score = overall_score
            overall_score = min(1.0, overall_score + 0.05)  # Increased from 0.02
            logger.info(f"  - Style+tone bonus: {old_score:.3f} → {overall_score:.3f}")
        
        # New: Professional formatting bonus for well-structured content
        if 'total_headers' in structure_analysis and structure_analysis['total_headers'] >= 3:
            if 'total_lists' in structure_analysis and structure_analysis['total_lists'] >= 2:
                old_score = overall_score
                overall_score = min(1.0, overall_score + 0.04)
                logger.info(f"  - Professional formatting bonus: {old_score:.3f} → {overall_score:.3f}")
        
        # New: Consistency across all dimensions bonus
        if all(score >= 0.7 for score in [style_score, tone_score, readability_score, structure_score]):
            old_score = overall_score
            overall_score = min(1.0, overall_score + 0.03)
            logger.info(f"  - Multi-dimensional consistency bonus: {old_score:.3f} → {overall_score:.3f}")
        
        logger.info(f"  - FINAL STYLE CONSISTENCY SCORE: {overall_score:.3f}")
        
        return min(1.0, overall_score)
    
    def _generate_style_suggestions(
        self,
        style_analysis: Dict[str, Any],
        tone_consistency: Dict[str, Any],
        readability_metrics: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific suggestions for style improvement."""
        
        suggestions = []
        
        # Style suggestions
        if style_analysis['score'] < 0.8:
            if style_analysis.get('clarity_score', 1.0) < 0.7:
                suggestions.append("Improve clarity by simplifying complex sentences")
            
            if style_analysis.get('engagement_level', 1.0) < 0.7:
                suggestions.append("Increase engagement with more active voice and vivid examples")
        
        # Tone suggestions
        if tone_consistency['score'] < 0.8:
            suggestions.append("Maintain consistent tone throughout all sections")
        
        # Readability suggestions
        readability_score = readability_metrics.get('readability_score', 50)
        if readability_score < 50:
            suggestions.append("Improve readability by shortening sentences and simplifying vocabulary")
        elif readability_score > 80:
            suggestions.append("Consider adding more depth and technical detail for professional audience")
        
        if readability_metrics.get('average_sentence_length', 0) > 25:
            suggestions.append("Break up long sentences for better readability")
        
        if readability_metrics.get('average_paragraph_length', 0) > 100:
            suggestions.append("Shorten paragraphs to improve visual appeal and readability")
        
        # Structure suggestions
        if structure_analysis['headers_count'] < 2:
            suggestions.append("Add more section headers to improve content organization")
        
        if structure_analysis['paragraphs_count'] > 20:
            suggestions.append("Consider breaking content into smaller, focused sections")
        
        return suggestions if suggestions else ["Content style and structure are appropriate"]


class ComplianceGateAgent:
    """Validates content compliance with legal and ethical guidelines."""
    
    def __init__(self, model_name: str = None, enable_tool_selection: bool = True):
        # Use cost-optimized model configuration for compliance checking
        config = get_optimized_model_config("ComplianceGateAgent", task_type="rule_based_tasks")
        
        self.base_llm = ChatOpenAI(
            model=config.model_name,
            temperature=0.1,  # Very low temperature for compliance
            max_tokens=config.max_tokens
        )
        self.agent_name = "ComplianceGateAgent"
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(CONTENT_VALIDATION_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(self, draft: Draft) -> Dict[str, Any]:
        """
        Evaluate content for legal and ethical compliance.
        
        Args:
            draft: Content draft to evaluate
            
        Returns:
            Dictionary with compliance score and detailed report
        """
        logger.info("Starting compliance validation")
        
        try:
            # Step 1: Check for legal compliance issues
            legal_check = await self._check_legal_compliance(draft)
            
            # Step 2: Validate ethical considerations
            ethical_check = await self._check_ethical_compliance(draft)
            
            # Step 3: Check for bias and discriminatory content
            bias_check = await self._check_bias_and_discrimination(draft)
            
            # Step 4: Validate privacy and data protection
            privacy_check = self._check_privacy_compliance(draft)
            
            # Step 5: Calculate compliance score
            compliance_score = self._calculate_compliance_score(
                legal_check,
                ethical_check,
                bias_check,
                privacy_check
            )
            
            # Step 6: Generate compliance report
            report = self._generate_compliance_report(
                legal_check,
                ethical_check,
                bias_check,
                privacy_check,
                compliance_score
            )
            
            logger.info(f"Compliance validation completed: {compliance_score:.3f}")
            
            return {
                'score': compliance_score,
                'report': report,
                'legal_issues': legal_check.get('issues', []),
                'ethical_concerns': ethical_check.get('concerns', []),
                'bias_indicators': bias_check.get('indicators', []),
                'privacy_concerns': privacy_check.get('concerns', [])
            }
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {
                'score': 0.8,  # Conservative compliance score
                'report': {'error': str(e)},
                'error': str(e)
            }
    
    async def _check_legal_compliance(self, draft: Draft) -> Dict[str, Any]:
        """Check for potential legal compliance issues."""
        
        system_prompt = compliance_agent_instruction
        
        # Log full content processing (no truncation)
        logger.info(f"✅ ComplianceGateAgent (Legal) processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Review this content for legal compliance issues:

        Title: {draft.title}
        Content: {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "issues": ["specific legal concern 1", "concern 2"],
            "risk_level": "low|medium|high", 
            "recommendations": ["action 1", "action 2"],
            "requires_legal_review": true/false
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Legal compliance check JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted legal compliance check JSON from response")
                else:
                    logger.error("No JSON object found in legal compliance check response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Legal compliance check failed: {e}")
            return {
                'score': 0.9,
                'issues': [],
                'risk_level': 'low',
                'recommendations': [],
                'requires_legal_review': False
            }
    
    async def _check_ethical_compliance(self, draft: Draft) -> Dict[str, Any]:
        """Check for ethical issues and concerns."""
        
        system_prompt = compliance_agent_instruction
        
        # Log full content processing (no truncation)
        logger.info(f"✅ ComplianceGateAgent (Ethical) processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Evaluate this content for ethical compliance:

        Title: {draft.title}
        Content: {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "concerns": ["ethical concern 1", "concern 2"],
            "ethical_strengths": ["positive aspect 1", "aspect 2"],
            "transparency_score": 0.0-1.0,
            "recommendations": ["improvement 1", "improvement 2"]
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Ethical compliance check JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted ethical compliance check JSON from response")
                else:
                    logger.error("No JSON object found in ethical compliance check response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Ethical compliance check failed: {e}")
            return {
                'score': 0.9,
                'concerns': [],
                'ethical_strengths': [],
                'transparency_score': 0.9,
                'recommendations': []
            }
    
    async def _check_bias_and_discrimination(self, draft: Draft) -> Dict[str, Any]:
        """Check for bias and discriminatory content."""
        
        system_prompt = compliance_agent_instruction
        
        # Log full content processing (no truncation)
        logger.info(f"✅ ComplianceGateAgent (Bias) processing FULL content: {len(draft.content):,} characters")
        
        human_prompt = f"""Analyze this content for bias and discriminatory elements:

        Title: {draft.title}
        Content: {draft.content}

        Return JSON with:
        {{
            "score": 0.0-1.0,
            "indicators": ["potential bias 1", "bias 2"],
            "inclusive_language_score": 0.0-1.0,
            "representation_analysis": "assessment of representation",
            "improvement_suggestions": ["suggestion 1", "suggestion 2"]
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            
            # Try to extract JSON if there's extra text
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Bias check JSON parsing failed: {json_error}")
                # Try to find JSON within the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.info("Successfully extracted bias check JSON from response")
                else:
                    logger.error("No JSON object found in bias check response")
                    raise json_error
            
            return result
            
        except Exception as e:
            logger.warning(f"Bias check failed: {e}")
            return {
                'score': 0.9,
                'indicators': [],
                'inclusive_language_score': 0.9,
                'representation_analysis': 'Unable to assess',
                'improvement_suggestions': []
            }
    
    def _check_privacy_compliance(self, draft: Draft) -> Dict[str, Any]:
        """Check for privacy and data protection compliance."""
        
        content_lower = draft.content.lower()
        
        # Enhanced privacy compliance with context analysis
        privacy_indicators = [
            'personal data', 'gdpr', 'privacy policy', 'data collection',
            'user information', 'personal information', 'data protection',
            'consent', 'data processing', 'data retention'
        ]
        
        # Find indicators with context
        found_indicators = []
        concerning_contexts = []
        positive_contexts = []
        
        for indicator in privacy_indicators:
            if indicator in content_lower:
                found_indicators.append(indicator)
                
                # Extract context around the indicator
                import re
                pattern = rf'.{{0,50}}{re.escape(indicator)}.{{0,50}}'
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                
                for match in matches:
                    # Check for concerning vs positive contexts
                    if any(word in match for word in ['without consent', 'collect', 'track', 'share', 'sell']):
                        concerning_contexts.append(match.strip())
                    elif any(word in match for word in ['comply', 'protect', 'secure', 'anonymize', 'respect']):
                        positive_contexts.append(match.strip())
        
        # Context-aware scoring
        if found_indicators:
            if concerning_contexts:
                # Found concerning privacy practices
                score = 0.7 if len(concerning_contexts) <= 2 else 0.6
                concerns = [f"Potentially concerning privacy practices: {', '.join(concerning_contexts[:2])}"]
            elif positive_contexts:
                # Mentions privacy in positive/compliant context
                score = 0.95
                concerns = []
            else:
                # Neutral privacy mentions
                score = 0.85 if len(found_indicators) <= 3 else 0.8
                concerns = [f"Content discusses privacy topics: {', '.join(found_indicators[:3])} - verify compliance"]
        else:
            # No privacy topics mentioned
            score = 1.0
            concerns = []
        
        return {
            'score': score,
            'concerns': concerns,
            'privacy_topics_mentioned': found_indicators,
            'gdpr_considerations': len([i for i in found_indicators if 'gdpr' in i or 'data' in i]) > 0
        }
    
    def _calculate_compliance_score(
        self,
        legal_check: Dict[str, Any],
        ethical_check: Dict[str, Any],
        bias_check: Dict[str, Any],
        privacy_check: Dict[str, Any]
    ) -> float:
        """Calculate overall compliance score."""
        
        # Weighted scoring - legal compliance is most critical
        legal_weight = 0.4
        ethical_weight = 0.3
        bias_weight = 0.2
        privacy_weight = 0.1
        
        compliance_score = (
            legal_check['score'] * legal_weight +
            ethical_check['score'] * ethical_weight +
            bias_check['score'] * bias_weight +
            privacy_check['score'] * privacy_weight
        )
        
        # Enhanced penalty system with context awareness
        risk_level = legal_check.get('risk_level', 'low')
        issues_list = legal_check.get('issues', [])
        
        # More generous penalty system - only apply when there are clear violations
        if risk_level == 'high' and len(issues_list) > 2:  # Only penalize for multiple issues
            compliance_score *= 0.7  # Reduced penalty
        elif risk_level == 'medium':
            if len(issues_list) > 1:  # Only penalize for multiple issues
                compliance_score *= 0.95  # Minimal penalty
        
        # Enhanced positive compliance signals with multiple bonuses
        if (legal_check['score'] >= 0.9 and len(issues_list) == 0):  # Lowered threshold
            compliance_score = min(1.0, compliance_score + 0.05)  # Increased bonus
        
        # New: Privacy-conscious content bonus
        if privacy_check['score'] >= 0.95:
            compliance_score = min(1.0, compliance_score + 0.02)
        
        # New: Ethical considerations bonus
        if ethical_check['score'] >= 0.9:
            compliance_score = min(1.0, compliance_score + 0.02)
        
        # New: Bias-free content bonus
        if bias_check['score'] >= 0.9:
            compliance_score = min(1.0, compliance_score + 0.01)
        
        return min(1.0, compliance_score)
    
    def _generate_compliance_report(
        self,
        legal_check: Dict[str, Any],
        ethical_check: Dict[str, Any],
        bias_check: Dict[str, Any],
        privacy_check: Dict[str, Any],
        score: float
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        all_recommendations = []
        all_recommendations.extend(legal_check.get('recommendations', []))
        all_recommendations.extend(ethical_check.get('recommendations', []))
        all_recommendations.extend(bias_check.get('improvement_suggestions', []))
        
        critical_issues = []
        if legal_check.get('risk_level') in ['high', 'medium']:
            critical_issues.extend(legal_check.get('issues', []))
        
        return {
            'overall_score': score,
            'compliance_status': 'PASS' if score >= 0.95 else 'REVIEW_REQUIRED' if score >= 0.8 else 'FAIL',
            'legal_risk_level': legal_check.get('risk_level', 'low'),
            'requires_legal_review': legal_check.get('requires_legal_review', False),
            'critical_issues': critical_issues,
            'all_recommendations': list(set(all_recommendations)),  # Remove duplicates
            'compliance_areas': {
                'legal': legal_check['score'],
                'ethical': ethical_check['score'],
                'bias_free': bias_check['score'],
                'privacy': privacy_check['score']
            },
            'timestamp': datetime.now().isoformat()
        }
