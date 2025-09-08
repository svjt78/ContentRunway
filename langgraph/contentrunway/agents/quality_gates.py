"""Quality Gate Agents - Parallel quality assessment agents for content validation."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import asyncio
import re
import json
from datetime import datetime

from ..state.pipeline_state import Draft, Source, ContentPipelineState
from ..tools.langgraph_tools import LANGGRAPH_VALIDATION_TOOLS, validate_content, analyze_content_quality

logger = logging.getLogger(__name__)


# Fact Check Gate Agent Instructions
fact_check_agent_role_and_goal = """
You are a Fact Check Gate Agent specializing in validating factual accuracy and verifying claims against credible sources.
Your primary goal is to extract verifiable factual claims from content, verify them against available sources, identify unsupported assertions, and generate comprehensive fact-check scores with detailed reports.
"""

fact_check_agent_hints = """
Fact-checking best practices:
- Focus on verifiable claims: statistics, historical facts, technical specifications, research findings, market data, regulatory statements
- Ignore opinions, predictions, and subjective statements that cannot be verified
- Verify claims against authoritative sources with high credibility scores
- Identify unsupported assertion patterns like 'studies show', 'research indicates', 'experts believe'
- Score claims as SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or INSUFFICIENT based on source evidence
- Apply confidence scores (0.0-1.0) for verification assessments
- Generate actionable recommendations for improving factual accuracy
- Prioritize high-credibility sources for claim verification
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
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        from app.services.rate_limiter import wrap_llm_with_caching
        
        base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Very low temperature for factual accuracy
            max_tokens=3000
        )
        self.base_llm = wrap_llm_with_caching(base_llm, "openai")
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_VALIDATION_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(self, draft: Draft, sources: List[Source]) -> Dict[str, Any]:
        """
        Perform comprehensive fact-checking of the content draft.
        
        Args:
            draft: Content draft to fact-check
            sources: Available sources for verification
            
        Returns:
            Dictionary with fact-check score and detailed report
        """
        logger.info("Starting fact-check validation")
        
        try:
            # Check for cached result first
            from app.services.redis_service import redis_service
            content_hash = redis_service.create_content_hash(draft.content, f"fact_check_{len(sources)}")
            
            cached_result = await redis_service.get_cached_quality_result(
                content_hash, 
                "fact_check"
            )
            
            if cached_result:
                logger.info("Using cached fact-check result")
                return cached_result
            
            # Step 1: Extract factual claims from content
            claims = await self._extract_factual_claims(draft)
            
            # Step 2: Verify claims against sources
            verification_results = await self._verify_claims(claims, sources)
            
            # Step 3: Check for unsupported assertions
            unsupported_claims = self._identify_unsupported_claims(draft, sources)
            
            # Step 4: Generate fact-check score
            fact_check_score = self._calculate_fact_check_score(
                verification_results, 
                unsupported_claims,
                len(claims)
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
                'unsupported_claims': len(unsupported_claims)
            }
            
            # Cache the result
            await redis_service.cache_quality_gate_result(
                content_hash,
                "fact_check", 
                result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return {
                'score': 0.5,  # Conservative fallback
                'report': {'error': str(e)},
                'claims_verified': 0,
                'unsupported_claims': 0
            }
    
    async def _extract_factual_claims(self, draft: Draft) -> List[Dict[str, Any]]:
        """Extract factual claims that need verification."""
        
        system_prompt = fact_check_agent_instruction
        
        human_prompt = f"""Extract verifiable factual claims from this content:

        Title: {draft.title}
        Content: {draft.content[:3000]}  # First 3000 chars

        Return JSON array of claims with format:
        [{{"claim": "specific factual statement", "type": "statistic|historical|technical|research|market|legal|company", "context": "surrounding context"}}]
        
        Limit to the 15 most important factual claims.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke_cached(messages, identifier="fact_check_claims")
            claims = json.loads(response.content)
            return claims[:15]  # Limit for performance
            
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []
    
    async def _verify_claims(
        self, 
        claims: List[Dict[str, Any]], 
        sources: List[Source]
    ) -> List[Dict[str, Any]]:
        """Verify each claim against available sources."""
        
        verification_results = []
        
        for claim in claims:
            # Find relevant sources for this claim
            relevant_sources = self._find_sources_for_claim(claim, sources)
            
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
        sources: List[Source]
    ) -> List[Source]:
        """Find sources most likely to contain information about the claim."""
        
        claim_text = claim['claim'].lower()
        claim_keywords = re.findall(r'\b[a-zA-Z]{3,}\b', claim_text)
        
        relevant_sources = []
        for source in sources:
            source_text = f"{source.title} {source.summary}".lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in claim_keywords if keyword in source_text)
            
            if matches >= 2 or any(keyword in source_text for keyword in claim_keywords[-3:]):
                relevant_sources.append(source)
        
        # Sort by credibility and relevance
        return sorted(relevant_sources, key=lambda s: s.credibility_score, reverse=True)[:5]
    
    async def _verify_single_claim(
        self, 
        claim: Dict[str, Any], 
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Verify a single claim against sources."""
        
        system_prompt = fact_check_agent_instruction
        
        source_summaries = [
            f"Source {i+1}: {source.title}\nSummary: {source.summary}"
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
            response = await self.llm.ainvoke_cached(messages, identifier="fact_check_verify")
            result = json.loads(response.content)
            
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
    
    def _identify_unsupported_claims(self, draft: Draft, sources: List[Source]) -> List[str]:
        """Identify potentially unsupported assertions in the content."""
        
        # Simple pattern matching for common unsupported claim patterns
        content = draft.content
        unsupported_patterns = [
            r'studies show',
            r'research indicates',
            r'experts believe',
            r'most companies',
            r'industry leaders',
            r'statistics reveal'
        ]
        
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
                    unsupported_claims.append(sentence)
        
        return unsupported_claims[:10]  # Limit results
    
    def _calculate_fact_check_score(
        self,
        verification_results: List[Dict[str, Any]],
        unsupported_claims: List[str],
        total_claims: int
    ) -> float:
        """Calculate overall fact-check score."""
        
        if not verification_results and not total_claims:
            return 0.85  # Neutral score for content without factual claims
        
        # Score based on verification results
        verification_score = 0.0
        if verification_results:
            status_weights = {
                'SUPPORTED': 1.0,
                'PARTIALLY_SUPPORTED': 0.7,
                'CONTRADICTED': 0.0,
                'INSUFFICIENT': 0.3,
                'no_sources': 0.2
            }
            
            total_weight = sum(
                status_weights.get(result['status'], 0.2) * result['confidence']
                for result in verification_results
            )
            verification_score = total_weight / len(verification_results)
        
        # Penalty for unsupported claims
        unsupported_penalty = min(0.3, len(unsupported_claims) * 0.05)
        
        # Final score calculation
        final_score = max(0.0, verification_score - unsupported_penalty)
        
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


class DomainExpertiseGateAgent:
    """Validates domain-specific expertise and technical accuracy."""
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        from app.services.rate_limiter import wrap_llm_with_caching
        
        base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Low temperature for technical accuracy
            max_tokens=3000
        )
        self.base_llm = wrap_llm_with_caching(base_llm, "openai")
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_VALIDATION_TOOLS)
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
        logger.info(f"Evaluating domain expertise for: {domains}")
        
        try:
            # Check for cached result first
            from app.services.redis_service import redis_service
            content_hash = redis_service.create_content_hash(draft.content, f"domain_expertise_{','.join(domains)}")
            
            cached_result = await redis_service.get_cached_quality_result(
                content_hash, 
                "domain_expertise"
            )
            
            if cached_result:
                logger.info("Using cached domain expertise result")
                return cached_result
            
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
                'domain_coverage': self._assess_domain_coverage(draft, domains)
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
            return {
                'score': 0.6,  # Conservative fallback
                'error': str(e)
            }
    
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
        
        human_prompt = f"""Evaluate the technical depth of this content for domains: {', '.join(domains)}

        Expected domain expertise areas:
        {chr(10).join(domain_context)}

        Content to evaluate:
        Title: {draft.title}
        Content: {draft.content[:2500]}

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
            response = await self.llm.ainvoke_cached(messages, identifier="domain_expertise_technical")
            return json.loads(response.content)
            
        except Exception as e:
            logger.warning(f"Technical depth assessment failed: {e}")
            return {
                'score': 0.6,
                'technical_concepts_identified': [],
                'depth_analysis': 'Assessment failed',
                'accuracy_concerns': [str(e)],
                'strengths': []
            }
    
    def _evaluate_terminology(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Evaluate use of domain-specific terminology."""
        
        content_lower = draft.content.lower()
        terminology_score = 0.0
        found_terms = []
        
        for domain in domains:
            if domain in self.domain_criteria:
                criteria = self.domain_criteria[domain]
                domain_terms = []
                
                # Collect all terms for this domain
                for category, terms in criteria.items():
                    domain_terms.extend([term.lower() for term in terms])
                
                # Count term usage
                terms_found = [term for term in domain_terms if term in content_lower]
                found_terms.extend(terms_found)
                
                # Score based on term coverage
                if domain_terms:
                    domain_coverage = len(terms_found) / len(domain_terms)
                    terminology_score += domain_coverage
        
        # Average across domains
        if domains:
            terminology_score /= len(domains)
        
        return {
            'score': min(1.0, terminology_score * 1.5),  # Boost score slightly
            'terms_found': found_terms,
            'term_coverage_percentage': terminology_score * 100
        }
    
    async def _assess_practical_value(self, draft: Draft, domains: List[str]) -> Dict[str, Any]:
        """Assess practical insights and actionable value."""
        
        system_prompt = domain_expertise_agent_instruction
        
        human_prompt = f"""Evaluate the practical value of this content for professionals in: {', '.join(domains)}

        Content:
        {draft.content[:2000]}

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
            response = await self.llm.ainvoke_cached(messages, identifier="domain_expertise_practical")
            return json.loads(response.content)
            
        except Exception as e:
            logger.warning(f"Practical value assessment failed: {e}")
            return {
                'score': 0.6,
                'actionable_insights': [],
                'practical_examples': [],
                'implementation_guidance': 'Assessment failed',
                'professional_relevance': 'Unable to assess'
            }
    
    def _calculate_expertise_score(
        self,
        technical_assessment: Dict[str, Any],
        terminology_assessment: Dict[str, Any],
        practical_insights: Dict[str, Any]
    ) -> float:
        """Calculate overall domain expertise score."""
        
        # Weighted scoring
        technical_weight = 0.4
        terminology_weight = 0.3
        practical_weight = 0.3
        
        expertise_score = (
            technical_assessment['score'] * technical_weight +
            terminology_assessment['score'] * terminology_weight +
            practical_insights['score'] * practical_weight
        )
        
        return min(1.0, expertise_score)
    
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
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        self.base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            max_tokens=3000
        )
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_VALIDATION_TOOLS)
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
        logger.info("Starting style and tone evaluation")
        
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
        
        target_audience = state.get('outline', {}).get('target_audience', 'professionals')
        
        system_prompt = style_critic_agent_instruction
        
        human_prompt = f"""Evaluate the writing style of this content:

        Target audience: {target_audience}
        Title: {draft.title}
        Content sample: {draft.content[:1500]}

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
            return json.loads(response.content)
            
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
        """Check tone consistency throughout the content."""
        
        # Split content into sections for analysis
        sections = self._split_content_sections(draft.content)
        
        if len(sections) < 2:
            return {'score': 1.0, 'consistency': 'single_section', 'variations': []}
        
        system_prompt = style_critic_agent_instruction
        
        section_summaries = [f"Section {i+1}: {section[:200]}..." for i, section in enumerate(sections[:5])]
        
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
            return json.loads(response.content)
            
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
        """Calculate basic readability metrics."""
        
        content = draft.content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = content.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
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
        
        return {
            'score': min(1.0, readability_score / 100.0),
            'average_sentence_length': avg_sentence_length,
            'average_syllables_per_word': avg_syllables_per_word,
            'readability_score': readability_score,
            'paragraph_count': len(paragraphs),
            'average_paragraph_length': avg_paragraph_length,
            'readability_level': self._interpret_readability_score(readability_score)
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
    
    def _analyze_structure(self, draft: Draft) -> Dict[str, Any]:
        """Analyze content structure and flow."""
        
        content = draft.content
        
        # Count structural elements
        headers = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Check for lists and bullets
        lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        
        # Structural score based on organization
        structure_score = 0.7  # Base score
        
        if headers >= 2:
            structure_score += 0.1  # Good use of headers
        if 3 <= paragraphs <= 15:
            structure_score += 0.1  # Reasonable paragraph count
        if lists > 0 or numbered_lists > 0:
            structure_score += 0.1  # Uses lists for clarity
        
        structure_score = min(1.0, structure_score)
        
        return {
            'score': structure_score,
            'headers_count': headers,
            'paragraphs_count': paragraphs,
            'lists_count': lists + numbered_lists,
            'has_clear_structure': headers >= 2 and paragraphs >= 3
        }
    
    def _calculate_style_score(
        self,
        style_analysis: Dict[str, Any],
        tone_consistency: Dict[str, Any],
        readability_metrics: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall style consistency score."""
        
        # Weighted scoring
        style_weight = 0.4
        tone_weight = 0.3
        readability_weight = 0.2
        structure_weight = 0.1
        
        overall_score = (
            style_analysis['score'] * style_weight +
            tone_consistency['score'] * tone_weight +
            readability_metrics['score'] * readability_weight +
            structure_analysis['score'] * structure_weight
        )
        
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
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        # Use OpenAI GPT-4 for sensitive compliance checking
        self.base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Very low temperature for compliance
            max_tokens=3000
        )
        
        # Hybrid approach: bind validation tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_VALIDATION_TOOLS)
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
        
        human_prompt = f"""Review this content for legal compliance issues:

        Title: {draft.title}
        Content: {draft.content[:2500]}

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
            return json.loads(response.content)
            
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
        
        human_prompt = f"""Evaluate this content for ethical compliance:

        Title: {draft.title}
        Content: {draft.content[:2500]}

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
            return json.loads(response.content)
            
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
        
        human_prompt = f"""Analyze this content for bias and discriminatory elements:

        Title: {draft.title}
        Content: {draft.content[:2500]}

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
            return json.loads(response.content)
            
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
        
        # Check for potential privacy concerns
        privacy_indicators = [
            'personal data', 'gdpr', 'privacy policy', 'data collection',
            'user information', 'personal information', 'data protection',
            'consent', 'data processing', 'data retention'
        ]
        
        found_indicators = [indicator for indicator in privacy_indicators 
                          if indicator in content_lower]
        
        # Simple privacy compliance scoring
        if found_indicators:
            # Content mentions privacy topics - needs careful review
            score = 0.8 if len(found_indicators) <= 3 else 0.7
            concerns = [f"Content discusses {', '.join(found_indicators[:3])} - ensure GDPR compliance"]
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
        
        # Apply penalties for high-risk issues
        if legal_check.get('risk_level') == 'high':
            compliance_score *= 0.5
        elif legal_check.get('risk_level') == 'medium':
            compliance_score *= 0.8
        
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