from .research import ResearchCoordinatorAgent
from .curation import ContentCuratorAgent
from .seo import SEOStrategistAgent
from .writing import ContentWriterAgent
from .quality_gates import (
    FactCheckGateAgent,
    DomainExpertiseGateAgent, 
    StyleCriticGateAgent,
    ComplianceGateAgent
)
from .editing import ContentEditorAgent
from .critique import CritiqueAgent
from .formatting import ContentFormatterAgent
from .human_review import HumanReviewGateAgent
from .publisher import PublisherAgent

# Enhanced publisher sub-agents for DigitalDossier integration
from .category_classifier_agent import CategoryClassifierAgent
from .title_generator_agent import TitleGeneratorAgent
from .cover_image_agent import CoverImageAgent

# Optional sub-agents (may not be available in all environments)
try:
    from .genre_generator_agent import GenreGeneratorAgent
except ImportError:
    GenreGeneratorAgent = None

__all__ = [
    "ResearchCoordinatorAgent",
    "ContentCuratorAgent",
    "SEOStrategistAgent", 
    "ContentWriterAgent",
    "FactCheckGateAgent",
    "DomainExpertiseGateAgent",
    "StyleCriticGateAgent", 
    "ComplianceGateAgent",
    "ContentEditorAgent",
    "CritiqueAgent",
    "ContentFormatterAgent",
    "HumanReviewGateAgent",
    "PublisherAgent",
    "CategoryClassifierAgent",
    "TitleGeneratorAgent", 
    "CoverImageAgent"
]

# Add optional agents to __all__ if available
if GenreGeneratorAgent is not None:
    __all__.append("GenreGeneratorAgent")