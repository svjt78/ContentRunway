import logging

logger = logging.getLogger(__name__)

# Base tools that should always work
available_tools = []

# Core tools - basic functionality
try:
    from .web_research import WebResearchTool
    available_tools.append("WebResearchTool")
except ImportError as e:
    logger.warning(f"WebResearchTool not available: {e}")
    class WebResearchTool:
        def __init__(self):
            self.available = False

try:
    from .searxng_tool import SearXNGTool
    available_tools.append("SearXNGTool")
except ImportError as e:
    logger.warning(f"SearXNGTool not available: {e}")
    class SearXNGTool:
        def __init__(self):
            self.available = False

try:
    from .content_analysis import ContentAnalysisTool
    available_tools.append("ContentAnalysisTool")
except ImportError as e:
    logger.warning(f"ContentAnalysisTool not available: {e}")
    class ContentAnalysisTool:
        def __init__(self):
            self.available = False

try:
    from .seo_analysis import SEOAnalysisTool
    available_tools.append("SEOAnalysisTool")
except ImportError as e:
    logger.warning(f"SEOAnalysisTool not available: {e}")
    class SEOAnalysisTool:
        def __init__(self):
            self.available = False

try:
    from .plagiarism_detection import PlagiarismDetectionTool
    available_tools.append("PlagiarismDetectionTool")
except ImportError as e:
    logger.warning(f"PlagiarismDetectionTool not available: {e}")
    class PlagiarismDetectionTool:
        def __init__(self):
            self.available = False

try:
    from .pii_scanner import PIIScannerTool
    available_tools.append("PIIScannerTool")
except ImportError as e:
    logger.warning(f"PIIScannerTool not available: {e}")
    class PIIScannerTool:
        def __init__(self):
            self.available = False

try:
    from .technical_validation import TechnicalValidationTool
    available_tools.append("TechnicalValidationTool")
except ImportError as e:
    logger.warning(f"TechnicalValidationTool not available: {e}")
    class TechnicalValidationTool:
        def __init__(self):
            self.available = False

try:
    from .embeddings_tool import EmbeddingsTool
    available_tools.append("EmbeddingsTool")
except ImportError as e:
    logger.warning(f"EmbeddingsTool not available: {e}")
    class EmbeddingsTool:
        def __init__(self):
            self.available = False

try:
    from .fact_checking import FactCheckTool
    available_tools.append("FactCheckTool")
except ImportError as e:
    logger.warning(f"FactCheckTool not available: {e}")
    class FactCheckTool:
        def __init__(self):
            self.available = False

try:
    from .compliance_tool import ComplianceTool
    available_tools.append("ComplianceTool")
except ImportError as e:
    logger.warning(f"ComplianceTool not available: {e}")
    class ComplianceTool:
        def __init__(self):
            self.available = False

try:
    from .publishing_api_tool import PublishingAPITool
    available_tools.append("PublishingAPITool")
except ImportError as e:
    logger.warning(f"PublishingAPITool not available: {e}")
    class PublishingAPITool:
        def __init__(self):
            self.available = False

# Publisher-specific tools with graceful fallbacks
try:
    from .digitaldossier_api_tool import DigitalDossierAPITool
    available_tools.append("DigitalDossierAPITool")
except ImportError as e:
    logger.warning(f"DigitalDossierAPITool not available: {e}")
    class DigitalDossierAPITool:
        def __init__(self):
            self.available = False

try:
    from .pdf_generator_tool import PDFGeneratorTool
    available_tools.append("PDFGeneratorTool")
except ImportError as e:
    logger.warning(f"PDFGeneratorTool not available: {e}")
    class PDFGeneratorTool:
        def __init__(self):
            self.available = False

try:
    from .content_classification_tool import ContentClassificationTool
    available_tools.append("ContentClassificationTool")
except ImportError as e:
    logger.warning(f"ContentClassificationTool not available: {e}")
    class ContentClassificationTool:
        def __init__(self):
            self.available = False

try:
    from .genre_mapping_tool import GenreMappingTool
    available_tools.append("GenreMappingTool")
except ImportError as e:
    logger.warning(f"GenreMappingTool not available: {e}")
    class GenreMappingTool:
        def __init__(self):
            self.available = False

try:
    from .cover_image_processor_tool import CoverImageProcessorTool
    available_tools.append("CoverImageProcessorTool")
except ImportError as e:
    logger.warning(f"CoverImageProcessorTool not available: {e}")
    class CoverImageProcessorTool:
        def __init__(self):
            self.available = False

try:
    from .dalle_image_generator_tool import DalleImageGeneratorTool
    available_tools.append("DalleImageGeneratorTool")
except ImportError as e:
    logger.warning(f"DalleImageGeneratorTool not available: {e}")
    class DalleImageGeneratorTool:
        def __init__(self):
            self.available = False

# Log successful imports
logger.info(f"Successfully loaded {len(available_tools)} tools: {', '.join(available_tools)}")

__all__ = [
    "WebResearchTool",
    "SearXNGTool", 
    "ContentAnalysisTool",
    "SEOAnalysisTool",
    "PlagiarismDetectionTool",
    "PIIScannerTool",
    "TechnicalValidationTool",
    "EmbeddingsTool",
    "FactCheckTool",
    "ComplianceTool",
    "PublishingAPITool",
    "DigitalDossierAPITool",
    "PDFGeneratorTool",
    "ContentClassificationTool",
    "GenreMappingTool",
    "CoverImageProcessorTool",
    "DalleImageGeneratorTool",
    "available_tools"
]