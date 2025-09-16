"""PDF Generation Tool - Convert content to formatted PDF documents."""

import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import markdown
from bs4 import BeautifulSoup
import re

from ..utils.publisher_logger import PublisherLogger


class PDFGeneratorTool:
    """Tool for generating formatted PDF documents from content."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for PDF generation."""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='Author',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7f8c8d'),
            fontName='Helvetica-Oblique'
        ))
        
        # Heading styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=10,
            spaceBefore=16,
            textColor=colors.HexColor('#34495e')
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=16
        ))
        
        # Code style - check if already exists
        if 'Code' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Code',
                parent=self.styles['Normal'],
                fontSize=10,
                fontName='Courier',
                backgroundColor=colors.HexColor('#f8f9fa'),
                borderColor=colors.HexColor('#e9ecef'),
                borderWidth=1,
                borderPadding=8,
                spaceAfter=12
            ))
    
    async def generate_pdf_from_content(
        self,
        title: str,
        content: str,
        summary: Optional[str] = None,
        content_format: str = 'markdown'
    ) -> Dict[str, Any]:
        """Generate PDF from content with title page."""
        
        operation_context = {
            "title": title,
            "content_length": len(content),
            "content_format": content_format,
            "has_summary": bool(summary)
        }
        
        self.logger.log_operation_start("generate_pdf", operation_context)
        
        try:
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content story
            story = []
            
            # Add title page
            story.extend(self._create_title_page(title, summary))
            
            # Add page break
            story.append(PageBreak())
            
            # Convert and add main content
            if content_format.lower() == 'markdown':
                content_elements = self._convert_markdown_to_pdf(content)
            else:
                content_elements = self._convert_html_to_pdf(content)
            
            story.extend(content_elements)
            
            # Build PDF
            doc.build(story)
            
            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            # Create result
            result = {
                "pdf_data": pdf_data,
                "pdf_base64": base64.b64encode(pdf_data).decode('utf-8'),
                "size_bytes": len(pdf_data),
                "filename": f"{self._sanitize_filename(title)}.pdf",
                "mime_type": "application/pdf",
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.log_operation_success(
                "generate_pdf",
                {"size_bytes": len(pdf_data), "filename": result["filename"]},
                operation_context
            )
            
            return result
            
        except Exception as e:
            error_msg = f"PDF generation failed: {e}"
            self.logger.log_operation_failure("generate_pdf", error_msg, operation_context)
            raise Exception(error_msg)
    
    def _create_title_page(self, title: str, summary: Optional[str] = None) -> list:
        """Create title page elements."""
        
        story = []
        
        # Add some top spacing
        story.append(Spacer(1, 2*inch))
        
        # Title
        title_para = Paragraph(title, self.styles['CustomTitle'])
        story.append(title_para)
        
        # Author
        author_para = Paragraph("by Suvojit Dutta", self.styles['Author'])
        story.append(author_para)
        
        # Add spacing
        story.append(Spacer(1, 0.5*inch))
        
        # Summary if provided
        if summary:
            summary_title = Paragraph("Summary", self.styles['CustomHeading2'])
            story.append(summary_title)
            
            summary_para = Paragraph(summary, self.styles['CustomBody'])
            story.append(summary_para)
        
        # Add generation date
        story.append(Spacer(1, 1*inch))
        date_para = Paragraph(
            f"Written on {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Author']
        )
        story.append(date_para)
        
        return story
    
    def _convert_markdown_to_pdf(self, markdown_content: str) -> list:
        """Convert markdown content to PDF elements."""
        
        try:
            # Convert markdown to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['codehilite', 'fenced_code', 'tables']
            )
            
            return self._convert_html_to_pdf(html_content)
            
        except Exception as e:
            self.logger.log_warning("markdown_conversion", f"Markdown conversion failed, using raw text: {e}")
            # Fallback to plain text
            return [Paragraph(markdown_content, self.styles['CustomBody'])]
    
    def _convert_html_to_pdf(self, html_content: str) -> list:
        """Convert HTML content to PDF elements."""
        
        story = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'pre', 'code', 'blockquote']):
                
                if element.name in ['h1', 'h2']:
                    style = self.styles['CustomHeading1'] if element.name == 'h1' else self.styles['CustomHeading2']
                    para = Paragraph(element.get_text().strip(), style)
                    story.append(para)
                
                elif element.name in ['h3', 'h4', 'h5', 'h6']:
                    para = Paragraph(element.get_text().strip(), self.styles['CustomHeading2'])
                    story.append(para)
                
                elif element.name == 'p':
                    text = element.get_text().strip()
                    if text:
                        para = Paragraph(text, self.styles['CustomBody'])
                        story.append(para)
                
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li'):
                        bullet = "• " if element.name == 'ul' else "1. "
                        text = bullet + li.get_text().strip()
                        para = Paragraph(text, self.styles['CustomBody'])
                        story.append(para)
                
                elif element.name in ['pre', 'code']:
                    code_text = element.get_text().strip()
                    para = Paragraph(code_text, self.styles['Code'])
                    story.append(para)
                
                elif element.name == 'blockquote':
                    quote_text = element.get_text().strip()
                    # Create indented style for quotes
                    quote_style = ParagraphStyle(
                        name='Quote',
                        parent=self.styles['CustomBody'],
                        leftIndent=36,
                        rightIndent=36,
                        fontName='Helvetica-Oblique',
                        textColor=colors.HexColor('#555555')
                    )
                    para = Paragraph(quote_text, quote_style)
                    story.append(para)
        
        except Exception as e:
            self.logger.log_warning("html_conversion", f"HTML conversion failed, using raw text: {e}")
            # Fallback to plain text
            clean_text = BeautifulSoup(html_content, 'html.parser').get_text()
            para = Paragraph(clean_text, self.styles['CustomBody'])
            story.append(para)
        
        return story
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for filename use."""
        
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^\w\s-]', '', title)
        sanitized = re.sub(r'[-\s]+', '-', sanitized)
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized.strip('-') or 'document'
    
    async def generate_pdf_from_dict(self, content_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate PDF from content dictionary (from pipeline state)."""
        
        title = content_dict.get('title', 'Untitled Document')
        content = content_dict.get('content', content_dict.get('body', ''))
        summary = content_dict.get('summary', content_dict.get('excerpt'))
        
        return await self.generate_pdf_from_content(title, content, summary)
    
    def create_api_pdf_object(self, pdf_data: bytes, filename: Optional[str] = None) -> Dict[str, str]:
        """Create PDF object for DigitalDossier API."""
        
        return {
            "data": base64.b64encode(pdf_data).decode('utf-8'),
            "filename": filename or "document.pdf",
            "mimeType": "application/pdf"
        }