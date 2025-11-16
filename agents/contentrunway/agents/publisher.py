"""Publisher Agent - Complete DigitalDossier integration for ContentRunway."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .category_classifier_agent import CategoryClassifierAgent
from .title_generator_agent import TitleGeneratorAgent
from .cover_image_agent import CoverImageAgent
from ..tools.genre_mapping_tool import GenreMappingTool
from ..tools.digitaldossier_api_tool import DigitalDossierAPITool
from ..tools.pdf_generator_tool import PDFGeneratorTool
from ..utils.publisher_logger import PublisherLogger


class PublisherAgent:
    """
    Main Publisher Agent for DigitalDossier integration.
    
    Orchestrates the complete publishing workflow:
    1. Content classification (Blog vs Product)
    2. Title generation and optimization
    3. Cover image selection and processing
    4. PDF generation
    5. Genre generation using LLM
    6. DigitalDossier API upload
    7. Pipeline state updates
    """
    
    def __init__(self):
        self.logger = PublisherLogger()
        
        # Initialize sub-agents
        self.category_classifier = CategoryClassifierAgent()
        self.title_generator = TitleGeneratorAgent()
        self.cover_image_agent = CoverImageAgent()
        
        # Initialize tools (graceful fallback if environment not configured)
        try:
            self.api_tool = DigitalDossierAPITool()
            self.genre_mapping_tool = GenreMappingTool(self.api_tool)
            self.publishing_enabled = True
        except ValueError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DigitalDossier publishing disabled: {e}")
            self.api_tool = None
            self.genre_mapping_tool = None
            self.publishing_enabled = False
        
        try:
            self.pdf_generator = PDFGeneratorTool()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"PDF generation disabled: {e}")
            self.pdf_generator = None
    
    async def execute(
        self,
        channel_drafts: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute complete DigitalDossier publishing workflow.
        
        Args:
            channel_drafts: Platform-specific formatted content from formatting step
            state: Current pipeline state with complete context
            
        Returns:
            Publishing results with DigitalDossier URLs and state updates
        """
        
        start_time = datetime.now()
        
        operation_context = {
            "pipeline_run_id": state.get('run_id'),
            "has_channel_drafts": bool(channel_drafts),
            "has_draft": bool(state.get('draft')),
            "current_step": "publishing"
        }
        
        self.logger.log_operation_start("digitaldossier_publishing", operation_context)
        
        try:
            # DEBUG: Log execution context
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Publisher execution context - pipeline_run_id: {state.get('run_id')}")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Channel drafts type: {type(channel_drafts)}, State type: {type(state)}")
            
            # Step 1: Extract content from state
            print("🔍 Step 1: Extracting content from state...")
            self.logger.log_info("digitaldossier_publishing", "🔧 DEBUG: Starting content extraction")
            content = await self._extract_content_from_state(channel_drafts, state)
            print(f"   ✅ Content extracted: {list(content.keys())}")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Content extraction completed, keys: {list(content.keys())}")
            
            # Step 2: Test API connection
            print("🔍 Step 2: Testing API connection...")
            await self._verify_api_connection()
            print("   ✅ API connection verified")
            
            # Step 3: Execute sub-agents in parallel where possible
            print("🔍 Step 3: Executing content analysis...")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: About to execute content analysis with content length: {len(content.get('content', ''))}")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Content title: '{content.get('title', 'NO_TITLE')}'")
            try:
                classification_result, title_result = await self._execute_content_analysis(content, state)
                print(f"   ✅ Classification: {classification_result.get('classification')}")
                print(f"   ✅ Title: {title_result.get('recommended_title')}")
                self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Content analysis completed successfully")
                self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Classification confidence: {classification_result.get('confidence_score')}")
            except Exception as e:
                print(f"   ❌ Content analysis failed: {e}")
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: Content analysis failed: {e}")
                import traceback
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: Content analysis traceback: {traceback.format_exc()}")
                raise
            
            # Step 4: Generate/Select cover image (attempt real, fallback to placeholder)
            print("🔍 Step 4: Selecting cover image...")
            try:
                image_result = await self.cover_image_agent.execute(
                    content,
                    classification_result['classification'],
                    classification_result['analysis'],
                    title_result['recommended_title'],
                    state
                )
                # Ensure API object exists for upload
                if not image_result.get('api_object'):
                    image_result['api_object'] = {
                        'data': image_result.get('image_base64'),
                        'filename': image_result.get('filename', 'cover.png'),
                        'mimeType': image_result.get('mime_type', 'image/png')
                    }
                print(f"   ✅ Cover image: {image_result.get('filename')} ({'placeholder' if image_result.get('is_placeholder') else 'real'})")
            except Exception as _cover_err:
                # Fallback placeholder (minimal 1x1 transparent PNG)
                placeholder_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFFeaR9cAAAAABJRU5ErkJggg=="
                image_result = {
                    'filename': 'placeholder-cover.png',
                    'is_placeholder': True,
                    'api_object': {
                        'data': placeholder_png_base64,
                        'filename': 'placeholder-cover.png',
                        'mimeType': 'image/png'
                    }
                }
                print(f"   ⏭️ Cover image: {image_result.get('filename')} (placeholder) due to error: {_cover_err}")
            
            # Step 5: Generate PDF
            print("🔍 Step 5: Generating PDF...")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: About to generate PDF with title: '{title_result['recommended_title']}'")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: PDF content length: {len(content.get('content', ''))}")
            try:
                pdf_result = await self._generate_pdf(content, title_result['recommended_title'])
                print(f"   ✅ PDF: {pdf_result.get('filename')}")
                self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: PDF generated successfully - Size: {pdf_result.get('size_bytes')} bytes")
                
                # CRITICAL: Validate PDF size
                if pdf_result.get('size_bytes', 0) < 1000:
                    error_msg = f"PDF generation failed: File too small ({pdf_result.get('size_bytes')} bytes)"
                    self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: {error_msg}")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                print(f"   ❌ PDF generation failed: {e}")
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: PDF generation failed: {e}")
                import traceback
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: PDF generation traceback: {traceback.format_exc()}")
                raise
            
            # Step 6: Map content to genre using enhanced LLM mapping
            print("🔍 Step 6: Mapping content to genre...")
            genre_id, genre_analysis = await self.genre_mapping_tool.map_content_to_genre(
                title=title_result['recommended_title'],
                classification=classification_result['classification'],
                classification_analysis=classification_result['analysis'],
                content_summary=content.get('summary', '')
            )
            print(f"   ✅ Genre: {genre_analysis.get('selected_genre_name')} (ID: {genre_id})")
            
            # Convert to expected format for backward compatibility
            genre_result = {
                'genre_id': genre_id,
                'selected_genre': genre_analysis.get('selected_genre_name', genre_analysis.get('selected_genre', 'Unknown')),
                **genre_analysis  # Include all analysis data
            }
            
            # Step 7: Upload to DigitalDossier with genre metadata
            print("🔍 Step 7: Uploading to DigitalDossier...")
            print(f"   📝 Title: {title_result['recommended_title']}")
            print(f"   🏷️ Genre: {genre_result.get('selected_genre_name')} (ID: {genre_result['genre_id']})")
            print(f"   🆕 New Genre Required: {genre_result.get('requires_new_genre', False)}")
            print(f"   🎨 Cover Image: {image_result.get('filename')} ({'placeholder' if image_result.get('is_placeholder') else 'real'})")
            
            # CRITICAL: Validate PDF data before creating file object
            pdf_data = pdf_result.get('pdf_data')
            pdf_filename = pdf_result.get('filename', 'document.pdf')
            
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: PDF data validation - Type: {type(pdf_data)}")
            if isinstance(pdf_data, str):
                self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: PDF data is string, length: {len(pdf_data)}")
                # If it's a string, it might be base64 encoded
                try:
                    import base64
                    pdf_data = base64.b64decode(pdf_data)
                    self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Successfully decoded base64 PDF data: {len(pdf_data)} bytes")
                except Exception as e:
                    self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: Failed to decode base64 PDF data: {e}")
                    raise ValueError(f"Invalid PDF data format: {e}")
            elif isinstance(pdf_data, bytes):
                self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: PDF data is bytes, length: {len(pdf_data)}")
            else:
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: Unexpected PDF data type: {type(pdf_data)}")
                raise ValueError(f"Invalid PDF data type: {type(pdf_data)}")
            
            # CRITICAL: Ensure we have valid PDF bytes
            if not pdf_data or len(pdf_data) < 1000:
                error_msg = f"PDF data validation failed: {len(pdf_data) if pdf_data else 0} bytes (minimum 1000 required)"
                self.logger.log_error("digitaldossier_publishing", f"🔧 DEBUG: {error_msg}")
                raise ValueError(error_msg)
            
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Creating PDF file object with {len(pdf_data)} bytes")
            
            upload_result = await self._upload_to_digitaldossier(
                title=title_result['recommended_title'],
                category=classification_result['classification'],
                genre_id=genre_result['genre_id'],
                genre_metadata=genre_result if genre_result.get('requires_new_genre') else None,
                cover_image=image_result['api_object'],
                pdf_file=self.api_tool.create_pdf_file_object(
                    pdf_data,
                    pdf_filename
                )
            )
            print(f"   ✅ Upload result: {upload_result.get('success', False)}")
            
            # Sanity check: require a document URL/id to treat as success
            if not upload_result.get('success') or not (upload_result.get('document_url') or upload_result.get('document_id')):
                raise ValueError(f"Upload appears incomplete (no document URL/ID). Result: {upload_result}")
            
            # Step 8: Update pipeline state
            print("🔍 Step 8: Updating pipeline state...")
            updated_state = await self._update_pipeline_state(
                state,
                {
                    'classification_result': classification_result,
                    'title_result': title_result,
                    'image_result': image_result,
                    'pdf_result': pdf_result,
                    'genre_result': genre_result,
                    'upload_result': upload_result
                }
            )
            print("   ✅ Pipeline state updated")
            
            # Step 9: Create comprehensive results
            print("🔍 Step 9: Creating final results...")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 9 - upload_result: {upload_result}")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 9 - upload_result success: {upload_result.get('success')}")
            
            final_results = self._create_publishing_results(
                upload_result,
                {
                    'classification': classification_result,
                    'title': title_result,
                    'image': image_result,
                    'pdf': pdf_result,
                    'genre': genre_result,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 9 - final_results: {final_results}")
            
            self.logger.log_operation_success(
                "digitaldossier_publishing",
                {
                    "upload_successful": upload_result.get('success', False),
                    "document_url": upload_result.get('document_url'),
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "classification": classification_result['classification'],
                    "final_title": title_result['recommended_title']
                },
                operation_context
            )
            
            print("🔍 Step 10: Creating return result...")
            
            # Debug the upload result success determination
            upload_success = upload_result.get('success', False)
            upload_doc_url = upload_result.get('document_url')
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 10 - upload_success: {upload_success}")
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 10 - upload_doc_url: {upload_doc_url}")
            
            result = {
                'results': final_results,
                'state_updates': updated_state,
                'successful_platforms': ['digitaldossier'] if upload_success else [],
                'failed_platforms': [] if upload_success else ['digitaldossier'],
                'published_urls': [upload_doc_url] if upload_doc_url else [],
                'publishing_summary': final_results
            }
            
            self.logger.log_info("digitaldossier_publishing", f"🔧 DEBUG: Step 10 - final result: {result}")
            print(f"   ✅ Successful platforms: {result['successful_platforms']}")
            print(f"   ❌ Failed platforms: {result['failed_platforms']}")
            return result
            
        except Exception as e:
            error_msg = f"DigitalDossier publishing failed: {e}"
            self.logger.log_operation_failure("digitaldossier_publishing", error_msg, operation_context)
            
            # Update state with error
            error_state_updates = {
                'status': 'failed',
                'error_message': error_msg,
                'progress_percentage': state.get('progress_percentage', 85),
                'step_history': state.get('step_history', []) + ['publishing_failed'],
                'publishing_results': {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return {
                'results': {'success': False, 'error': error_msg},
                'state_updates': error_state_updates,
                'successful_platforms': [],
                'failed_platforms': [{'platform': 'digitaldossier', 'error': error_msg}],
                'published_urls': [],
                'publishing_summary': {
                    'success': False,
                    'error': error_msg,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            }
    
    async def _extract_content_from_state(
        self,
        channel_drafts: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract content from pipeline state."""
        
        self.logger.log_info("extract_content", "Extracting content from pipeline state")
        
        # DEBUG: Log available data sources
        self.logger.log_info("extract_content", f"🔧 DEBUG: ===== CONTENT EXTRACTION START =====")
        self.logger.log_info("extract_content", f"🔧 DEBUG: channel_drafts type: {type(channel_drafts)}")
        self.logger.log_info("extract_content", f"🔧 DEBUG: channel_drafts keys: {list(channel_drafts.keys()) if isinstance(channel_drafts, dict) else 'Not dict'}")
        self.logger.log_info("extract_content", f"🔧 DEBUG: channel_drafts value: {channel_drafts}")
        self.logger.log_info("extract_content", f"🔧 DEBUG: state has draft: {bool(state.get('draft'))}")
        self.logger.log_info("extract_content", f"🔧 DEBUG: state keys: {list(state.keys())}")
        
        # Log detailed channel_drafts content if available
        if isinstance(channel_drafts, dict):
            for platform, content in channel_drafts.items():
                self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} content type: {type(content)}")
                if isinstance(content, dict):
                    self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} content keys: {list(content.keys())}")
                    if 'content' in content:
                        content_length = len(content['content'])
                        title = content.get('title', 'No title')
                        self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} - Title: '{title}', Content: {content_length} chars")
                        if content_length > 0:
                            self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} - First 200 chars: {content['content'][:200]}")
                    else:
                        self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} - No 'content' key found")
                else:
                    self.logger.log_info("extract_content", f"🔧 DEBUG: {platform} - Content: {content}")
        
        # Log draft details if available
        if state.get('draft'):
            draft = state['draft']
            self.logger.log_info("extract_content", f"🔧 DEBUG: Draft type: {type(draft)}")
            if hasattr(draft, 'title'):
                self.logger.log_info("extract_content", f"🔧 DEBUG: Draft title: '{getattr(draft, 'title', 'No title')}'")
                self.logger.log_info("extract_content", f"🔧 DEBUG: Draft content length: {len(getattr(draft, 'content', ''))}")
                if hasattr(draft, 'content') and getattr(draft, 'content'):
                    content_preview = getattr(draft, 'content')[:200]
                    self.logger.log_info("extract_content", f"🔧 DEBUG: Draft content preview: {content_preview}")
            elif isinstance(draft, dict):
                self.logger.log_info("extract_content", f"🔧 DEBUG: Draft dict keys: {list(draft.keys())}")
                if 'content' in draft:
                    self.logger.log_info("extract_content", f"🔧 DEBUG: Draft content length: {len(draft['content'])}")
            else:
                self.logger.log_info("extract_content", f"🔧 DEBUG: Draft: {draft}")
        
        content = None
        
        # Priority 1: Try channel_drafts (from formatting stage)
        if channel_drafts and isinstance(channel_drafts, dict):
            self.logger.log_info("extract_content", f"🔧 DEBUG: Checking channel_drafts with keys: {list(channel_drafts.keys())}")
            
            # Check for digitaldossier format first (highest priority)
            if 'digitaldossier' in channel_drafts:
                dd_content = channel_drafts['digitaldossier']
                self.logger.log_info("extract_content", f"🔧 DEBUG: Found digitaldossier content type: {type(dd_content)}")
                if isinstance(dd_content, dict) and dd_content.get('content'):
                    content = dd_content
                    self.logger.log_info("extract_content", f"✅ Using digitaldossier channel content - {len(content['content'])} chars")
                    self.logger.log_info("extract_content", f"🔧 DEBUG: DigitalDossier title: '{content.get('title', 'No title')}'")
                else:
                    self.logger.log_warning("extract_content", f"🔧 DEBUG: DigitalDossier content invalid: {dd_content}")
                    
            # Fallback to personal_blog format
            elif 'personal_blog' in channel_drafts:
                pb_content = channel_drafts['personal_blog']
                self.logger.log_info("extract_content", f"🔧 DEBUG: Found personal_blog content type: {type(pb_content)}")
                if isinstance(pb_content, dict) and pb_content.get('content'):
                    content = pb_content
                    self.logger.log_info("extract_content", f"✅ Using personal_blog channel content - {len(content['content'])} chars")
                else:
                    self.logger.log_warning("extract_content", f"🔧 DEBUG: Personal blog content invalid: {pb_content}")
                    
            # Try any other available channel
            elif channel_drafts:
                self.logger.log_info("extract_content", f"🔧 DEBUG: No standard formats found, trying first available channel")
                first_channel_key = next(iter(channel_drafts.keys()))
                first_channel = channel_drafts[first_channel_key]
                self.logger.log_info("extract_content", f"🔧 DEBUG: First available channel '{first_channel_key}': {type(first_channel)}")
                
                if isinstance(first_channel, dict) and first_channel.get('content'):
                    content = first_channel
                    self.logger.log_info("extract_content", f"✅ Using first available channel content ({first_channel_key}) - {len(content['content'])} chars")
                else:
                    self.logger.log_warning("extract_content", f"🔧 DEBUG: First channel content invalid: {first_channel}")
        else:
            self.logger.log_info("extract_content", f"🔧 DEBUG: No valid channel_drafts found, proceeding to draft fallback")
        
        # Priority 2: Try state draft object
        if not content and state.get('draft'):
            draft = state['draft']
            # Handle both dict and object drafts
            if hasattr(draft, 'title') and hasattr(draft, 'content'):
                # Object with attributes
                content = {
                    'title': getattr(draft, 'title', 'Untitled'),
                    'content': getattr(draft, 'content', ''),
                    'summary': getattr(draft, 'abstract', '') or getattr(draft, 'summary', ''),
                    'meta_description': getattr(draft, 'meta_description', ''),
                    'keywords': getattr(draft, 'keywords', []),
                    'tags': getattr(draft, 'tags', []),
                    'word_count': getattr(draft, 'word_count', 0)
                }
                self.logger.log_info("extract_content", "✅ Using state draft object")
            elif isinstance(draft, dict) and draft.get('content'):
                # Dictionary draft
                content = draft
                self.logger.log_info("extract_content", "✅ Using state draft dict")
        
        # Priority 3: Emergency fallback - check if we have ANY content
        if not content or not content.get('content'):
            self.logger.log_error("extract_content", "❌ CRITICAL: No valid content found in any source!")
            self.logger.log_error("extract_content", f"❌ Channel drafts type: {type(channel_drafts)}")
            self.logger.log_error("extract_content", f"❌ Channel drafts content: {channel_drafts}")
            self.logger.log_error("extract_content", f"❌ State draft type: {type(state.get('draft'))}")
            self.logger.log_error("extract_content", f"❌ State draft content: {state.get('draft')}")
            self.logger.log_error("extract_content", f"❌ State keys: {list(state.keys())}")
            
            # CRITICAL: Raise exception immediately - do not create fallback content
            # This ensures the pipeline fails fast and doesn't upload dummy content
            raise ValueError(f"❌ CRITICAL ERROR: No valid content found for publishing. Pipeline run: {state.get('run_id')}. This indicates content generation or formatting stages failed. Check pipeline logs for previous stage errors.")
        
        # Ensure content has all required fields
        if isinstance(content, dict):
            content.setdefault('title', 'Untitled Document')
            content.setdefault('content', '')
            content.setdefault('summary', '')
            content.setdefault('meta_description', '')
            content.setdefault('keywords', [])
            content.setdefault('tags', [])
        
        # CRITICAL: Sanitize content to remove editorial metadata before validation
        content_text = content.get('content', '')
        if content_text:
            # Remove editorial audit sections that shouldn't appear in published content
            content_text = self._sanitize_editorial_metadata(content_text)
            content['content'] = content_text
            self.logger.log_info("extract_content", f"✅ Content sanitized, length after sanitization: {len(content_text)} chars")
        
        # Validate content is not empty
        if not content_text or len(content_text.strip()) < 100:
            self.logger.log_error("extract_content", f"❌ CRITICAL: Content is too short ({len(content_text)} chars)")
            self.logger.log_error("extract_content", f"❌ CRITICAL: Content preview: '{content_text[:200]}'")
            self.logger.log_error("extract_content", f"❌ CRITICAL: Full content object keys: {list(content.keys())}")
            raise ValueError(f"❌ CRITICAL ERROR: Content too short for publishing: {len(content_text)} characters. Minimum required: 100 characters.")
        
        # Log successful extraction with details
        content_length = len(content.get('content', ''))
        title = content.get('title', 'No title')
        self.logger.log_info("extract_content", f"🔧 DEBUG: ===== CONTENT EXTRACTION SUCCESS =====")
        self.logger.log_info("extract_content", f"🔧 DEBUG: Final content title: '{title}'")
        self.logger.log_info("extract_content", f"🔧 DEBUG: Final content length: {content_length} chars")
        self.logger.log_info("extract_content", f"🔧 DEBUG: Final content keys: {list(content.keys())}")
        self.logger.log_info("extract_content", f"🔧 DEBUG: Final content preview: {content.get('content', '')[:200]}")
        
        self.logger.log_info(
            "extract_content",
            "Content extracted and validated successfully",
            {
                "source": "channel_drafts" if channel_drafts else "state_draft",
                "has_title": bool(content.get('title')),
                "content_length": len(content.get('content', '')),
                "has_summary": bool(content.get('summary')),
                "word_count": content.get('word_count', 'unknown'),
                "final_title": content.get('title', 'No title')
            }
        )
        
        return content
    
    def _sanitize_editorial_metadata(self, content: str) -> str:
        """
        Remove editorial metadata sections from content before publishing.
        
        Strips sections like:
        - Editing Summary
        - Feedback Addressed  
        - Priorities Handled
        - Any content after delimiter patterns
        """
        
        # Log original content preview for debugging
        self.logger.log_info("sanitize_content", f"🔧 DEBUG: Original content length: {len(content)} chars")
        self.logger.log_info("sanitize_content", f"🔧 DEBUG: Content preview (first 300 chars): {content[:300]}")
        
        # Define patterns to identify editorial metadata sections
        editorial_patterns = [
            r'\n\s*#*\s*Editing Summary.*$',
            r'\n\s*#*\s*Feedback Addressed.*$', 
            r'\n\s*#*\s*Priorities Handled.*$',
            r'\n\s*#*\s*Changes Made.*$',
            r'\n\s*#*\s*Editorial Notes.*$',
            r'\n\s*#*\s*Quality Assessment.*$',
            r'\n\s*#*\s*Analysis:.*$',
            r'\n\s*#*\s*Summary:.*$',
            r'\n\s*•\s*Improvements Applied:.*$',
            r'\n\s*•\s*Changes Detected:.*$'
        ]
        
        # Look for delimiter patterns that separate main content from editorial metadata
        delimiter_patterns = [
            r'\n\s*—+\s*\n',  # Em dashes
            r'\n\s*-{3,}\s*\n',  # Multiple hyphens
            r'\n\s*={3,}\s*\n',  # Multiple equals signs
            r'\n\s*\*{3,}\s*\n'  # Multiple asterisks
        ]
        
        import re
        
        # First, try to find delimiter patterns and cut content there
        for delimiter_pattern in delimiter_patterns:
            match = re.search(delimiter_pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                # Cut content at the delimiter
                sanitized_content = content[:match.start()].strip()
                self.logger.log_info("sanitize_content", f"✅ Found delimiter pattern, cut content at position {match.start()}")
                self.logger.log_info("sanitize_content", f"✅ Content length after delimiter cut: {len(sanitized_content)} chars")
                return sanitized_content
        
        # If no delimiter found, look for specific editorial section headers
        for pattern in editorial_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                # Cut content at the start of the editorial section
                sanitized_content = content[:match.start()].strip()
                self.logger.log_info("sanitize_content", f"✅ Found editorial pattern '{pattern}', cut content at position {match.start()}")
                self.logger.log_info("sanitize_content", f"✅ Content length after pattern cut: {len(sanitized_content)} chars")
                return sanitized_content
        
        # If no editorial metadata patterns found, check for common transition phrases
        transition_patterns = [
            r'\n\s*Improvements Applied:.*$',
            r'\n\s*Changes Detected:.*$',
            r'\n\s*Quality Improvement Estimate:.*$',
            r'\n\s*Editing Effectiveness Ratio:.*$',
            r'\n\s*•\s*Enhanced.*$',
            r'\n\s*•\s*Simplified.*$',
            r'\n\s*•\s*Strengthened.*$',
            r'\n\s*•\s*Improved.*$',
            r'\n\s*•\s*Ensured.*$',
            r'\n\s*Confidence.*assessment.*$',
            r'\n\s*Moderate.*confidence.*$'
        ]
        
        for pattern in transition_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                # Cut content at the transition phrase
                sanitized_content = content[:match.start()].strip()
                self.logger.log_info("sanitize_content", f"✅ Found transition pattern '{pattern}', cut content at position {match.start()}")
                self.logger.log_info("sanitize_content", f"✅ Content length after transition cut: {len(sanitized_content)} chars")
                return sanitized_content
        
        # If no patterns found, return original content
        self.logger.log_info("sanitize_content", f"ℹ️  No editorial metadata patterns found, returning original content")
        return content
    
    async def _verify_api_connection(self):
        """Verify DigitalDossier API connection."""
        
        self.logger.log_info("verify_api", "Testing DigitalDossier API connection")
        
        try:
            connection_result = await self.api_tool.test_connection()
            
            if connection_result['status'] != 'success':
                raise Exception(f"API connection failed: {connection_result['message']}")
            
            self.logger.log_info(
                "verify_api",
                "API connection successful",
                connection_result
            )
            
        except Exception as e:
            self.logger.log_error("verify_api", f"API connection failed: {e}")
            raise
    
    async def _execute_content_analysis(
        self,
        content: Dict[str, Any],
        state: Dict[str, Any]
    ) -> tuple:
        """Execute content classification and title generation in parallel."""
        
        self.logger.log_info("content_analysis", "Starting content classification and title generation")
        
        # Execute classification first, then title generation
        classification_result = await self.category_classifier.execute(content, state)
        
        # COMMENTED OUT: Execute title generation using classification results
        # title_result = await self.title_generator.execute(
        #     content,
        #     classification_result['classification'],
        #     classification_result['analysis'],
        #     state
        # )
        
        # PRESERVE UPSTREAM TITLE: Create passthrough result using original title
        original_title = content.get('title', 'Untitled')
        title_result = {
            'original_title': original_title,
            'generated_titles': [],
            'recommended_title': original_title,  # Use original instead of LLM-generated
            'recommended_reasoning': 'Preserving upstream title (LLM generation disabled)',
            'recommended_score': 1.0,
            'classification': classification_result['classification'],
            'domain': classification_result.get('analysis', {}).get('domain', 'General'),
            'agent': 'PublisherAgent (passthrough)',
            'title_generation_disabled': True
        }
        
        self.logger.log_info(
            "content_analysis",
            "Content analysis completed",
            {
                "classification": classification_result['classification'],
                "confidence": classification_result['confidence_score'],
                "recommended_title": title_result['recommended_title'],
                "title_generation_disabled": True
            }
        )
        
        return classification_result, title_result
    
    async def _generate_pdf(
        self,
        content: Dict[str, Any],
        optimized_title: str
    ) -> Dict[str, Any]:
        """Generate PDF from content with optimized title."""
        
        self.logger.log_info("generate_pdf", "Generating PDF document")
        
        try:
            # Use optimized title instead of original
            pdf_content = content.copy()
            pdf_content['title'] = optimized_title
            # Sanitize editorial metadata before generating the PDF
            raw_content = pdf_content.get('content', '')
            sanitized = self._sanitize_editorial_metadata(raw_content) if isinstance(raw_content, str) else raw_content
            pdf_content['content'] = sanitized
            
            pdf_result = await self.pdf_generator.generate_pdf_from_dict(pdf_content)
            
            self.logger.log_info(
                "generate_pdf",
                "PDF generated successfully",
                {
                    "size_bytes": pdf_result['size_bytes'],
                    "filename": pdf_result['filename']
                }
            )
            
            return pdf_result
            
        except Exception as e:
            self.logger.log_error("generate_pdf", f"PDF generation failed: {e}")
            raise
    
    async def _upload_to_digitaldossier(
        self,
        title: str,
        category: str,
        genre_id: int,
        cover_image: Dict[str, str],
        pdf_file: Dict[str, str],
        genre_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload document to DigitalDossier API."""
        
        self.logger.log_info(
            "upload_document",
            "Uploading to DigitalDossier",
            {
                "title": title,
                "category": category,
                "genre_id": genre_id
            }
        )
        
        try:
            # CRITICAL: Validate PDF file object before upload
            if not pdf_file or not isinstance(pdf_file, dict):
                raise ValueError(f"Invalid PDF file object: {type(pdf_file)}")
            
            pdf_data_b64 = pdf_file.get('data', '')
            if not pdf_data_b64 or len(pdf_data_b64) < 1000:
                raise ValueError(f"PDF file object has invalid data: {len(pdf_data_b64)} chars")
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: PDF file object validation - data length: {len(pdf_data_b64)} chars")
            self.logger.log_info("upload_document", f"🔧 DEBUG: PDF file object keys: {list(pdf_file.keys())}")
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: About to call upload_document with title='{title}', category='{category}', genre_id={genre_id}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image type: {type(cover_image)}, PDF type: {type(pdf_file)}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image keys: {list(cover_image.keys()) if cover_image else 'None'}")
            
            upload_result = await self.api_tool.upload_document(
                title=title,
                category=category,
                genre_id=genre_id,
                cover_image=cover_image,
                pdf_file=pdf_file,
                genre_metadata=genre_metadata
                # Note: Excluding summary and content for PDF-only uploads to prevent duplicate text display
            )
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: Upload completed, result type: {type(upload_result)}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Upload result keys: {list(upload_result.keys()) if isinstance(upload_result, dict) else 'Not dict'}")
            
            # Add success indicator and extract URL if available
            # Handle API response format: {success: true, data: {...}}
            self.logger.log_info("upload_document", f"🔧 DEBUG: Raw upload_result: {upload_result}")
            
            response_data = upload_result.get('data', upload_result)  # Fallback to top-level if no 'data' field
            self.logger.log_info("upload_document", f"🔧 DEBUG: Extracted response_data: {response_data}")
            
            extracted_url = response_data.get('url') or response_data.get('document_url') or response_data.get('pdfUrl')
            extracted_id = response_data.get('id') or response_data.get('document_id')
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: Extracted URL: {extracted_url}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Extracted ID: {extracted_id}")
            
            result = {
                'success': True,
                'api_response': upload_result,
                'document_url': extracted_url,
                'document_id': extracted_id,
                'upload_timestamp': datetime.now().isoformat()
            }
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: Final result: {result}")
            
            self.logger.log_info(
                "upload_document",
                "Upload successful",
                {
                    "document_id": result.get('document_id'),
                    "document_url": result.get('document_url')
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.log_error("upload_document", f"🔧 DEBUG: Upload failed with exception type: {type(e)}")
            self.logger.log_error("upload_document", f"🔧 DEBUG: Upload failed with message: {str(e)}")
            
            # Log more details about the exception
            import traceback
            self.logger.log_error("upload_document", f"🔧 DEBUG: Full traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'upload_timestamp': datetime.now().isoformat()
            }
    
    async def _update_pipeline_state(
        self,
        state: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update pipeline state with publishing results."""
        
        # Set progress to 99% before final completion
        state_updates = {
            'progress_percentage': 99.0,
            'step_history': state.get('step_history', []) + ['publishing_completed'],
            'publishing_results': {
                'digitaldossier': {
                    'success': results['upload_result']['success'],
                    'document_url': results['upload_result'].get('document_url'),
                    'document_id': results['upload_result'].get('document_id'),
                    'classification': results['classification_result']['classification'],
                    'final_title': results['title_result']['recommended_title'],
                    'genre_id': results['genre_result']['genre_id'],
                    'genre_name': results['genre_result']['selected_genre'],
                    'processing_details': {
                        'classification_confidence': results['classification_result']['confidence_score'],
                        'title_score': results['title_result']['recommended_score'],
                        'image_placeholder': results['image_result']['is_placeholder'],
                        'pdf_size_bytes': results['pdf_result']['size_bytes']
                    }
                }
            },
            'published_urls': [results['upload_result'].get('document_url')] if results['upload_result'].get('document_url') else []
        }
        
        # Set final status
        if results['upload_result']['success']:
            state_updates['status'] = 'completed'
        else:
            state_updates['status'] = 'failed'
            state_updates['error_message'] = results['upload_result'].get('error', 'Upload failed')
        
        return state_updates
    
    def _create_publishing_results(
        self,
        upload_result: Dict[str, Any],
        processing_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive publishing results."""
        
        return {
            'platform': 'digitaldossier',
            'success': upload_result['success'],
            'document_url': upload_result.get('document_url'),
            'document_id': upload_result.get('document_id'),
            'error': upload_result.get('error'),
            'content_details': {
                'final_classification': processing_details['classification']['classification'],
                'classification_confidence': processing_details['classification']['confidence_score'],
                'original_title': processing_details['title']['original_title'],
                'final_title': processing_details['title']['recommended_title'],
                'title_optimization_score': processing_details['title']['recommended_score'],
                'cover_image_placeholder': processing_details['image']['is_placeholder'],
                'pdf_size_bytes': processing_details['pdf']['size_bytes'],
                'genre_id': processing_details['genre']['genre_id'],
                'genre_name': processing_details['genre']['selected_genre']
            },
            'processing_metrics': {
                'total_processing_time_seconds': processing_details['processing_time'],
                'classification_domain': processing_details['classification']['domain'],
                'api_tokens_used': (
                    processing_details['classification']['analysis'].get('tokens_used', 0) +
                    processing_details['title'].get('generation_details', {}).get('tokens_used', 0) +
                    processing_details['genre'].get('generation_details', {}).get('tokens_used', 0)
                )
            },
            'timestamp': datetime.now().isoformat(),
            'agent': 'PublisherAgent'
        }
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.api_tool.close()
        except Exception as e:
            self.logger.log_warning("cleanup", f"Error during cleanup: {e}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # Additional utility methods
    
    async def get_publishing_status(self, run_id: str) -> Dict[str, Any]:
        """Get publishing status for a specific pipeline run."""
        
        # This would integrate with database/state storage in full implementation
        return {
            'run_id': run_id,
            'publishing_status': 'completed',
            'platform_status': {
                'digitaldossier': 'published'
            },
            'last_updated': datetime.now().isoformat()
        }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate publisher configuration and dependencies."""
        
        validation_results = {
            'api_connection': False,
            'cover_images_available': False,
            'environment_variables': False,
            'dependencies': False,
            'errors': []
        }
        
        try:
            # Test API connection
            connection_result = await self.api_tool.test_connection()
            validation_results['api_connection'] = connection_result['status'] == 'success'
            if not validation_results['api_connection']:
                validation_results['errors'].append(f"API connection failed: {connection_result['message']}")
        
        except Exception as e:
            validation_results['errors'].append(f"API connection test failed: {e}")
        
        try:
            # Check cover images
            image_summary = await self.cover_image_agent.get_available_images_summary()
            validation_results['cover_images_available'] = image_summary['total_images'] > 0
            if not validation_results['cover_images_available']:
                validation_results['errors'].append("No cover images available in docs/cover-image/ directories")
        
        except Exception as e:
            validation_results['errors'].append(f"Cover image check failed: {e}")
        
        # Check environment variables for JWT authentication
        required_env_vars = [
            'DIGITALDOSSIER_BASE_URL',
            'DIGITALDOSSIER_ADMIN_EMAIL',
            'DIGITALDOSSIER_ADMIN_PASSWORD',
            'OPENAI_API_KEY'
        ]
        
        import os
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        validation_results['environment_variables'] = len(missing_vars) == 0
        if missing_vars:
            validation_results['errors'].append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        validation_results['overall_valid'] = (
            validation_results['api_connection'] and
            validation_results['environment_variables']
        )
        
        return validation_results
