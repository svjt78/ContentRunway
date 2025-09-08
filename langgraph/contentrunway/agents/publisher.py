"""Publisher Agent - Complete DigitalDossier integration for ContentRunway."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .category_classifier_agent import CategoryClassifierAgent
from .title_generator_agent import TitleGeneratorAgent
from .cover_image_agent import CoverImageAgent
from ..tools.digitaldossier_api_tool import DigitalDossierAPITool
from ..tools.pdf_generator_tool import PDFGeneratorTool
from ..tools.genre_mapping_tool import GenreMappingTool
from ..utils.publisher_logger import PublisherLogger


class PublisherAgent:
    """
    Main Publisher Agent for DigitalDossier integration.
    
    Orchestrates the complete publishing workflow:
    1. Content classification (Blog vs Product)
    2. Title generation and optimization
    3. Cover image selection and processing
    4. PDF generation
    5. Genre mapping
    6. DigitalDossier API upload
    7. Pipeline state updates
    """
    
    def __init__(self):
        self.logger = PublisherLogger()
        
        # Initialize sub-agents
        self.category_classifier = CategoryClassifierAgent()
        self.title_generator = TitleGeneratorAgent()
        self.cover_image_agent = CoverImageAgent()
        
        # Initialize tools
        self.api_tool = DigitalDossierAPITool()
        self.pdf_generator = PDFGeneratorTool()
        self.genre_mapper = GenreMappingTool(self.api_tool)
    
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
            # Step 1: Extract content from state
            content = await self._extract_content_from_state(channel_drafts, state)
            
            # Step 2: Test API connection
            await self._verify_api_connection()
            
            # Step 3: Execute sub-agents in parallel where possible
            classification_result, title_result = await self._execute_content_analysis(content, state)
            
            # Step 4: Generate cover image (depends on classification)
            image_result = await self.cover_image_agent.execute(
                content,
                classification_result['classification'],
                classification_result['analysis'],
                title_result['recommended_title'],
                state
            )
            
            # Step 5: Generate PDF
            pdf_result = await self._generate_pdf(content, title_result['recommended_title'])
            
            # Step 6: Map to genre
            genre_id, genre_analysis = await self.genre_mapper.map_content_to_genre(
                title_result['recommended_title'],
                classification_result['classification'],
                classification_result['analysis'],
                content.get('summary', '')
            )
            
            # Step 7: Upload to DigitalDossier
            upload_result = await self._upload_to_digitaldossier(
                title=title_result['recommended_title'],
                category=classification_result['classification'],
                genre_id=genre_id,
                cover_image=image_result['api_object'],
                pdf_file=self.api_tool.create_pdf_file_object(
                    pdf_result['pdf_data'],
                    pdf_result['filename']
                ),
                summary=content.get('summary', ''),
                content_text=content.get('content', '')
            )
            
            # Step 8: Update pipeline state
            updated_state = await self._update_pipeline_state(
                state,
                {
                    'classification_result': classification_result,
                    'title_result': title_result,
                    'image_result': image_result,
                    'pdf_result': pdf_result,
                    'genre_analysis': genre_analysis,
                    'upload_result': upload_result
                }
            )
            
            # Step 9: Create comprehensive results
            final_results = self._create_publishing_results(
                upload_result,
                {
                    'classification': classification_result,
                    'title': title_result,
                    'image': image_result,
                    'pdf': pdf_result,
                    'genre': genre_analysis,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            )
            
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
            
            return {
                'results': final_results,
                'state_updates': updated_state,
                'successful_platforms': ['digitaldossier'],
                'failed_platforms': [],
                'published_urls': [upload_result.get('document_url')] if upload_result.get('document_url') else [],
                'publishing_summary': final_results
            }
            
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
        
        # Priority: channel_drafts -> draft -> state content
        if channel_drafts and hasattr(channel_drafts, 'personal_blog') and channel_drafts.personal_blog:
            content = channel_drafts.personal_blog
        elif channel_drafts and isinstance(channel_drafts, dict) and 'personal_blog' in channel_drafts:
            content = channel_drafts['personal_blog']
        elif state.get('draft'):
            content = state['draft']
        else:
            # Extract from state directly
            content = {
                'title': state.get('title', 'Untitled'),
                'content': state.get('content', ''),
                'summary': state.get('summary', ''),
                'tags': state.get('tags', [])
            }
        
        # Ensure content has required fields
        if isinstance(content, dict):
            content.setdefault('title', 'Untitled Document')
            content.setdefault('content', '')
            content.setdefault('summary', '')
        
        self.logger.log_info(
            "extract_content",
            "Content extracted successfully",
            {
                "has_title": bool(content.get('title')),
                "content_length": len(content.get('content', '')),
                "has_summary": bool(content.get('summary'))
            }
        )
        
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
        
        # Execute classification and title generation in parallel
        classification_task = self.category_classifier.execute(content, state)
        
        # Create a delayed title generation that waits for classification
        async def title_generation_with_classification():
            classification_result = await classification_task
            return await self.title_generator.execute(
                content,
                classification_result['classification'],
                classification_result['analysis'],
                state
            )
        
        classification_result, title_result = await asyncio.gather(
            classification_task,
            title_generation_with_classification()
        )
        
        self.logger.log_info(
            "content_analysis",
            "Content analysis completed",
            {
                "classification": classification_result['classification'],
                "confidence": classification_result['confidence_score'],
                "recommended_title": title_result['recommended_title']
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
        summary: Optional[str] = None,
        content_text: Optional[str] = None
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
            upload_result = await self.api_tool.upload_document(
                title=title,
                category=category,
                genre_id=genre_id,
                cover_image=cover_image,
                pdf_file=pdf_file,
                summary=summary,
                content=content_text
            )
            
            # Add success indicator and extract URL if available
            result = {
                'success': True,
                'api_response': upload_result,
                'document_url': upload_result.get('url') or upload_result.get('document_url'),
                'document_id': upload_result.get('id') or upload_result.get('document_id'),
                'upload_timestamp': datetime.now().isoformat()
            }
            
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
            self.logger.log_error("upload_document", f"Upload failed: {e}")
            
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
        
        state_updates = {
            'progress_percentage': 95.0,
            'step_history': state.get('step_history', []) + ['publishing_completed'],
            'publishing_results': {
                'digitaldossier': {
                    'success': results['upload_result']['success'],
                    'document_url': results['upload_result'].get('document_url'),
                    'document_id': results['upload_result'].get('document_id'),
                    'classification': results['classification_result']['classification'],
                    'final_title': results['title_result']['recommended_title'],
                    'genre_id': results['genre_analysis']['selected_genre_id'],
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
                'genre_id': processing_details['genre']['selected_genre_id'],
                'genre_name': processing_details['genre']['selected_genre_name']
            },
            'processing_metrics': {
                'total_processing_time_seconds': processing_details['processing_time'],
                'classification_domain': processing_details['classification']['domain'],
                'api_tokens_used': (
                    processing_details['classification']['analysis'].get('tokens_used', 0) +
                    processing_details['title'].get('generation_details', {}).get('tokens_used', 0) +
                    processing_details['genre'].get('tokens_used', 0)
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
        
        # Check environment variables
        required_env_vars = [
            'DIGITALDOSSIER_API_TOKEN',
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