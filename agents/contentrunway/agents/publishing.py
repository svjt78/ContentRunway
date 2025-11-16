"""Publishing Agent - Manages content publishing to multiple platforms."""

from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime
import uuid
import json

from ..state.pipeline_state import ChannelDrafts, PublishingResults, ContentPipelineState

logger = logging.getLogger(__name__)


class PublishingAgent:
    """Manages content publishing to multiple platforms with platform-specific APIs."""
    
    def __init__(self):
        # Platform configurations
        self.platform_configs = {
            'personal_blog': {
                'api_base': 'custom',  # Would be configured per user
                'auth_type': 'api_key',
                'supports_drafts': True,
                'supports_scheduling': True,
                'max_title_length': 200,
                'supported_formats': ['markdown', 'html']
            }
        }
        
        # Publishing modes
        self.publishing_modes = {
            'draft': 'Save as draft for later review',
            'schedule': 'Schedule for future publishing',
            'immediate': 'Publish immediately',
            'test': 'Test mode - simulate publishing'
        }
    
    async def execute(
        self,
        channel_drafts: ChannelDrafts,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Publish content to multiple platforms.
        
        Args:
            channel_drafts: Platform-specific formatted content
            state: Current pipeline state with publishing configuration
            
        Returns:
            Dictionary with publishing results and URLs
        """
        logger.info("Starting multi-platform content publishing")
        
        try:
            # Step 1: Determine publishing strategy
            publishing_strategy = self._determine_publishing_strategy(state)
            
            # Step 2: Prepare publishing tasks
            publishing_tasks = self._prepare_publishing_tasks(
                channel_drafts,
                publishing_strategy,
                state
            )
            
            # Step 3: Execute publishing in parallel
            publishing_results = await self._execute_publishing_tasks(publishing_tasks)
            
            # Step 4: Process results and generate URLs
            final_results = self._process_publishing_results(publishing_results, state)
            
            # Step 5: Create PublishingResults object
            results_obj = PublishingResults(**final_results['platform_results'])
            
            logger.info(f"Publishing completed for {len(final_results['successful_platforms'])} platforms")
            
            return {
                'results': results_obj,
                'urls': final_results['published_urls'],
                'successful_platforms': final_results['successful_platforms'],
                'failed_platforms': final_results['failed_platforms'],
                'publishing_summary': final_results['summary']
            }
            
        except Exception as e:
            logger.error(f"Publishing process failed: {e}")
            return self._create_fallback_publishing_results(channel_drafts, state, str(e))
    
    def _determine_publishing_strategy(self, state: ContentPipelineState) -> Dict[str, Any]:
        """Determine the publishing strategy based on state configuration."""
        
        # Check for publishing configuration in state
        config_overrides = state.get('config_overrides', {})
        
        # Default strategy for Phase 1 (single-tenant personal use)
        strategy = {
            'mode': config_overrides.get('publishing_mode', 'draft'),  # Default to draft mode
            'platforms': config_overrides.get('target_platforms', ['personal_blog']),
            'schedule_time': config_overrides.get('schedule_time'),
            'auto_publish': config_overrides.get('auto_publish', False),
            'test_mode': config_overrides.get('test_mode', True),  # Default to test mode for safety
            'backup_on_failure': True
        }
        
        # Override for human-approved content
        if state.get('human_review_feedback', {}).get('decision') == 'approved':
            if strategy['mode'] == 'draft' and strategy.get('auto_publish'):
                strategy['mode'] = 'immediate'
        
        return strategy
    
    def _prepare_publishing_tasks(
        self,
        channel_drafts: ChannelDrafts,
        strategy: Dict[str, Any],
        state: ContentPipelineState
    ) -> List[Dict[str, Any]]:
        """Prepare individual publishing tasks for each platform."""
        
        tasks = []
        
        # Get available channel drafts
        available_platforms = self._get_available_platforms(channel_drafts)
        
        for platform in strategy['platforms']:
            if platform not in available_platforms:
                logger.warning(f"No content available for platform: {platform}")
                continue
            
            # Get platform-specific content
            platform_content = getattr(channel_drafts, platform, None)
            if not platform_content:
                continue
            
            # Create publishing task
            task = {
                'platform': platform,
                'content': platform_content,
                'strategy': strategy,
                'config': self.platform_configs.get(platform, {}),
                'credentials': self._get_platform_credentials(platform, state),
                'task_id': str(uuid.uuid4())
            }
            
            tasks.append(task)
        
        return tasks
    
    def _get_available_platforms(self, channel_drafts: ChannelDrafts) -> List[str]:
        """Get list of platforms with available content."""
        
        available = []
        
        if channel_drafts.personal_blog:
            available.append('personal_blog')
        
        return available
    
    def _get_platform_credentials(self, platform: str, state: ContentPipelineState) -> Dict[str, Any]:
        """Get platform-specific authentication credentials."""
        
        # In real implementation, this would securely retrieve credentials
        # For now, return placeholder configuration
        
        credentials = {
            'personal_blog': {
                'auth_type': 'api_key',
                'api_key': 'BLOG_API_KEY',
                'endpoint': 'https://user-blog.com/api',
                'user_id': 'USER_ID'
            }
        }
        
        return credentials.get(platform, {})
    
    async def _execute_publishing_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute publishing tasks in parallel."""
        
        # Create async tasks for parallel execution
        async_tasks = [
            self._publish_to_platform(task)
            for task in tasks
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Process results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'task_id': tasks[i]['task_id'],
                    'platform': tasks[i]['platform'],
                    'success': False,
                    'error': str(result),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _publish_to_platform(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Publish content to a specific platform."""
        
        platform = task['platform']
        content = task['content']
        strategy = task['strategy']
        config = task['config']
        
        logger.info(f"Publishing to {platform} in {strategy['mode']} mode")
        
        try:
            # Platform-specific publishing logic
            if platform == 'personal_blog':
                result = await self._publish_to_personal_blog(content, strategy, config, task)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            return {
                'task_id': task['task_id'],
                'platform': platform,
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Publishing to {platform} failed: {e}")
            return {
                'task_id': task['task_id'],
                'platform': platform,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    
    async def _publish_to_personal_blog(
        self,
        content: Dict[str, Any],
        strategy: Dict[str, Any],
        config: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish content to personal blog."""
        
        if strategy['test_mode']:
            await asyncio.sleep(0.3)
            
            slug = content.get('title', 'article').replace(' ', '-').lower()
            
            return {
                'platform': 'personal_blog',
                'status': 'draft' if strategy['mode'] == 'draft' else 'published',
                'url': f"https://user-blog.com/posts/{slug}",
                'post_id': f"blog_{uuid.uuid4().hex[:8]}",
                'published_at': datetime.now().isoformat() if strategy['mode'] == 'immediate' else None,
                'slug': slug,
                'categories': content.get('tags', [])[:3]
            }
        
        return {
            'platform': 'personal_blog',
            'status': 'test_mode',
            'url': f"https://user-blog.com/test/{uuid.uuid4().hex[:8]}",
            'post_id': f"test_blog_{uuid.uuid4().hex[:8]}",
            'message': 'Test mode - no actual publishing performed'
        }
    
    
    
    def _process_publishing_results(
        self,
        results: List[Dict[str, Any]],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Process and organize publishing results."""
        
        successful_platforms = []
        failed_platforms = []
        published_urls = []
        platform_results = {}
        
        for result in results:
            platform = result['platform']
            
            if result['success']:
                successful_platforms.append(platform)
                
                platform_result = result['result']
                platform_results[platform] = platform_result
                
                if platform_result.get('url'):
                    published_urls.append(platform_result['url'])
                    
            else:
                failed_platforms.append({
                    'platform': platform,
                    'error': result['error']
                })
        
        # Generate publishing summary
        summary = self._generate_publishing_summary(
            successful_platforms,
            failed_platforms,
            platform_results
        )
        
        return {
            'successful_platforms': successful_platforms,
            'failed_platforms': failed_platforms,
            'published_urls': published_urls,
            'platform_results': platform_results,
            'summary': summary
        }
    
    def _generate_publishing_summary(
        self,
        successful: List[str],
        failed: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive publishing summary."""
        
        total_platforms = len(successful) + len(failed)
        success_rate = (len(successful) / total_platforms * 100) if total_platforms > 0 else 0
        
        # Count drafts vs published
        drafts_created = sum(1 for result in results.values() if result.get('status') == 'draft')
        content_published = sum(1 for result in results.values() if result.get('status') == 'published')
        
        # Collect platform-specific metrics
        platform_metrics = {}
        for platform, result in results.items():
            platform_metrics[platform] = {
                'status': result.get('status'),
                'url': result.get('url'),
                'post_id': result.get('post_id'),
                'published_at': result.get('published_at')
            }
        
        return {
            'total_platforms_attempted': total_platforms,
            'successful_platforms': len(successful),
            'failed_platforms': len(failed),
            'success_rate_percentage': round(success_rate, 1),
            'drafts_created': drafts_created,
            'content_published': content_published,
            'platform_metrics': platform_metrics,
            'failed_platform_errors': {item['platform']: item['error'] for item in failed},
            'publishing_timestamp': datetime.now().isoformat()
        }
    
    def _create_fallback_publishing_results(
        self,
        channel_drafts: ChannelDrafts,
        state: ContentPipelineState,
        error: str
    ) -> Dict[str, Any]:
        """Create fallback publishing results when the process fails."""
        
        available_platforms = self._get_available_platforms(channel_drafts)
        
        # Create minimal results structure
        fallback_results = PublishingResults()
        
        # Set fallback data for available platforms
        for platform in available_platforms:
            if hasattr(fallback_results, platform):
                setattr(fallback_results, platform, {
                    'platform': platform,
                    'status': 'failed',
                    'error': error,
                    'url': None,
                    'post_id': None
                })
        
        return {
            'results': fallback_results,
            'urls': [],
            'successful_platforms': [],
            'failed_platforms': [{'platform': p, 'error': error} for p in available_platforms],
            'publishing_summary': {
                'total_platforms_attempted': len(available_platforms),
                'successful_platforms': 0,
                'failed_platforms': len(available_platforms),
                'success_rate_percentage': 0.0,
                'error': error
            }
        }
    
    def get_publishing_status(self, run_id: str) -> Dict[str, Any]:
        """Get publishing status for a specific pipeline run."""
        
        # In real implementation, this would query database for publishing status
        return {
            'run_id': run_id,
            'publishing_status': 'completed',  # pending, in_progress, completed, failed
            'platforms_status': {
                'personal_blog': 'draft'
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def retry_failed_publishing(
        self,
        run_id: str,
        failed_platforms: List[str]
    ) -> Dict[str, Any]:
        """Retry publishing for specific failed platforms."""
        
        # In real implementation, this would retry publishing to failed platforms
        logger.info(f"Retrying publishing for {len(failed_platforms)} platforms")
        
        return {
            'retry_initiated': True,
            'platforms_to_retry': failed_platforms,
            'estimated_completion': (datetime.now().isoformat()),
            'retry_id': str(uuid.uuid4())
        }