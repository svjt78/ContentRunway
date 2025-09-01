"""Publishing API tool for multi-platform content distribution."""

import logging
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64
from urllib.parse import urljoin, quote
import hashlib

logger = logging.getLogger(__name__)


class PublishingAPITool:
    """Tool for publishing content to multiple platforms via their APIs."""
    
    def __init__(self):
        """Initialize publishing API tool."""
        
        # Platform configurations
        self.platform_configs = {
            'personal_blog': {
                'name': 'Personal Blog',
                'api_base_url': None,  # Custom implementation
                'auth_type': 'custom',
                'required_credentials': ['webhook_url', 'api_key'],
                'supported_formats': ['markdown', 'html'],
                'max_title_length': 200,
                'max_content_length': 500000,
                'supports_tags': True,
                'supports_canonical_url': True,
                'supports_publish_status': True,
                'rate_limits': {
                    'requests_per_hour': 1000,
                    'posts_per_day': 100
                }
            }
        }
        
        # Content transformation templates
        self.content_templates = {
            'personal_blog': {
                'title_prefix': '',
                'title_suffix': '',
                'content_header': '',
                'content_footer': '',
                'tag_format': 'lowercase',
                'max_tags': 10
            }
        }
        
        # Rate limiting tracking
        self.rate_limits = {}
        
        # Publishing status tracking
        self.publishing_history = []
    
    async def publish_to_platform(
        self,
        platform: str,
        content: Dict[str, Any],
        credentials: Dict[str, str],
        publish_settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Publish content to a specific platform.
        
        Args:
            platform: Target platform ('personal_blog')
            content: Content dictionary with title, body, tags, etc.
            credentials: Platform-specific credentials
            publish_settings: Additional publishing settings
            
        Returns:
            Dictionary with publishing results
        """
        logger.info(f"Publishing content to platform: {platform}")
        
        try:
            # Validate platform
            if platform not in self.platform_configs:
                raise ValueError(f"Unsupported platform: {platform}")
            
            platform_config = self.platform_configs[platform]
            publish_settings = publish_settings or {}
            
            # Validate credentials
            missing_creds = [
                cred for cred in platform_config['required_credentials']
                if cred not in credentials
            ]
            if missing_creds:
                return {
                    'success': False,
                    'error': f"Missing required credentials: {missing_creds}",
                    'platform': platform
                }
            
            # Check rate limits
            rate_limit_check = self._check_rate_limits(platform)
            if not rate_limit_check['allowed']:
                return {
                    'success': False,
                    'error': f"Rate limit exceeded: {rate_limit_check['message']}",
                    'platform': platform,
                    'retry_after': rate_limit_check.get('retry_after')
                }
            
            # Transform content for platform
            transformed_content = self._transform_content_for_platform(content, platform)
            
            # Validate content constraints
            validation_result = self._validate_content_constraints(transformed_content, platform_config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Content validation failed: {validation_result['errors']}",
                    'platform': platform
                }
            
            # Publish to platform
            if platform == 'personal_blog':
                result = await self._publish_to_personal_blog(transformed_content, credentials, publish_settings)
            else:
                raise ValueError(f"Publishing to {platform} not implemented")
            
            # Update rate limiting
            self._update_rate_limits(platform)
            
            # Track publishing history
            self._track_publishing_attempt(platform, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Publishing to {platform} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform,
                'timestamp': datetime.now().isoformat()
            }
    
    def _transform_content_for_platform(self, content: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Transform content to match platform requirements."""
        
        template = self.content_templates[platform]
        platform_config = self.platform_configs[platform]
        
        # Transform title
        title = content.get('title', '')
        max_title_length = platform_config['max_title_length']
        
        if max_title_length > 0 and len(title) > max_title_length:
            title = title[:max_title_length-3] + "..."
        
        # Transform content body
        body = content.get('body', '')
        
        # Add template elements
        if template['content_header']:
            body = template['content_header'] + body
        
        if template['content_footer']:
            body = body + template['content_footer']
        
        # Transform tags
        tags = content.get('tags', [])
        transformed_tags = self._transform_tags(tags, template)
        
        # Ensure content length constraints
        max_content_length = platform_config['max_content_length']
        if len(body) > max_content_length:
            # Truncate content intelligently
            body = self._truncate_content_intelligently(body, max_content_length)
        
        return {
            'title': title,
            'body': body,
            'tags': transformed_tags,
            'original_content': content,
            'platform': platform
        }
    
    def _transform_tags(self, tags: List[str], template: Dict[str, Any]) -> List[str]:
        """Transform tags according to platform requirements."""
        
        tag_format = template.get('tag_format', 'lowercase')
        max_tags = template.get('max_tags', 5)
        
        # Limit number of tags
        tags = tags[:max_tags]
        
        # Transform format
        if tag_format == 'hashtag':
            tags = [f"#{tag.replace(' ', '').replace('-', '')}" for tag in tags]
        elif tag_format == 'lowercase':
            tags = [tag.lower().replace(' ', '-') for tag in tags]
        
        return tags
    
    def _truncate_content_intelligently(self, content: str, max_length: int) -> str:
        """Truncate content while preserving structure."""
        
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundaries
        sentences = content.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= max_length - 20:  # Leave room for ellipsis
                truncated += sentence + '. '
            else:
                break
        
        if truncated:
            return truncated.rstrip() + "..."
        else:
            # Fallback to character truncation
            return content[:max_length-3] + "..."
    
    def _validate_content_constraints(self, content: Dict[str, Any], platform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content against platform constraints."""
        
        errors = []
        
        # Check title length
        title = content.get('title', '')
        max_title_length = platform_config['max_title_length']
        if max_title_length > 0 and len(title) > max_title_length:
            errors.append(f"Title too long: {len(title)} > {max_title_length}")
        
        # Check content length
        body = content.get('body', '')
        max_content_length = platform_config['max_content_length']
        if len(body) > max_content_length:
            errors.append(f"Content too long: {len(body)} > {max_content_length}")
        
        # Check tags
        tags = content.get('tags', [])
        if not platform_config['supports_tags'] and tags:
            errors.append("Platform does not support tags")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    
    
    
    async def _publish_to_personal_blog(
        self,
        content: Dict[str, Any],
        credentials: Dict[str, str],
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish content to personal blog via webhook."""
        
        try:
            webhook_url = credentials.get('webhook_url')
            api_key = credentials.get('api_key')
            
            if not webhook_url:
                return {
                    'success': False,
                    'error': 'Webhook URL required for personal blog publishing',
                    'platform': 'personal_blog'
                }
            
            # Prepare webhook payload
            payload = {
                'title': content['title'],
                'content': content['body'],
                'tags': content.get('tags', []),
                'publish_status': settings.get('publish_status', 'draft'),
                'author': settings.get('author', 'ContentRunway AI'),
                'category': settings.get('category', 'AI Content'),
                'canonical_url': settings.get('canonical_url'),
                'metadata': {
                    'created_by': 'ContentRunway',
                    'content_type': 'ai_generated',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status in [200, 201]:
                        response_data = await response.json()
                        
                        return {
                            'success': True,
                            'platform': 'personal_blog',
                            'post_id': response_data.get('post_id', 'unknown'),
                            'post_url': response_data.get('post_url'),
                            'publish_status': response_data.get('status', settings.get('publish_status')),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"Personal blog webhook error: {error_text}",
                            'status_code': response.status,
                            'platform': 'personal_blog'
                        }
        
        except Exception as e:
            logger.error(f"Personal blog publishing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'personal_blog'
            }
    
    
    def _check_rate_limits(self, platform: str) -> Dict[str, Any]:
        """Check if platform rate limits allow publishing."""
        
        now = datetime.now()
        platform_limits = self.platform_configs[platform]['rate_limits']
        
        if platform not in self.rate_limits:
            self.rate_limits[platform] = {
                'hourly_requests': 0,
                'daily_posts': 0,
                'last_reset_hour': now.hour,
                'last_reset_day': now.day
            }
        
        limits = self.rate_limits[platform]
        
        # Reset hourly counter
        if now.hour != limits['last_reset_hour']:
            limits['hourly_requests'] = 0
            limits['last_reset_hour'] = now.hour
        
        # Reset daily counter
        if now.day != limits['last_reset_day']:
            limits['daily_posts'] = 0
            limits['last_reset_day'] = now.day
        
        # Check limits
        if limits['hourly_requests'] >= platform_limits['requests_per_hour']:
            return {
                'allowed': False,
                'message': f"Hourly request limit exceeded ({platform_limits['requests_per_hour']})",
                'retry_after': 3600 - (now.minute * 60 + now.second)
            }
        
        if limits['daily_posts'] >= platform_limits['posts_per_day']:
            return {
                'allowed': False,
                'message': f"Daily post limit exceeded ({platform_limits['posts_per_day']})",
                'retry_after': 86400  # 24 hours
            }
        
        return {'allowed': True}
    
    def _update_rate_limits(self, platform: str):
        """Update rate limit counters after successful publishing."""
        
        if platform in self.rate_limits:
            self.rate_limits[platform]['hourly_requests'] += 1
            self.rate_limits[platform]['daily_posts'] += 1
    
    def _track_publishing_attempt(self, platform: str, result: Dict[str, Any]):
        """Track publishing attempt for analytics."""
        
        self.publishing_history.append({
            'platform': platform,
            'success': result.get('success', False),
            'timestamp': datetime.now().isoformat(),
            'post_id': result.get('post_id'),
            'error': result.get('error')
        })
        
        # Keep only last 100 entries
        if len(self.publishing_history) > 100:
            self.publishing_history = self.publishing_history[-100:]
    
    async def publish_to_multiple_platforms(
        self,
        platforms: List[str],
        content: Dict[str, Any],
        platform_credentials: Dict[str, Dict[str, str]],
        publish_settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Publish content to multiple platforms simultaneously.
        
        Args:
            platforms: List of platform names to publish to
            content: Content dictionary
            platform_credentials: Credentials for each platform
            publish_settings: Global publishing settings
            
        Returns:
            Dictionary with results for each platform
        """
        logger.info(f"Publishing to multiple platforms: {platforms}")
        
        publish_settings = publish_settings or {}
        
        # Create publishing tasks for each platform
        publishing_tasks = []
        
        for platform in platforms:
            if platform in platform_credentials:
                task = self.publish_to_platform(
                    platform=platform,
                    content=content,
                    credentials=platform_credentials[platform],
                    publish_settings=publish_settings.get(platform, {})
                )
                publishing_tasks.append((platform, task))
        
        # Execute all publishing tasks concurrently
        results = {}
        
        if publishing_tasks:
            task_results = await asyncio.gather(
                *[task for _, task in publishing_tasks],
                return_exceptions=True
            )
            
            for (platform, _), result in zip(publishing_tasks, task_results):
                if isinstance(result, Exception):
                    results[platform] = {
                        'success': False,
                        'error': str(result),
                        'platform': platform
                    }
                else:
                    results[platform] = result
        
        # Calculate summary statistics
        successful_publishes = sum(1 for result in results.values() if result.get('success', False))
        total_platforms = len(platforms)
        
        return {
            'multi_platform_publish': True,
            'platforms_attempted': platforms,
            'total_platforms': total_platforms,
            'successful_publishes': successful_publishes,
            'success_rate': round(successful_publishes / total_platforms, 3) if total_platforms > 0 else 0,
            'platform_results': results,
            'overall_success': successful_publishes == total_platforms,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_publishing_report(self, content: Dict[str, Any], platforms: List[str]) -> Dict[str, Any]:
        """Generate a report for publishing readiness across platforms."""
        
        logger.info("Generating publishing readiness report")
        
        platform_readiness = {}
        
        for platform in platforms:
            if platform not in self.platform_configs:
                platform_readiness[platform] = {
                    'ready': False,
                    'error': 'Unsupported platform'
                }
                continue
            
            platform_config = self.platform_configs[platform]
            
            # Check content compatibility
            issues = []
            warnings = []
            
            # Title length check
            title = content.get('title', '')
            max_title_length = platform_config['max_title_length']
            if max_title_length > 0 and len(title) > max_title_length:
                issues.append(f"Title too long ({len(title)} > {max_title_length})")
            
            # Content length check
            body = content.get('body', '')
            max_content_length = platform_config['max_content_length']
            if len(body) > max_content_length:
                warnings.append(f"Content will be truncated ({len(body)} > {max_content_length})")
            
            # Tags support check
            tags = content.get('tags', [])
            if tags and not platform_config['supports_tags']:
                warnings.append("Platform doesn't support tags - will be ignored")
            
            # Format support check
            content_format = content.get('format', 'markdown')
            if content_format not in platform_config['supported_formats']:
                warnings.append(f"Content format ({content_format}) may need conversion")
            
            # Rate limit check (simulated)
            rate_limit_status = self._check_rate_limits(platform)
            if not rate_limit_status['allowed']:
                issues.append(f"Rate limit exceeded: {rate_limit_status['message']}")
            
            platform_readiness[platform] = {
                'ready': len(issues) == 0,
                'platform_name': platform_config['name'],
                'issues': issues,
                'warnings': warnings,
                'content_transformations_needed': len(warnings) > 0,
                'estimated_publish_time': self._estimate_publish_time(platform, content)
            }
        
        # Calculate overall readiness
        ready_platforms = sum(1 for p in platform_readiness.values() if p['ready'])
        total_platforms = len(platforms)
        readiness_score = ready_platforms / total_platforms if total_platforms > 0 else 0
        
        return {
            'overall_readiness_score': round(readiness_score, 3),
            'ready_platforms': ready_platforms,
            'total_platforms': total_platforms,
            'platform_readiness': platform_readiness,
            'can_publish_immediately': readiness_score == 1.0,
            'blocked_platforms': [
                platform for platform, status in platform_readiness.items()
                if not status['ready']
            ],
            'recommendations': self._generate_publishing_recommendations(platform_readiness),
            'timestamp': datetime.now().isoformat()
        }
    
    def _estimate_publish_time(self, platform: str, content: Dict[str, Any]) -> str:
        """Estimate time required to publish to platform."""
        
        # Base times in seconds
        base_times = {
            'personal_blog': 4
        }
        
        base_time = base_times.get(platform, 5)
        
        # Adjust for content complexity
        content_length = len(content.get('body', ''))
        if content_length > 5000:
            base_time += 2
        
        # Adjust for tags
        tags = content.get('tags', [])
        if len(tags) > 3:
            base_time += 1
        
        return f"{base_time}-{base_time + 2} seconds"
    
    def _generate_publishing_recommendations(self, platform_readiness: Dict[str, Any]) -> List[str]:
        """Generate recommendations for publishing optimization."""
        
        recommendations = []
        
        # Check for common issues
        blocked_platforms = [
            platform for platform, status in platform_readiness.items()
            if not status['ready']
        ]
        
        if blocked_platforms:
            recommendations.append(f"Resolve blocking issues for: {', '.join(blocked_platforms)}")
        
        # Check for warnings
        platforms_with_warnings = [
            platform for platform, status in platform_readiness.items()
            if status.get('warnings')
        ]
        
        if platforms_with_warnings:
            recommendations.append("Review warnings for content transformations needed")
        
        # General recommendations
        recommendations.append("Test publishing to one platform before batch publishing")
        recommendations.append("Ensure all platform credentials are current and valid")
        
        if not blocked_platforms:
            recommendations.append("All platforms ready - content can be published immediately")
        
        return recommendations[:5]
    
    def get_platform_capabilities(self, platform: str = None) -> Dict[str, Any]:
        """Get capabilities and constraints for platform(s)."""
        
        if platform:
            if platform not in self.platform_configs:
                return {'error': f'Unknown platform: {platform}'}
            
            config = self.platform_configs[platform]
            template = self.content_templates[platform]
            
            return {
                'platform': platform,
                'name': config['name'],
                'api_base_url': config['api_base_url'],
                'supported_formats': config['supported_formats'],
                'content_constraints': {
                    'max_title_length': config['max_title_length'],
                    'max_content_length': config['max_content_length'],
                    'supports_tags': config['supports_tags'],
                    'max_tags': template.get('max_tags', 0)
                },
                'features': {
                    'supports_canonical_url': config['supports_canonical_url'],
                    'supports_publish_status': config['supports_publish_status'],
                    'thread_support': template.get('thread_support', False)
                },
                'rate_limits': config['rate_limits'],
                'required_credentials': config['required_credentials']
            }
        else:
            # Return all platforms
            return {
                'supported_platforms': list(self.platform_configs.keys()),
                'platform_details': {
                    platform: self.get_platform_capabilities(platform)
                    for platform in self.platform_configs.keys()
                }
            }
    
    def get_publishing_history(self, platform: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get publishing history with optional platform filtering."""
        
        history = self.publishing_history
        
        if platform:
            history = [h for h in history if h.get('platform') == platform]
        
        # Limit results
        history = history[-limit:] if limit else history
        
        # Calculate statistics
        total_attempts = len(history)
        successful_attempts = sum(1 for h in history if h.get('success', False))
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
        
        return {
            'publishing_history': history,
            'statistics': {
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'success_rate': round(success_rate, 3),
                'platforms_used': list(set(h.get('platform') for h in history)),
                'recent_failures': [
                    h for h in history[-5:]
                    if not h.get('success', False)
                ]
            }
        }
    
    async def test_platform_connection(self, platform: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test connection to platform API."""
        
        logger.info(f"Testing connection to {platform}")
        
        try:
            if platform not in self.platform_configs:
                return {
                    'success': False,
                    'error': f'Unsupported platform: {platform}',
                    'platform': platform
                }
            
            platform_config = self.platform_configs[platform]
            
            # Validate credentials
            missing_creds = [
                cred for cred in platform_config['required_credentials']
                if cred not in credentials
            ]
            if missing_creds:
                return {
                    'success': False,
                    'error': f'Missing credentials: {missing_creds}',
                    'platform': platform
                }
            
            # Test API connection
            if platform == 'personal_blog':
                return await self._test_personal_blog_connection(credentials)
            else:
                return {
                    'success': False,
                    'error': f'Connection test not implemented for {platform}',
                    'platform': platform
                }
        
        except Exception as e:
            logger.error(f"Connection test failed for {platform}: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform
            }
    
    
    
    
    async def _test_personal_blog_connection(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test personal blog webhook connection."""
        
        try:
            webhook_url = credentials.get('webhook_url')
            api_key = credentials.get('api_key')
            
            if not webhook_url:
                return {
                    'success': False,
                    'platform': 'personal_blog',
                    'error': 'Webhook URL required'
                }
            
            # Test webhook with ping
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'X-API-Key': api_key
                }
                
                test_payload = {
                    'action': 'ping',
                    'timestamp': datetime.now().isoformat()
                }
                
                async with session.post(
                    webhook_url,
                    headers=headers,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status in [200, 201, 204]:
                        return {
                            'success': True,
                            'platform': 'personal_blog',
                            'message': 'Personal blog webhook connection successful',
                            'webhook_url': webhook_url,
                            'response_status': response.status
                        }
                    else:
                        return {
                            'success': False,
                            'platform': 'personal_blog',
                            'error': f'Webhook returned status {response.status}',
                            'status_code': response.status
                        }
        
        except asyncio.TimeoutError:
            return {
                'success': False,
                'platform': 'personal_blog',
                'error': 'Webhook connection timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'platform': 'personal_blog',
                'error': str(e)
            }
    
    def create_publishing_schedule(
        self,
        content_list: List[Dict[str, Any]],
        platforms: List[str],
        schedule_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a publishing schedule respecting platform rate limits."""
        
        logger.info("Creating publishing schedule")
        
        schedule = []
        platform_queues = {platform: [] for platform in platforms}
        
        # Distribute content across platforms based on rate limits
        for content in content_list:
            for platform in platforms:
                platform_limits = self.platform_configs[platform]['rate_limits']
                daily_limit = platform_limits['posts_per_day']
                
                # Simple round-robin distribution for now
                if len(platform_queues[platform]) < daily_limit:
                    platform_queues[platform].append(content)
        
        # Create time-based schedule
        base_time = datetime.now()
        time_intervals = schedule_settings.get('interval_minutes', 30)
        
        schedule_items = []
        current_time = base_time
        
        for platform, content_queue in platform_queues.items():
            for i, content in enumerate(content_queue):
                schedule_items.append({
                    'scheduled_time': current_time.isoformat(),
                    'platform': platform,
                    'content_id': content.get('id', f'content_{i}'),
                    'content_title': content.get('title', 'Untitled'),
                    'estimated_duration': self._estimate_publish_time(platform, content)
                })
                
                # Stagger publishing times
                current_time = current_time.replace(minute=current_time.minute + time_intervals)
        
        return {
            'publishing_schedule': sorted(schedule_items, key=lambda x: x['scheduled_time']),
            'total_scheduled_posts': len(schedule_items),
            'schedule_duration_hours': len(schedule_items) * time_intervals / 60,
            'platforms_included': platforms,
            'content_distribution': {
                platform: len(queue) for platform, queue in platform_queues.items()
            },
            'schedule_created': datetime.now().isoformat()
        }