#!/usr/bin/env python3
"""
Test script for SearXNG integration with ContentRunway.
Run this after starting the Docker services to verify trending content discovery.
"""

import asyncio
import sys
import os

# Add the langgraph directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langgraph'))

from langgraph.contentrunway.tools.searxng_tool import SearXNGTool
from langgraph.contentrunway.tools.web_research import WebResearchTool


async def test_searxng_basic():
    """Test basic SearXNG functionality."""
    print("🔍 Testing SearXNG basic search...")
    
    searxng_tool = SearXNGTool("http://localhost/search")
    
    try:
        # Test creative query generation
        queries = await searxng_tool.generate_creative_search_queries(
            base_topic="AI agents",
            domain="agentic_ai",
            platforms=["twitter", "linkedin"]
        )
        
        print(f"✅ Generated {len(queries)} creative search queries:")
        for i, query in enumerate(queries[:3], 1):
            print(f"   {i}. {query}")
        
        # Test trending content search
        trending_content = await searxng_tool.search_trending_content(
            domain="agentic_ai",
            platforms=["twitter", "linkedin"],
            max_results_per_platform=3
        )
        
        print(f"✅ Found {len(trending_content)} trending content items")
        for item in trending_content[:2]:
            print(f"   - {item.get('title', 'No title')[:60]}... (Engagement: {item.get('engagement_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ SearXNG basic test failed: {e}")
        return False


async def test_web_research_integration():
    """Test WebResearchTool integration with SearXNG."""
    print("\n🔍 Testing WebResearchTool integration...")
    
    web_research_tool = WebResearchTool("http://localhost/search")
    
    try:
        # Test enhanced domain search
        enhanced_results = await web_research_tool.enhanced_domain_search(
            query="multi-agent systems",
            domain="agentic_ai",
            include_trending=True,
            max_sources=10
        )
        
        print(f"✅ Enhanced search returned {len(enhanced_results)} sources")
        
        # Categorize results
        traditional_sources = [s for s in enhanced_results if s.get('source_type') not in ['trending_post', 'trending_topic_idea']]
        trending_sources = [s for s in enhanced_results if s.get('source_type') in ['trending_post', 'trending_topic_idea']]
        
        print(f"   - Traditional sources: {len(traditional_sources)}")
        print(f"   - Trending sources: {len(trending_sources)}")
        
        # Show top trending source
        if trending_sources:
            top_trending = trending_sources[0]
            print(f"   - Top trending: {top_trending.get('title', 'No title')[:60]}... (Platform: {top_trending.get('platform', 'unknown')})")
        
        return True
        
    except Exception as e:
        print(f"❌ WebResearchTool integration test failed: {e}")
        return False


async def test_research_coordinator_integration():
    """Test ResearchCoordinatorAgent integration."""
    print("\n🔍 Testing ResearchCoordinatorAgent integration...")
    
    try:
        from langgraph.contentrunway.agents.research import ResearchCoordinatorAgent
        from langgraph.contentrunway.state.pipeline_state import ContentPipelineState
        
        research_coordinator = ResearchCoordinatorAgent()
        
        # Create minimal state for testing
        test_state = ContentPipelineState(
            run_id="test-run-123",
            research_query="AI automation trends",
            domain_focus=["agentic_ai"],
            config_overrides={"include_trending": True}
        )
        
        # Execute research
        research_results = await research_coordinator.execute(
            query="AI automation trends",
            domains=["agentic_ai"], 
            state=test_state
        )
        
        sources = research_results.get('sources', [])
        topics = research_results.get('topics', [])
        
        print(f"✅ Research coordination completed:")
        print(f"   - Total sources: {len(sources)}")
        print(f"   - Generated topics: {len(topics)}")
        
        # Show breakdown of source types
        traditional_count = len([s for s in sources if s.get('source_type') not in ['trending_post', 'trending_topic_idea']])
        trending_count = len(sources) - traditional_count
        
        print(f"   - Traditional sources: {traditional_count}")
        print(f"   - Trending sources: {trending_count}")
        
        if topics:
            top_topic = topics[0]
            print(f"   - Top topic: {top_topic.title[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ ResearchCoordinatorAgent test failed: {e}")
        print(f"   Error details: {str(e)}")
        return False


async def main():
    """Run all SearXNG integration tests."""
    print("🚀 Starting SearXNG Integration Tests")
    print("="*50)
    
    # Check if Docker services are running
    print("📋 Prerequisites:")
    print("   - Ensure Docker services are running: docker-compose up -d")
    print("   - SearXNG should be accessible at http://localhost:8080")
    print("   - Nginx proxy should route /search/ to SearXNG")
    print()
    
    tests = [
        ("SearXNG Basic", test_searxng_basic),
        ("WebResearchTool Integration", test_web_research_integration), 
        ("ResearchCoordinatorAgent Integration", test_research_coordinator_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SearXNG integration is ready.")
        print("\n💡 Usage Tips:")
        print("   - Trending content discovery is now integrated into the research pipeline")
        print("   - Set SEARXNG_ENABLED=false in .env to disable trending content")
        print("   - Adjust SEARXNG_RATE_LIMIT to control search frequency")
    else:
        print("⚠️  Some tests failed. Check Docker services and configuration.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)