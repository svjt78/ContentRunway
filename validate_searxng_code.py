#!/usr/bin/env python3
"""
Code validation for SearXNG integration.
Tests import structure and basic functionality without requiring running services.
"""

import sys
import os

# Add the langgraph directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langgraph'))

def test_imports():
    """Test that all SearXNG-related imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        from langgraph.contentrunway.tools.searxng_tool import SearXNGTool
        print("✅ SearXNGTool import successful")
        
        from langgraph.contentrunway.tools.web_research import WebResearchTool
        print("✅ WebResearchTool import successful")
        
        from langgraph.contentrunway.tools import SearXNGTool as ToolsSearXNG
        print("✅ Tools __init__ export successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_class_initialization():
    """Test that classes can be initialized correctly."""
    print("\n🔍 Testing class initialization...")
    
    try:
        from langgraph.contentrunway.tools.searxng_tool import SearXNGTool
        from langgraph.contentrunway.tools.web_research import WebResearchTool
        
        # Test SearXNGTool initialization
        searxng_tool = SearXNGTool("http://localhost/search")
        print("✅ SearXNGTool initialization successful")
        
        # Test WebResearchTool initialization with SearXNG
        web_research_tool = WebResearchTool("http://localhost/search")
        print("✅ WebResearchTool initialization with SearXNG successful")
        
        # Test search template structure
        templates = searxng_tool.search_templates
        domains = list(templates.get('twitter', {}).keys())
        print(f"✅ Search templates configured for domains: {domains}")
        
        return True
        
    except Exception as e:
        print(f"❌ Class initialization failed: {e}")
        return False


def test_method_signatures():
    """Test that key methods have correct signatures."""
    print("\n🔍 Testing method signatures...")
    
    try:
        from langgraph.contentrunway.tools.searxng_tool import SearXNGTool
        from langgraph.contentrunway.tools.web_research import WebResearchTool
        import inspect
        
        # Test SearXNGTool methods
        searxng_tool = SearXNGTool()
        
        required_methods = [
            'search_trending_content',
            'generate_creative_search_queries', 
            'extract_trending_topics'
        ]
        
        for method_name in required_methods:
            if hasattr(searxng_tool, method_name):
                method = getattr(searxng_tool, method_name)
                if callable(method):
                    sig = inspect.signature(method)
                    print(f"✅ {method_name}{sig}")
                else:
                    print(f"❌ {method_name} is not callable")
                    return False
            else:
                print(f"❌ {method_name} method not found")
                return False
        
        # Test WebResearchTool new methods
        web_research_tool = WebResearchTool()
        
        new_methods = ['search_trending_topics', 'enhanced_domain_search']
        
        for method_name in new_methods:
            if hasattr(web_research_tool, method_name):
                method = getattr(web_research_tool, method_name)
                if callable(method):
                    sig = inspect.signature(method)
                    print(f"✅ WebResearchTool.{method_name}{sig}")
                else:
                    print(f"❌ WebResearchTool.{method_name} is not callable")
                    return False
            else:
                print(f"❌ WebResearchTool.{method_name} method not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Method signature test failed: {e}")
        return False


def test_search_templates():
    """Test search template structure and content."""
    print("\n🔍 Testing search templates...")
    
    try:
        from langgraph.contentrunway.tools.searxng_tool import SearXNGTool
        
        searxng_tool = SearXNGTool()
        templates = searxng_tool.search_templates
        
        expected_platforms = ['twitter', 'linkedin']
        expected_domains = ['it_insurance', 'ai', 'agentic_ai', 'ai_software_engineering']
        
        # Check platform coverage
        for platform in expected_platforms:
            if platform not in templates:
                print(f"❌ Missing templates for platform: {platform}")
                return False
            print(f"✅ {platform} templates found")
        
        # Check domain coverage within each platform
        for platform in expected_platforms:
            platform_templates = templates[platform]
            for domain in expected_domains:
                if domain not in platform_templates:
                    print(f"❌ Missing {domain} templates for {platform}")
                    return False
                
                domain_queries = platform_templates[domain]
                if not isinstance(domain_queries, list) or len(domain_queries) == 0:
                    print(f"❌ Invalid templates for {platform}/{domain}")
                    return False
                
                print(f"✅ {platform}/{domain}: {len(domain_queries)} query templates")
        
        return True
        
    except Exception as e:
        print(f"❌ Search template test failed: {e}")
        return False


def main():
    """Run all code validation tests."""
    print("🚀 Starting SearXNG Code Validation")
    print("="*50)
    
    tests = [
        ("Import Structure", test_imports),
        ("Class Initialization", test_class_initialization),
        ("Method Signatures", test_method_signatures),
        ("Search Templates", test_search_templates)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Validation Results:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 Code validation successful! SearXNG integration code is ready.")
        print("\n📋 Next Steps:")
        print("   1. Start Docker services: docker-compose up -d")
        print("   2. Run integration test: python test_searxng_integration.py")
        print("   3. Configure .env with SEARXNG_SECRET_KEY")
    else:
        print("⚠️  Some validations failed. Check code structure.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)