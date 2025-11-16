#!/usr/bin/env python3
"""
Quality Gate Cache Bypass Script

This script provides utilities to bypass or clear quality gate caches 
when deploying scoring improvements.

Usage:
    python scripts/quality_gate_cache_bypass.py --action bypass
    python scripts/quality_gate_cache_bypass.py --action clear
    python scripts/quality_gate_cache_bypass.py --action analyze --content "draft content here"
"""

import sys
import os
import asyncio
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.contentrunway.agents.quality_gates import (
    force_cache_bypass,
    invalidate_quality_gate_caches,
    analyze_draft_structure,
    QUALITY_GATE_SCORING_VERSION
)


async def main():
    parser = argparse.ArgumentParser(description="Quality Gate Cache Management")
    parser.add_argument(
        "--action", 
        choices=["bypass", "clear", "analyze", "status"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--content",
        help="Draft content to analyze (for analyze action)"
    )
    
    args = parser.parse_args()
    
    if args.action == "status":
        print(f"📊 Current Quality Gate Scoring Version: {QUALITY_GATE_SCORING_VERSION}")
        
    elif args.action == "bypass":
        new_version = force_cache_bypass()
        print(f"🚫 Cache bypass activated! New version: {new_version}")
        print("All quality gates will now recalculate without using cached results.")
        
    elif args.action == "clear":
        print("🗑️  Clearing all quality gate caches...")
        cleared_count = await invalidate_quality_gate_caches()
        print(f"✅ Cache invalidation complete: {cleared_count} keys cleared")
        
    elif args.action == "analyze":
        if not args.content:
            print("❌ Error: --content required for analyze action")
            return
            
        print("🔍 Analyzing draft content structure...")
        analysis = analyze_draft_structure(args.content)
        
        print("\n📋 ANALYSIS RESULTS:")
        print("="*50)
        
        for gate_name, signals in analysis.items():
            if gate_name == "overall_assessment":
                continue
                
            print(f"\n{gate_name.upper()}:")
            for key, value in signals.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {len(value)} items (showing first 3: {value[:3]})")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n🎯 OVERALL READINESS:")
        assessment = analysis["overall_assessment"]
        for gate, ready in assessment.items():
            status = "✅ READY" if ready else "❌ NEEDS WORK"
            print(f"  {gate}: {status}")
        
        all_ready = all(assessment.values())
        print(f"\n📈 Pipeline Ready: {'✅ YES' if all_ready else '❌ NO'}")


if __name__ == "__main__":
    asyncio.run(main())