#!/usr/bin/env python3
"""
Simple script to view Redis data
"""

import redis
import json
import sys
from datetime import datetime

def connect_to_redis():
    """Connect to Redis instance."""
    try:
        r = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        # Test connection
        r.ping()
        print("✅ Connected to Redis")
        return r
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return None

def explore_redis_data(r):
    """Explore all Redis data."""
    try:
        # Get basic info
        info = r.info()
        print(f"\n📊 Redis Info:")
        print(f"   Version: {info.get('redis_version', 'Unknown')}")
        print(f"   Memory Used: {info.get('used_memory_human', 'Unknown')}")
        print(f"   Connected Clients: {info.get('connected_clients', 'Unknown')}")
        
        # Get all keys
        keys = r.keys("*")
        print(f"\n🔑 Total Keys: {len(keys)}")
        
        if not keys:
            print("📭 No keys found in Redis")
            return
        
        # Group keys by pattern
        key_patterns = {}
        for key in keys:
            pattern = key.split(':')[0] if ':' in key else 'other'
            if pattern not in key_patterns:
                key_patterns[pattern] = []
            key_patterns[pattern].append(key)
        
        print(f"\n📋 Key Patterns:")
        for pattern, pattern_keys in key_patterns.items():
            print(f"   {pattern}: {len(pattern_keys)} keys")
        
        # Explore each key
        print(f"\n🔍 Key Details:")
        for i, key in enumerate(keys[:20]):  # Limit to first 20 keys
            key_type = r.type(key)
            ttl = r.ttl(key)
            ttl_info = f"TTL: {ttl}s" if ttl > 0 else "No expiry" if ttl == -1 else "Expired"
            
            print(f"\n   {i+1}. Key: {key}")
            print(f"      Type: {key_type}")
            print(f"      {ttl_info}")
            
            # Get value based on type
            try:
                if key_type == 'string':
                    value = r.get(key)
                    if len(str(value)) > 200:
                        print(f"      Value: {str(value)[:200]}... (truncated)")
                    else:
                        print(f"      Value: {value}")
                        
                elif key_type == 'hash':
                    hash_data = r.hgetall(key)
                    print(f"      Fields: {len(hash_data)}")
                    for field, field_value in list(hash_data.items())[:5]:
                        print(f"        {field}: {str(field_value)[:100]}")
                    if len(hash_data) > 5:
                        print(f"        ... and {len(hash_data) - 5} more fields")
                        
                elif key_type == 'list':
                    list_len = r.llen(key)
                    print(f"      Length: {list_len}")
                    if list_len > 0:
                        items = r.lrange(key, 0, 4)  # Get first 5 items
                        for idx, item in enumerate(items):
                            print(f"        [{idx}]: {str(item)[:100]}")
                        if list_len > 5:
                            print(f"        ... and {list_len - 5} more items")
                            
                elif key_type == 'set':
                    set_size = r.scard(key)
                    print(f"      Size: {set_size}")
                    if set_size > 0:
                        members = list(r.smembers(key))[:5]  # Get first 5 members
                        for member in members:
                            print(f"        - {str(member)[:100]}")
                        if set_size > 5:
                            print(f"        ... and {set_size - 5} more members")
                            
                elif key_type == 'zset':
                    zset_size = r.zcard(key)
                    print(f"      Size: {zset_size}")
                    if zset_size > 0:
                        items = r.zrange(key, 0, 4, withscores=True)
                        for member, score in items:
                            print(f"        {member}: {score}")
                        if zset_size > 5:
                            print(f"        ... and {zset_size - 5} more items")
                            
            except Exception as e:
                print(f"      Error reading value: {e}")
        
        if len(keys) > 20:
            print(f"\n   ... and {len(keys) - 20} more keys")
            
    except Exception as e:
        print(f"❌ Error exploring Redis: {e}")

def search_redis_patterns(r, pattern="*"):
    """Search for keys matching a pattern."""
    print(f"\n🔍 Searching for pattern: {pattern}")
    keys = r.keys(pattern)
    
    if not keys:
        print("   No matching keys found")
        return
    
    print(f"   Found {len(keys)} matching keys:")
    for key in keys[:10]:  # Show first 10
        print(f"   - {key}")
    
    if len(keys) > 10:
        print(f"   ... and {len(keys) - 10} more")

def main():
    """Main function to explore Redis data."""
    print("🔍 Redis Data Explorer")
    print("=" * 50)
    
    # Connect to Redis
    r = connect_to_redis()
    if not r:
        sys.exit(1)
    
    # Explore all data
    explore_redis_data(r)
    
    # Search for common patterns
    common_patterns = [
        "cache:*",
        "session:*", 
        "user:*",
        "research:*",
        "pipeline:*",
        "*celery*",
        "*kombu*"
    ]
    
    print(f"\n🔍 Pattern Search:")
    for pattern in common_patterns:
        matching_keys = r.keys(pattern)
        if matching_keys:
            print(f"   {pattern}: {len(matching_keys)} keys")
            
if __name__ == "__main__":
    main()