#!/usr/bin/env python3
"""
Simple script to view Milvus data
"""

from pymilvus import connections, Collection, list_collections
import sys
import json

def connect_to_milvus():
    """Connect to Milvus instance."""
    try:
        connections.connect(
            alias="default",
            host="localhost",  # Milvus running on localhost
            port="19530"       # Default Milvus port
        )
        print("✅ Connected to Milvus")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def list_all_collections():
    """List all collections in Milvus."""
    try:
        collections = list_collections()
        print(f"\n📋 Collections in Milvus: {len(collections)}")
        for i, collection_name in enumerate(collections, 1):
            print(f"  {i}. {collection_name}")
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def view_collection_info(collection_name):
    """View detailed information about a collection."""
    try:
        collection = Collection(collection_name)
        collection.load()  # Load collection into memory
        
        print(f"\n🔍 Collection: {collection_name}")
        print(f"   Schema: {collection.schema}")
        print(f"   Count: {collection.num_entities}")
        
        # Get some sample data
        if collection.num_entities > 0:
            print(f"   📊 Sample data (first 5 records):")
            
            # Query first 5 records
            results = collection.query(
                expr="",  # Empty expression gets all records
                limit=5,
                output_fields=["*"]  # Get all fields
            )
            
            for i, record in enumerate(results):
                print(f"     Record {i+1}:")
                for key, value in record.items():
                    if isinstance(value, (list, dict)):
                        print(f"       {key}: {json.dumps(value, indent=8)[:200]}...")
                    else:
                        print(f"       {key}: {str(value)[:100]}...")
                print()
        
        return True
    except Exception as e:
        print(f"❌ Error viewing collection {collection_name}: {e}")
        return False

def main():
    """Main function to explore Milvus data."""
    print("🔍 Milvus Data Explorer")
    print("=" * 50)
    
    # Connect to Milvus
    if not connect_to_milvus():
        sys.exit(1)
    
    # List all collections
    collections = list_all_collections()
    
    if not collections:
        print("\n📭 No collections found in Milvus")
        return
    
    # View each collection
    for collection_name in collections:
        view_collection_info(collection_name)
        print("-" * 50)

if __name__ == "__main__":
    main()