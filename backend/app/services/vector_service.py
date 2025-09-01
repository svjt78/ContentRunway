from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorService:
    """Service for managing vector database operations with Milvus."""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.connection_alias = "default"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Collection names (Phase 1: single tenant, so no prefixes needed)
        self.collections = {
            "research_sources": "research_sources",
            "content_drafts": "content_drafts", 
            "published_content": "published_content",
            "domain_knowledge": "domain_knowledge",
            "style_examples": "style_examples"
        }
        
        self._connect()
        self._initialize_collections()
    
    def _connect(self):
        """Connect to Milvus vector database."""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection_schema(self, collection_name: str) -> CollectionSchema:
        """Create schema for a collection."""
        if collection_name == "research_sources":
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="pipeline_run_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="credibility_score", dtype=DataType.FLOAT),
                FieldSchema(name="relevance_score", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
        
        elif collection_name == "content_drafts":
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="pipeline_run_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="topic_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="stage", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="word_count", dtype=DataType.INT32),
                FieldSchema(name="final_quality_score", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
        
        elif collection_name == "published_content":
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="pipeline_run_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="platform", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="published_url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="performance_score", dtype=DataType.FLOAT),
                FieldSchema(name="engagement_metrics", dtype=DataType.VARCHAR, max_length=2000),  # JSON as string
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
        
        elif collection_name == "domain_knowledge":
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="concept", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="confidence_score", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
        
        elif collection_name == "style_examples":
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="style_category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="text_sample", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="tone", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="quality_score", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
        
        else:
            raise ValueError(f"Unknown collection name: {collection_name}")
        
        return CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
    
    def _initialize_collections(self):
        """Initialize all collections if they don't exist."""
        for collection_key, collection_name in self.collections.items():
            if not utility.has_collection(collection_name):
                schema = self._create_collection_schema(collection_key)
                collection = Collection(
                    name=collection_name,
                    schema=schema,
                    using=self.connection_alias
                )
                
                # Create index for vector similarity search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding."""
        return self.encoder.encode([text])[0].tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts to vector embeddings."""
        embeddings = self.encoder.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    async def insert_research_source(
        self,
        source_id: str,
        pipeline_run_id: str,
        url: str,
        title: str,
        domain: str,
        source_type: str,
        summary: str,
        credibility_score: float,
        relevance_score: float
    ) -> bool:
        """Insert research source into vector database."""
        try:
            collection = Collection(self.collections["research_sources"])
            
            # Create embedding from title + summary
            text_to_embed = f"{title} {summary}"
            embedding = self.encode_text(text_to_embed)
            
            data = [{
                "id": source_id,
                "pipeline_run_id": pipeline_run_id,
                "url": url,
                "title": title,
                "domain": domain,
                "source_type": source_type,
                "summary": summary,
                "credibility_score": credibility_score,
                "relevance_score": relevance_score,
                "embedding": embedding
            }]
            
            collection.insert(data)
            collection.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert research source: {e}")
            return False
    
    async def search_similar_sources(
        self,
        query_text: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar research sources."""
        try:
            collection = Collection(self.collections["research_sources"])
            collection.load()
            
            query_embedding = self.encode_text(query_text)
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Build expression for filtering
            expression = None
            if domain:
                expression = f'domain == "{domain}"'
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expression,
                output_fields=["id", "url", "title", "domain", "summary", "credibility_score"]
            )
            
            similar_sources = []
            for hit in results[0]:
                similar_sources.append({
                    "id": hit.entity.get("id"),
                    "url": hit.entity.get("url"),
                    "title": hit.entity.get("title"),
                    "domain": hit.entity.get("domain"),
                    "summary": hit.entity.get("summary"),
                    "credibility_score": hit.entity.get("credibility_score"),
                    "similarity_score": hit.score
                })
            
            return similar_sources
            
        except Exception as e:
            logger.error(f"Failed to search similar sources: {e}")
            return []
    
    async def insert_content_draft(
        self,
        draft_id: str,
        pipeline_run_id: str,
        topic_id: str,
        title: str,
        content: str,
        stage: str,
        word_count: int,
        quality_score: Optional[float] = None
    ) -> bool:
        """Insert content draft into vector database."""
        try:
            collection = Collection(self.collections["content_drafts"])
            
            # Create embedding from title + content (truncated for performance)
            content_sample = content[:1000]  # First 1000 chars
            text_to_embed = f"{title} {content_sample}"
            embedding = self.encode_text(text_to_embed)
            
            data = [{
                "id": draft_id,
                "pipeline_run_id": pipeline_run_id,
                "topic_id": topic_id,
                "title": title,
                "stage": stage,
                "word_count": word_count,
                "final_quality_score": quality_score or 0.0,
                "embedding": embedding
            }]
            
            collection.insert(data)
            collection.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert content draft: {e}")
            return False
    
    async def find_similar_content(
        self,
        content_text: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar published content for plagiarism checking."""
        try:
            collection = Collection(self.collections["published_content"])
            collection.load()
            
            query_embedding = self.encode_text(content_text)
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["id", "title", "platform", "published_url"]
            )
            
            similar_content = []
            for hit in results[0]:
                # Only consider high similarity as potential plagiarism
                if hit.score > 0.85:  # Threshold for similarity concern
                    similar_content.append({
                        "id": hit.entity.get("id"),
                        "title": hit.entity.get("title"),
                        "platform": hit.entity.get("platform"),
                        "published_url": hit.entity.get("published_url"),
                        "similarity_score": hit.score
                    })
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Failed to find similar content: {e}")
            return []
    
    async def get_domain_knowledge(
        self,
        domain: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant domain knowledge for content creation."""
        try:
            collection = Collection(self.collections["domain_knowledge"])
            collection.load()
            
            query_embedding = self.encode_text(query)
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            expression = f'domain == "{domain}"'
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expression,
                output_fields=["concept", "description", "source_url", "confidence_score"]
            )
            
            knowledge_items = []
            for hit in results[0]:
                knowledge_items.append({
                    "concept": hit.entity.get("concept"),
                    "description": hit.entity.get("description"),
                    "source_url": hit.entity.get("source_url"),
                    "confidence_score": hit.entity.get("confidence_score"),
                    "relevance_score": hit.score
                })
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Failed to get domain knowledge: {e}")
            return []
    
    def close_connection(self):
        """Close connection to Milvus."""
        try:
            connections.disconnect(self.connection_alias)
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")

# Global instance
vector_service = VectorService()