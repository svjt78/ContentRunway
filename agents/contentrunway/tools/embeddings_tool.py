"""Embeddings tool for semantic similarity, topic clustering, and vector operations."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import pickle
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingsTool:
    """Tool for generating embeddings and performing semantic analysis operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embeddings tool with sentence transformer model.
        
        Args:
            model_name: Sentence transformer model to use for embeddings
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.model = None
            self.model_name = None
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        
        # Clustering parameters
        self.clustering_methods = {
            'kmeans': KMeans,
            'dbscan': DBSCAN
        }
        
        # Dimensionality reduction methods
        self.reduction_methods = {
            'pca': PCA,
            'tsne': TSNE,
            'umap': umap.UMAP
        }
    
    def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            use_cache: Whether to use cached embeddings
            normalize: Whether to normalize embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if not self.model:
            logger.error("Sentence transformer model not available")
            return {
                'embeddings': None,
                'error': 'Model not available'
            }
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings_list = []
            cache_hits = 0
            
            for i, text in enumerate(texts):
                # Create cache key
                cache_key = hashlib.md5(f"{text}_{self.model_name}".encode()).hexdigest()
                
                if use_cache and cache_key in self.embedding_cache:
                    embedding = self.embedding_cache[cache_key]
                    cache_hits += 1
                else:
                    # Generate new embedding
                    embedding = self.model.encode([text])[0]
                    
                    if use_cache:
                        self.embedding_cache[cache_key] = embedding
                
                if normalize:
                    # L2 normalization
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
                embeddings_list.append(embedding)
            
            embeddings_array = np.array(embeddings_list)
            
            return {
                'embeddings': embeddings_array,
                'embedding_dimension': embeddings_array.shape[1],
                'num_embeddings': len(embeddings_list),
                'cache_hits': cache_hits,
                'cache_miss': len(texts) - cache_hits,
                'model_used': self.model_name,
                'normalized': normalize,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                'embeddings': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
        similarity_metric: str = 'cosine'
    ) -> Dict[str, Any]:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            similarity_metric: Similarity metric to use ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Dictionary with similarity results
        """
        logger.info("Calculating semantic similarity between texts")
        
        try:
            # Generate embeddings
            embedding_result = self.generate_embeddings([text1, text2])
            
            if embedding_result['embeddings'] is None:
                return {
                    'similarity_score': 0.0,
                    'error': embedding_result.get('error', 'Unknown error')
                }
            
            embeddings = embedding_result['embeddings']
            
            # Calculate similarity
            if similarity_metric == 'cosine':
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            elif similarity_metric == 'euclidean':
                similarity = 1 / (1 + np.linalg.norm(embeddings[0] - embeddings[1]))
            elif similarity_metric == 'manhattan':
                similarity = 1 / (1 + np.sum(np.abs(embeddings[0] - embeddings[1])))
            else:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Categorize similarity level
            similarity_level = self._categorize_similarity_level(similarity)
            
            return {
                'similarity_score': round(float(similarity), 4),
                'similarity_level': similarity_level,
                'similarity_metric': similarity_metric,
                'text1_length': len(text1.split()),
                'text2_length': len(text2.split()),
                'embedding_dimension': embeddings.shape[1],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {
                'similarity_score': 0.0,
                'error': str(e)
            }
    
    def _categorize_similarity_level(self, similarity: float) -> str:
        """Categorize similarity score into readable levels."""
        
        if similarity >= 0.9:
            return 'very_high'
        elif similarity >= 0.8:
            return 'high'
        elif similarity >= 0.6:
            return 'moderate'
        elif similarity >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def cluster_topics(
        self,
        texts: List[str],
        topic_labels: List[str] = None,
        num_clusters: Optional[int] = None,
        clustering_method: str = 'kmeans',
        min_cluster_size: int = 2
    ) -> Dict[str, Any]:
        """
        Cluster texts into topics using semantic embeddings.
        
        Args:
            texts: List of texts to cluster
            topic_labels: Optional labels for texts
            num_clusters: Number of clusters (auto-detected if None)
            clustering_method: Clustering method ('kmeans', 'dbscan')
            min_cluster_size: Minimum cluster size for DBSCAN
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering {len(texts)} texts using {clustering_method}")
        
        if len(texts) < 2:
            return {
                'clusters': [],
                'error': 'Need at least 2 texts for clustering'
            }
        
        try:
            # Generate embeddings
            embedding_result = self.generate_embeddings(texts)
            
            if embedding_result['embeddings'] is None:
                return {
                    'clusters': [],
                    'error': embedding_result.get('error', 'Embedding generation failed')
                }
            
            embeddings = embedding_result['embeddings']
            
            # Auto-detect number of clusters if not specified
            if num_clusters is None and clustering_method == 'kmeans':
                num_clusters = self._estimate_optimal_clusters(embeddings)
            
            # Perform clustering
            if clustering_method == 'kmeans':
                clusterer = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings)
                cluster_centers = clusterer.cluster_centers_
            elif clustering_method == 'dbscan':
                clusterer = DBSCAN(eps=0.3, min_samples=min_cluster_size)
                cluster_labels = clusterer.fit_predict(embeddings)
                cluster_centers = None
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # Organize results
            clusters = self._organize_clustering_results(
                texts,
                topic_labels or [f"Text_{i}" for i in range(len(texts))],
                cluster_labels,
                embeddings,
                cluster_centers
            )
            
            # Calculate clustering quality metrics
            quality_metrics = self._calculate_clustering_quality(embeddings, cluster_labels)
            
            return {
                'clusters': clusters,
                'num_clusters': num_clusters,
                'clustering_method': clustering_method,
                'quality_metrics': quality_metrics,
                'total_texts': len(texts),
                'noise_points': sum(1 for label in cluster_labels if label == -1),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Topic clustering failed: {e}")
            return {
                'clusters': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _estimate_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method."""
        
        n_samples = len(embeddings)
        
        # Test different numbers of clusters
        max_clusters = min(10, n_samples // 2)  # Reasonable upper bound
        
        if max_clusters < 2:
            return 2
        
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            except Exception as e:
                logger.warning(f"Failed to test {k} clusters: {e}")
                break
        
        if len(inertias) < 2:
            return 3  # Default fallback
        
        # Find elbow point (simple method)
        # Calculate rate of change
        changes = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        
        # Find where improvement starts to diminish
        optimal_k = 2
        for i in range(1, len(changes)):
            if changes[i] < changes[i-1] * 0.7:  # Significant decrease in improvement
                optimal_k = i + 2  # +2 because we started from k=2
                break
        
        return min(optimal_k, max_clusters)
    
    def _organize_clustering_results(
        self,
        texts: List[str],
        labels: List[str],
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        cluster_centers: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Organize clustering results into structured format."""
        
        clusters = {}
        
        # Group texts by cluster
        for i, (text, label, cluster_id) in enumerate(zip(texts, labels, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'cluster_id': int(cluster_id),
                    'texts': [],
                    'labels': [],
                    'embeddings': [],
                    'cluster_size': 0
                }
            
            clusters[cluster_id]['texts'].append(text)
            clusters[cluster_id]['labels'].append(label)
            clusters[cluster_id]['embeddings'].append(embeddings[i])
            clusters[cluster_id]['cluster_size'] += 1
        
        # Process each cluster
        organized_clusters = []
        
        for cluster_id, cluster_data in clusters.items():
            if cluster_id == -1:  # Noise points in DBSCAN
                cluster_name = "Unclustered"
                cluster_theme = "No clear theme identified"
            else:
                # Generate cluster summary
                cluster_texts = cluster_data['texts']
                cluster_name = f"Cluster {cluster_id + 1}"
                cluster_theme = self._identify_cluster_theme(cluster_texts)
            
            # Calculate intra-cluster similarity
            cluster_embeddings = np.array(cluster_data['embeddings'])
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            else:
                avg_similarity = 1.0
            
            # Find representative text (closest to centroid)
            if len(cluster_embeddings) > 1:
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = [cosine_similarity([embedding], [centroid])[0][0] for embedding in cluster_embeddings]
                representative_idx = np.argmax(distances)
                representative_text = cluster_texts[representative_idx]
            else:
                representative_text = cluster_texts[0] if cluster_texts else ""
            
            organized_clusters.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'cluster_theme': cluster_theme,
                'cluster_size': cluster_data['cluster_size'],
                'texts': cluster_data['texts'],
                'labels': cluster_data['labels'],
                'representative_text': representative_text[:200] + "..." if len(representative_text) > 200 else representative_text,
                'avg_intra_similarity': round(float(avg_similarity), 4),
                'cohesion_score': round(float(avg_similarity), 3)
            })
        
        # Sort clusters by size (largest first)
        organized_clusters.sort(key=lambda x: x['cluster_size'], reverse=True)
        
        return organized_clusters
    
    def _identify_cluster_theme(self, texts: List[str]) -> str:
        """Identify the main theme of a cluster based on its texts."""
        
        # Combine all texts in cluster
        combined_text = ' '.join(texts).lower()
        
        # Extract key terms using simple frequency analysis
        words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text)  # Words with 4+ letters
        
        # Remove common stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'their',
            'which', 'would', 'could', 'should', 'about', 'there', 'where',
            'when', 'what', 'will', 'can', 'all', 'some', 'more', 'other',
            'like', 'just', 'also', 'than', 'only', 'very', 'well', 'much'
        }
        
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Get most frequent words
        from collections import Counter
        word_counts = Counter(meaningful_words)
        top_words = [word for word, count in word_counts.most_common(5) if count > 1]
        
        # Create theme description
        if len(top_words) >= 2:
            theme = f"Topics related to {', '.join(top_words[:3])}"
        elif len(top_words) == 1:
            theme = f"Content focused on {top_words[0]}"
        else:
            theme = "Mixed topics"
        
        return theme
    
    def _calculate_clustering_quality(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering quality metrics."""
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # Remove noise points for quality calculation
            non_noise_mask = cluster_labels != -1
            clean_embeddings = embeddings[non_noise_mask]
            clean_labels = cluster_labels[non_noise_mask]
            
            if len(set(clean_labels)) < 2:
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'quality_assessment': 'insufficient_clusters'
                }
            
            # Calculate silhouette score
            silhouette = silhouette_score(clean_embeddings, clean_labels)
            
            # Calculate Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(clean_embeddings, clean_labels)
            
            # Assess overall quality
            if silhouette >= 0.5:
                quality_assessment = 'excellent'
            elif silhouette >= 0.3:
                quality_assessment = 'good'
            elif silhouette >= 0.1:
                quality_assessment = 'fair'
            else:
                quality_assessment = 'poor'
            
            return {
                'silhouette_score': round(float(silhouette), 4),
                'calinski_harabasz_score': round(float(calinski_harabasz), 2),
                'quality_assessment': quality_assessment,
                'num_clusters_evaluated': len(set(clean_labels))
            }
            
        except Exception as e:
            logger.warning(f"Clustering quality calculation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'quality_assessment': 'calculation_failed',
                'error': str(e)
            }
    
    def find_similar_content(
        self,
        query_text: str,
        candidate_texts: List[str],
        candidate_labels: List[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Find most similar content to a query text.
        
        Args:
            query_text: Text to find similarities for
            candidate_texts: List of candidate texts to search
            candidate_labels: Optional labels for candidate texts
            top_k: Number of top similar texts to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with similar content results
        """
        logger.info(f"Finding similar content from {len(candidate_texts)} candidates")
        
        if not candidate_texts:
            return {
                'similar_texts': [],
                'total_candidates': 0
            }
        
        try:
            # Generate embeddings for query and candidates
            all_texts = [query_text] + candidate_texts
            embedding_result = self.generate_embeddings(all_texts)
            
            if embedding_result['embeddings'] is None:
                return {
                    'similar_texts': [],
                    'error': embedding_result.get('error', 'Embedding generation failed')
                }
            
            embeddings = embedding_result['embeddings']
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # Create similarity results
            similarity_results = []
            for i, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    result = {
                        'text': candidate_texts[i],
                        'label': candidate_labels[i] if candidate_labels else f"Candidate_{i}",
                        'similarity_score': round(float(similarity), 4),
                        'similarity_level': self._categorize_similarity_level(similarity),
                        'text_length': len(candidate_texts[i].split()),
                        'index': i
                    }
                    similarity_results.append(result)
            
            # Sort by similarity score (descending)
            similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Get top-k results
            top_similar = similarity_results[:top_k]
            
            return {
                'similar_texts': top_similar,
                'total_candidates': len(candidate_texts),
                'candidates_above_threshold': len(similarity_results),
                'highest_similarity': similarity_results[0]['similarity_score'] if similarity_results else 0.0,
                'average_similarity': round(float(np.mean(similarities)), 4),
                'similarity_threshold': similarity_threshold,
                'query_text_length': len(query_text.split()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            return {
                'similar_texts': [],
                'error': str(e)
            }
    
    def reduce_dimensionality(
        self,
        embeddings: np.ndarray,
        method: str = 'umap',
        target_dimensions: int = 2,
        preserve_distances: bool = True
    ) -> Dict[str, Any]:
        """
        Reduce dimensionality of embeddings for visualization or analysis.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ('pca', 'tsne', 'umap')
            target_dimensions: Target number of dimensions
            preserve_distances: Whether to preserve distance relationships
            
        Returns:
            Dictionary with reduced embeddings and metadata
        """
        logger.info(f"Reducing embeddings dimensionality using {method}")
        
        if method not in self.reduction_methods:
            return {
                'reduced_embeddings': None,
                'error': f"Unsupported reduction method: {method}"
            }
        
        try:
            # Configure reduction method
            if method == 'pca':
                reducer = PCA(n_components=target_dimensions, random_state=42)
            elif method == 'tsne':
                # t-SNE parameters
                perplexity = min(30, max(5, len(embeddings) // 3))
                reducer = TSNE(
                    n_components=target_dimensions,
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=1000
                )
            elif method == 'umap':
                # UMAP parameters
                n_neighbors = min(15, max(5, len(embeddings) // 4))
                reducer = umap.UMAP(
                    n_components=target_dimensions,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=42
                )
            
            # Perform dimensionality reduction
            reduced_embeddings = reducer.fit_transform(embeddings)
            
            # Calculate preservation metrics
            preservation_metrics = self._calculate_preservation_metrics(
                embeddings,
                reduced_embeddings,
                method
            )
            
            return {
                'reduced_embeddings': reduced_embeddings,
                'original_dimensions': embeddings.shape[1],
                'target_dimensions': target_dimensions,
                'actual_dimensions': reduced_embeddings.shape[1],
                'reduction_method': method,
                'preservation_metrics': preservation_metrics,
                'num_points': len(reduced_embeddings),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return {
                'reduced_embeddings': None,
                'error': str(e)
            }
    
    def _calculate_preservation_metrics(
        self,
        original_embeddings: np.ndarray,
        reduced_embeddings: np.ndarray,
        method: str
    ) -> Dict[str, Any]:
        """Calculate how well the reduction preserves original relationships."""
        
        try:
            # Calculate pairwise distances in both spaces
            original_distances = cosine_similarity(original_embeddings)
            reduced_distances = cosine_similarity(reduced_embeddings)
            
            # Calculate correlation between distance matrices
            from scipy.stats import pearsonr
            
            # Flatten upper triangular matrices (avoid diagonal)
            n = len(original_distances)
            triu_indices = np.triu_indices(n, k=1)
            
            original_flat = original_distances[triu_indices]
            reduced_flat = reduced_distances[triu_indices]
            
            correlation, p_value = pearsonr(original_flat, reduced_flat)
            
            # Calculate stress (for methods like t-SNE)
            stress = np.sum((original_flat - reduced_flat) ** 2)
            normalized_stress = stress / np.sum(original_flat ** 2)
            
            return {
                'distance_correlation': round(float(correlation), 4),
                'correlation_p_value': round(float(p_value), 6),
                'stress': round(float(normalized_stress), 4),
                'preservation_quality': 'excellent' if correlation > 0.8 else 'good' if correlation > 0.6 else 'fair' if correlation > 0.4 else 'poor'
            }
            
        except Exception as e:
            logger.warning(f"Preservation metrics calculation failed: {e}")
            return {
                'distance_correlation': 0.0,
                'preservation_quality': 'unknown',
                'error': str(e)
            }
    
    def analyze_topic_coherence(
        self,
        topics: List[Dict[str, Any]],
        topic_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze coherence and quality of topics.
        
        Args:
            topics: List of topic dictionaries
            topic_embeddings: Optional pre-computed embeddings for topics
            
        Returns:
            Dictionary with coherence analysis
        """
        logger.info(f"Analyzing coherence for {len(topics)} topics")
        
        try:
            # Extract text content from topics
            topic_texts = []
            for topic in topics:
                if isinstance(topic, dict):
                    # Combine title and description
                    text = ""
                    if 'title' in topic:
                        text += topic['title'] + ". "
                    if 'description' in topic:
                        text += topic['description']
                    elif 'summary' in topic:
                        text += topic['summary']
                    topic_texts.append(text.strip())
                else:
                    topic_texts.append(str(topic))
            
            # Generate embeddings if not provided
            if topic_embeddings is None:
                embedding_result = self.generate_embeddings(topic_texts)
                if embedding_result['embeddings'] is None:
                    return {
                        'coherence_score': 0.5,
                        'error': 'Failed to generate topic embeddings'
                    }
                topic_embeddings = embedding_result['embeddings']
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(topic_embeddings)
            
            # Calculate coherence metrics
            coherence_metrics = self._calculate_coherence_metrics(similarities, topic_texts)
            
            # Identify topic relationships
            topic_relationships = self._identify_topic_relationships(similarities, topics)
            
            # Detect potential topic overlaps
            overlaps = self._detect_topic_overlaps(similarities, topics, threshold=0.8)
            
            return {
                'coherence_score': coherence_metrics['overall_coherence'],
                'coherence_metrics': coherence_metrics,
                'topic_relationships': topic_relationships,
                'potential_overlaps': overlaps,
                'topics_analyzed': len(topics),
                'avg_topic_similarity': round(float(np.mean(similarities[np.triu_indices_from(similarities, k=1)])), 4),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Topic coherence analysis failed: {e}")
            return {
                'coherence_score': 0.5,
                'error': str(e)
            }
    
    def _calculate_coherence_metrics(self, similarities: np.ndarray, topic_texts: List[str]) -> Dict[str, Any]:
        """Calculate various coherence metrics for topics."""
        
        n_topics = len(similarities)
        
        if n_topics < 2:
            return {
                'overall_coherence': 1.0,
                'avg_pairwise_similarity': 1.0,
                'topic_diversity': 0.0
            }
        
        # Calculate average pairwise similarity (excluding diagonal)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        avg_similarity = np.mean(upper_triangle)
        
        # Calculate topic diversity (inverse of average similarity)
        topic_diversity = 1.0 - avg_similarity
        
        # Calculate coherence based on similarity distribution
        similarity_std = np.std(upper_triangle)
        
        # Good coherence means moderate similarity (not too high, not too low)
        # Optimal range: 0.3-0.7 similarity
        optimal_range_count = sum(1 for sim in upper_triangle if 0.3 <= sim <= 0.7)
        coherence_score = optimal_range_count / len(upper_triangle)
        
        # Adjust for diversity
        if topic_diversity < 0.2:  # Too similar
            coherence_score *= 0.8
        elif topic_diversity > 0.8:  # Too diverse
            coherence_score *= 0.9
        
        return {
            'overall_coherence': round(float(coherence_score), 4),
            'avg_pairwise_similarity': round(float(avg_similarity), 4),
            'topic_diversity': round(float(topic_diversity), 4),
            'similarity_std_dev': round(float(similarity_std), 4),
            'optimal_similarity_pairs': optimal_range_count,
            'total_pairs': len(upper_triangle)
        }
    
    def _identify_topic_relationships(
        self,
        similarities: np.ndarray,
        topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify relationships between topics based on similarity."""
        
        relationships = []
        n_topics = len(topics)
        
        # Find high similarity pairs
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                similarity = similarities[i][j]
                
                if similarity >= 0.7:  # High similarity threshold
                    relationship = {
                        'topic1_index': i,
                        'topic2_index': j,
                        'topic1_title': topics[i].get('title', f'Topic {i}') if isinstance(topics[i], dict) else str(topics[i])[:50],
                        'topic2_title': topics[j].get('title', f'Topic {j}') if isinstance(topics[j], dict) else str(topics[j])[:50],
                        'similarity_score': round(float(similarity), 4),
                        'relationship_type': 'highly_related' if similarity >= 0.8 else 'related'
                    }
                    relationships.append(relationship)
        
        # Sort by similarity (highest first)
        relationships.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return relationships[:10]  # Top 10 relationships
    
    def _detect_topic_overlaps(
        self,
        similarities: np.ndarray,
        topics: List[Dict[str, Any]],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Detect potential topic overlaps that might need consolidation."""
        
        overlaps = []
        n_topics = len(topics)
        
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                similarity = similarities[i][j]
                
                if similarity >= threshold:
                    overlap = {
                        'topic1_index': i,
                        'topic2_index': j,
                        'topic1_title': topics[i].get('title', f'Topic {i}') if isinstance(topics[i], dict) else str(topics[i])[:50],
                        'topic2_title': topics[j].get('title', f'Topic {j}') if isinstance(topics[j], dict) else str(topics[j])[:50],
                        'overlap_score': round(float(similarity), 4),
                        'recommendation': 'Consider consolidating these similar topics'
                    }
                    overlaps.append(overlap)
        
        return overlaps
    
    def create_semantic_search_index(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'content'
    ) -> Dict[str, Any]:
        """
        Create a semantic search index from documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text content
            
        Returns:
            Dictionary with search index data
        """
        logger.info(f"Creating semantic search index for {len(documents)} documents")
        
        try:
            # Extract texts
            texts = []
            doc_metadata = []
            
            for i, doc in enumerate(documents):
                if isinstance(doc, dict) and text_field in doc:
                    text = doc[text_field]
                    metadata = {k: v for k, v in doc.items() if k != text_field}
                    metadata['doc_index'] = i
                else:
                    text = str(doc)
                    metadata = {'doc_index': i}
                
                texts.append(text)
                doc_metadata.append(metadata)
            
            # Generate embeddings
            embedding_result = self.generate_embeddings(texts)
            
            if embedding_result['embeddings'] is None:
                return {
                    'index_created': False,
                    'error': embedding_result.get('error', 'Embedding generation failed')
                }
            
            embeddings = embedding_result['embeddings']
            
            # Create search index structure
            search_index = {
                'embeddings': embeddings,
                'documents': documents,
                'doc_metadata': doc_metadata,
                'text_field': text_field,
                'embedding_dimension': embeddings.shape[1],
                'index_size': len(documents),
                'model_used': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Semantic search index created: {len(documents)} documents, {embeddings.shape[1]} dimensions")
            
            return {
                'index_created': True,
                'search_index': search_index,
                'index_stats': {
                    'total_documents': len(documents),
                    'embedding_dimension': embeddings.shape[1],
                    'avg_text_length': round(np.mean([len(text.split()) for text in texts]), 1),
                    'model_used': self.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Search index creation failed: {e}")
            return {
                'index_created': False,
                'error': str(e)
            }
    
    def search_semantic_index(
        self,
        search_index: Dict[str, Any],
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Search the semantic index for relevant documents.
        
        Args:
            search_index: Previously created search index
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching semantic index for: {query[:50]}...")
        
        try:
            # Get index data
            index_embeddings = search_index['embeddings']
            documents = search_index['documents']
            doc_metadata = search_index['doc_metadata']
            
            # Generate query embedding
            query_embedding_result = self.generate_embeddings([query])
            
            if query_embedding_result['embeddings'] is None:
                return {
                    'search_results': [],
                    'error': 'Failed to generate query embedding'
                }
            
            query_embedding = query_embedding_result['embeddings'][0:1]
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, index_embeddings)[0]
            
            # Create search results
            search_results = []
            for i, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    result = {
                        'document': documents[i],
                        'metadata': doc_metadata[i],
                        'similarity_score': round(float(similarity), 4),
                        'similarity_level': self._categorize_similarity_level(similarity),
                        'doc_index': i
                    }
                    search_results.append(result)
            
            # Sort by similarity (descending)
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Get top-k results
            top_results = search_results[:top_k]
            
            return {
                'search_results': top_results,
                'total_results': len(search_results),
                'query': query,
                'top_k_requested': top_k,
                'similarity_threshold': similarity_threshold,
                'highest_similarity': top_results[0]['similarity_score'] if top_results else 0.0,
                'search_performed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                'search_results': [],
                'error': str(e)
            }
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Save embeddings to file for later use."""
        
        try:
            data_to_save = {
                'embeddings': embeddings,
                'metadata': metadata or {},
                'model_name': self.model_name,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            logger.info(f"Embeddings saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    def load_embeddings(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load embeddings from file."""
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Embeddings loaded from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None