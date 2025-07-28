from sentence_transformers import SentenceTransformer
from typing import Union, List, Dict, Optional
import numpy as np
import os
import logging
import json
from sklearn.metrics.pairwise import cosine_similarity
from src.multilingual_support import multilingual_support
from src.smart_cache import smart_cache
from src.performance_monitor import performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaEmbedder:
    """
    Enhanced embedder with persona-aware embedding and advanced similarity metrics.
    """

    def __init__(self, model_path: str = "models/bge-small-en-v1.5", cache_embeddings: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found at: {model_path}")
        
        self.model_path = model_path
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        
        try:
            self.model = SentenceTransformer(model_path)
            logger.info(f"Successfully loaded embedding model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(self, text: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """
        Enhanced text embedding with caching and error handling.

        Args:
            text: str or list of str
            use_cache: Whether to use embedding cache

        Returns:
            np.ndarray of embeddings
        """
        # Start performance monitoring
        timer = performance_monitor.start_stage_timer("embedding_generation")
        
        if isinstance(text, str):
            text = [text]

        # Check smart cache first
        if use_cache:
            cache_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, txt in enumerate(text):
                cache_key = smart_cache.get_cache_key(txt)
                cached_embedding = smart_cache.get_cached_embedding(cache_key)
                
                if cached_embedding is not None:
                    embedding, metadata = cached_embedding
                    cache_results.append((i, embedding))
                    performance_monitor.record_metric("embedding_cache_hit", 1, "count", "performance")
                else:
                    uncached_texts.append(txt)
                    uncached_indices.append(i)
                    performance_monitor.record_metric("embedding_cache_miss", 1, "count", "performance")
            
            # If all embeddings were cached
            if not uncached_texts:
                embeddings = np.zeros((len(text), self.model.get_sentence_embedding_dimension()))
                for idx, embedding in cache_results:
                    embeddings[idx] = embedding
                performance_monitor.end_stage_timer(timer)
                return embeddings
        # Embed uncached texts if any
        if uncached_texts:
            try:
                new_embeddings = self.model.encode(
                    uncached_texts, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Cache new embeddings
                for j, txt in enumerate(uncached_texts):
                    cache_key = smart_cache.get_cache_key(txt)
                    smart_cache.cache_embedding(
                        cache_key, 
                        new_embeddings[j], 
                        {'text_length': len(txt), 'model': self.model_path}
                    )
                    cache_results.append((uncached_indices[j], new_embeddings[j]))
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                performance_monitor.end_stage_timer(timer)
                raise

        # Reconstruct embeddings in original order
        all_results = cache_results
        all_results.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in all_results])
        
        # Record performance metrics
        performance_monitor.record_metric("embeddings_generated", len(uncached_texts), "count", "processing")
        performance_monitor.end_stage_timer(timer)

        return embeddings[0] if len(embeddings) == 1 else embeddings

    def embed_persona_job(self, persona: str, job: str) -> np.ndarray:
        """
        Enhanced persona-job embedding with contextual understanding.

        Args:
            persona: e.g. "Travel Planner"
            job: e.g. "Plan a 4-day trip for 10 college friends."

        Returns:
            Contextualized embedding optimized for the specific task
        """
        # Extract key context from the job description
        context_info = self._extract_job_context(job)
        
        # Create enhanced prompt with context
        enhanced_prompt = self._create_enhanced_prompt(persona, job, context_info)
        
        return self.embed_text(enhanced_prompt)

    def _extract_job_context(self, job: str) -> Dict[str, any]:
        """
        Extract contextual information from job description.
        """
        job_lower = job.lower()
        context = {
            'duration': None,
            'group_size': None,
            'group_type': None,
            'interests': [],
            'age_group': None
        }

        # Extract duration
        import re
        duration_patterns = [
            r'(\d+)\s*day', r'(\d+)\s*week', r'(\d+)\s*month',
            r'(\d+)-day', r'(\d+)-week', r'(\d+)-month'
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, job_lower)
            if match:
                context['duration'] = int(match.group(1))
                break

        # Extract group size
        size_patterns = [r'(\d+)\s*people', r'group\s*of\s*(\d+)', r'(\d+)\s*friends', r'(\d+)\s*family']
        for pattern in size_patterns:
            match = re.search(pattern, job_lower)
            if match:
                context['group_size'] = int(match.group(1))
                break

        # Identify group type and age
        if 'college' in job_lower or 'student' in job_lower:
            context['group_type'] = 'college'
            context['age_group'] = 'young_adult'
        elif 'family' in job_lower or 'children' in job_lower:
            context['group_type'] = 'family'
            context['age_group'] = 'mixed'
        elif 'couple' in job_lower:
            context['group_type'] = 'couple'
            context['age_group'] = 'adult'

        # Extract interests/activities
        interest_keywords = {
            'adventure': ['adventure', 'hiking', 'outdoor', 'sports'],
            'culture': ['culture', 'museum', 'history', 'art'],
            'nightlife': ['nightlife', 'party', 'club', 'bar'],
            'food': ['food', 'dining', 'cuisine', 'restaurant'],
            'relaxation': ['relax', 'spa', 'beach', 'peaceful'],
            'nature': ['nature', 'park', 'wildlife', 'scenic']
        }

        for interest, keywords in interest_keywords.items():
            if any(keyword in job_lower for keyword in keywords):
                context['interests'].append(interest)

        return context

    def _create_enhanced_prompt(self, persona: str, job: str, context: Dict[str, any]) -> str:
        """
        Create contextually enhanced prompt for better embedding with dynamic adaptation.
        """
        base_prompt = f"Persona: {persona}\nTask: {job}"
        
        # Add context-specific enhancements dynamically
        enhancements = []
        
        # Dynamic group type detection and enhancement
        job_lower = job.lower()
        
        # Use multilingual support for enhanced detection
        lang_info = multilingual_support.get_language_info(job)
        
        # Enhanced multilingual group detection
        if multilingual_support.calculate_multilingual_indicator_score(job, 'youth_indicators') > 0.15:
            enhancements.append("Focus on: social experiences, group activities, energetic attractions, diverse options")
        elif multilingual_support.calculate_multilingual_indicator_score(job, 'family_indicators') > 0.15:
            enhancements.append("Focus on: family-friendly activities, safe environments, educational experiences")
        elif any(word in job_lower for word in ['couple', 'romantic', 'honeymoon']):
            enhancements.append("Focus on: intimate experiences, romantic settings, couple activities")
        elif multilingual_support.calculate_multilingual_indicator_score(job, 'business_indicators') > 0.15:
            enhancements.append("Focus on: professional accommodations, convenient transportation, business amenities")

        # Duration-based enhancements
        if context['duration']:
            if context['duration'] <= 3:
                enhancements.append("Prioritize: must-see attractions, efficient itinerary, highlights")
            elif context['duration'] >= 7:
                enhancements.append("Include: diverse experiences, hidden gems, local culture, comprehensive coverage")

        # Enhanced multilingual budget and luxury considerations
        if multilingual_support.calculate_multilingual_indicator_score(job, 'budget_indicators') > 0.1:
            enhancements.append("Consider: budget-friendly options, value experiences, cost-effective choices")
        elif multilingual_support.calculate_multilingual_indicator_score(job, 'luxury_indicators') > 0.1:
            enhancements.append("Consider: premium experiences, luxury accommodations, exclusive options")

        # Interest extraction from job description
        if context['interests']:
            interest_text = ", ".join(context['interests'])
            enhancements.append(f"Interests: {interest_text}")

        if enhancements:
            enhanced_prompt = base_prompt + "\n" + " | ".join(enhancements)
        else:
            enhanced_prompt = base_prompt

        return enhanced_prompt

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Enhanced cosine similarity with error handling.
        """
        try:
            # Handle potential shape issues
            if vec1.ndim == 1:
                vec1 = vec1.reshape(1, -1)
            if vec2.ndim == 1:
                vec2 = vec2.reshape(1, -1)
                
            similarity = cosine_similarity(vec1, vec2)[0, 0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Alias for cosine_similarity with additional validation.
        """
        return self.cosine_similarity(vec1, vec2)

    def contextual_similarity(self, text: str, persona_job_embedding: np.ndarray, 
                            content_type: str = None, boost_factors: Dict[str, float] = None) -> float:
        """
        Compute contextual similarity with content-type aware boosting.
        
        Args:
            text: Text to compare
            persona_job_embedding: Reference embedding
            content_type: Type of content (activities, dining, etc.)
            boost_factors: Optional boosting factors for different content types
            
        Returns:
            Enhanced similarity score
        """
        text_embedding = self.embed_text(text)
        base_similarity = self.similarity(text_embedding, persona_job_embedding)
        
        # Apply content-type boosting if available
        if content_type and boost_factors and content_type in boost_factors:
            boosted_similarity = base_similarity * boost_factors[content_type]
            return min(boosted_similarity, 1.0)  # Cap at 1.0
        
        return base_similarity

    def get_embedding_stats(self) -> Dict[str, any]:
        """
        Get statistics about the embedding cache and model.
        """
        return {
            'model_path': self.model_path,
            'cache_size': len(self.embedding_cache),
            'cache_enabled': self.cache_embeddings,
            'embedding_dimension': self.model.get_sentence_embedding_dimension()
        }
