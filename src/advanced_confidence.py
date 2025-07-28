#!/usr/bin/env python3
"""
Enhanced confidence scoring system with multiple confidence metrics.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring with multiple metrics and quality assessment.
    """
    
    def __init__(self):
        self.confidence_weights = {
            'semantic_similarity': 0.4,
            'keyword_relevance': 0.2,
            'content_quality': 0.15,
            'persona_alignment': 0.15,
            'document_authority': 0.1
        }
        
    def calculate_comprehensive_confidence(self, 
                                         section_text: str,
                                         persona_embedding: np.ndarray,
                                         section_embedding: np.ndarray,
                                         job_context: Dict,
                                         boost_factors: Dict,
                                         document_stats: Dict) -> Dict:
        """
        Calculate comprehensive confidence score with multiple metrics.
        """
        
        # 1. Semantic Similarity (existing)
        semantic_score = self._calculate_semantic_similarity(section_embedding, persona_embedding)
        
        # 2. Keyword Relevance Score
        keyword_score = self._calculate_keyword_relevance(section_text, job_context)
        
        # 3. Content Quality Score
        quality_score = self._assess_content_quality(section_text)
        
        # 4. Persona Alignment Score
        alignment_score = self._calculate_persona_alignment(section_text, job_context, boost_factors)
        
        # 5. Document Authority Score
        authority_score = self._calculate_document_authority(document_stats)
        
        # Weighted composite score
        composite_confidence = (
            semantic_score * self.confidence_weights['semantic_similarity'] +
            keyword_score * self.confidence_weights['keyword_relevance'] +
            quality_score * self.confidence_weights['content_quality'] +
            alignment_score * self.confidence_weights['persona_alignment'] +
            authority_score * self.confidence_weights['document_authority']
        )
        
        return {
            'composite_confidence': composite_confidence,
            'semantic_similarity': semantic_score,
            'keyword_relevance': keyword_score,
            'content_quality': quality_score,
            'persona_alignment': alignment_score,
            'document_authority': authority_score,
            'confidence_breakdown': {
                'primary_factors': ['semantic_similarity', 'keyword_relevance'],
                'quality_factors': ['content_quality', 'persona_alignment'],
                'authority_factors': ['document_authority']
            }
        }
    
    def _calculate_semantic_similarity(self, section_embedding: np.ndarray, persona_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        if section_embedding is None or persona_embedding is None:
            return 0.0
            
        cosine_sim = np.dot(section_embedding, persona_embedding) / (
            np.linalg.norm(section_embedding) * np.linalg.norm(persona_embedding)
        )
        return max(0.0, min(1.0, cosine_sim))
    
    def _calculate_keyword_relevance(self, text: str, job_context: Dict) -> float:
        """Calculate relevance based on job-specific keywords."""
        if not job_context or not text:
            return 0.0
            
        text_lower = text.lower()
        relevant_keywords = job_context.get('keywords', [])
        
        if not relevant_keywords:
            return 0.5  # Neutral score if no keywords
            
        matches = sum(1 for keyword in relevant_keywords if keyword.lower() in text_lower)
        return min(1.0, matches / len(relevant_keywords))
    
    def _assess_content_quality(self, text: str) -> float:
        """Assess the quality and informativeness of content."""
        if not text:
            return 0.0
            
        # Quality indicators
        quality_score = 0.0
        
        # Length appropriateness (not too short, not too long)
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            quality_score += 0.3
        elif 30 <= word_count <= 500:
            quality_score += 0.2
        else:
            quality_score += 0.1
            
        # Information density (proper nouns, specific details)
        proper_nouns = sum(1 for word in text.split() if word[0].isupper() and len(word) > 2)
        if proper_nouns >= 3:
            quality_score += 0.2
            
        # Numerical information (addresses, prices, times)
        import re
        numbers = len(re.findall(r'\d+', text))
        if numbers >= 2:
            quality_score += 0.2
            
        # Specific travel terms
        travel_terms = ['hotel', 'restaurant', 'museum', 'beach', 'location', 'address', 'phone', 'hours']
        term_matches = sum(1 for term in travel_terms if term in text.lower())
        if term_matches >= 2:
            quality_score += 0.3
            
        return min(1.0, quality_score)
    
    def _calculate_persona_alignment(self, text: str, job_context: Dict, boost_factors: Dict) -> float:
        """Calculate how well content aligns with persona needs."""
        if not job_context:
            return 0.5
            
        alignment_score = 0.0
        text_lower = text.lower()
        
        # Check for persona-specific content
        persona_indicators = job_context.get('persona_type', '')
        
        if 'student' in persona_indicators or 'college' in persona_indicators:
            student_terms = ['budget', 'cheap', 'affordable', 'young', 'group', 'party']
            matches = sum(1 for term in student_terms if term in text_lower)
            alignment_score += min(0.4, matches * 0.1)
            
        if 'family' in persona_indicators:
            family_terms = ['family', 'children', 'kid', 'safe', 'suitable']
            matches = sum(1 for term in family_terms if term in text_lower)
            alignment_score += min(0.4, matches * 0.1)
            
        if 'luxury' in persona_indicators:
            luxury_terms = ['luxury', 'premium', 'exclusive', 'high-end', 'deluxe']
            matches = sum(1 for term in luxury_terms if term in text_lower)
            alignment_score += min(0.4, matches * 0.1)
            
        # Boost factor contribution
        max_boost = max(boost_factors.values()) if boost_factors else 1.0
        if max_boost > 1.2:
            alignment_score += 0.3
        elif max_boost > 1.1:
            alignment_score += 0.2
            
        return min(1.0, alignment_score + 0.3)  # Base alignment score
    
    def _calculate_document_authority(self, document_stats: Dict) -> float:
        """Calculate document authority based on various factors."""
        if not document_stats:
            return 0.5
            
        authority_score = 0.5  # Base score
        
        # Document completeness
        if document_stats.get('word_count', 0) > 100:
            authority_score += 0.2
            
        # Section structure
        if document_stats.get('has_proper_sections', False):
            authority_score += 0.2
            
        # Content freshness (if available)
        if document_stats.get('is_recent', True):
            authority_score += 0.1
            
        return min(1.0, authority_score)

# Global instance
advanced_confidence_scorer = AdvancedConfidenceScorer()
