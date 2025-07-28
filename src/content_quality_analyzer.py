#!/usr/bin/env python3
"""
Content quality analyzer for enhanced section evaluation.
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ContentQualityAnalyzer:
    """
    Analyzes content quality and provides detailed quality metrics.
    """
    
    def __init__(self):
        self.quality_indicators = {
            'specificity': {
                'proper_nouns': 0.2,
                'specific_locations': 0.2,
                'contact_info': 0.15,
                'numerical_data': 0.15
            },
            'informativeness': {
                'actionable_info': 0.3,
                'practical_details': 0.25,
                'descriptive_content': 0.2
            },
            'relevance': {
                'travel_terminology': 0.3,
                'contextual_relevance': 0.25
            }
        }
    
    def analyze_content_quality(self, text: str, content_type: str, job_context: Dict) -> Dict:
        """
        Comprehensive content quality analysis.
        """
        if not text:
            return self._empty_quality_report()
            
        analysis = {
            'overall_quality_score': 0.0,
            'specificity_score': self._analyze_specificity(text),
            'informativeness_score': self._analyze_informativeness(text, content_type),
            'relevance_score': self._analyze_relevance(text, job_context),
            'readability_score': self._analyze_readability(text),
            'actionability_score': self._analyze_actionability(text),
            'quality_breakdown': {},
            'improvement_suggestions': []
        }
        
        # Calculate overall quality score
        analysis['overall_quality_score'] = (
            analysis['specificity_score'] * 0.25 +
            analysis['informativeness_score'] * 0.3 +
            analysis['relevance_score'] * 0.25 +
            analysis['readability_score'] * 0.1 +
            analysis['actionability_score'] * 0.1
        )
        
        analysis['quality_breakdown'] = self._generate_quality_breakdown(analysis)
        analysis['improvement_suggestions'] = self._generate_improvement_suggestions(analysis, text)
        
        return analysis
    
    def _analyze_specificity(self, text: str) -> float:
        """Analyze how specific and detailed the content is."""
        specificity_score = 0.0
        
        # Proper nouns (places, names)
        proper_nouns = len([word for word in text.split() if word[0].isupper() and len(word) > 2])
        specificity_score += min(0.3, proper_nouns * 0.05)
        
        # Specific locations and addresses
        location_patterns = [
            r'\d+\s+\w+\s+Street', r'\d+\s+\w+\s+Ave', r'\d+\s+\w+\s+Road',
            r'Tel:', r'Phone:', r'Address:', r'Located at', r'Situated in'
        ]
        location_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in location_patterns)
        specificity_score += min(0.2, location_matches * 0.1)
        
        # Numerical data (prices, times, distances)
        numbers = len(re.findall(r'\d+', text))
        specificity_score += min(0.2, numbers * 0.02)
        
        # Time information
        time_patterns = [r'\d+:\d+', r'\d+\s*am', r'\d+\s*pm', r'open', r'closed', r'hours']
        time_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in time_patterns)
        specificity_score += min(0.15, time_matches * 0.05)
        
        # Contact information
        contact_patterns = [r'www\.', r'http', r'@', r'phone', r'tel', r'contact']
        contact_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in contact_patterns)
        specificity_score += min(0.15, contact_matches * 0.05)
        
        return min(1.0, specificity_score)
    
    def _analyze_informativeness(self, text: str, content_type: str) -> float:
        """Analyze how informative the content is."""
        informativeness_score = 0.0
        text_lower = text.lower()
        
        # Content type specific information
        type_keywords = {
            'accommodation': ['room', 'bed', 'bathroom', 'amenities', 'service', 'location', 'price'],
            'dining': ['menu', 'cuisine', 'dish', 'restaurant', 'food', 'taste', 'chef', 'speciality'],
            'activities': ['activity', 'tour', 'visit', 'experience', 'attraction', 'fun', 'entertainment'],
            'culture': ['history', 'museum', 'art', 'culture', 'heritage', 'tradition', 'local'],
            'transportation': ['bus', 'train', 'car', 'taxi', 'transport', 'route', 'schedule'],
            'practical': ['tip', 'advice', 'important', 'note', 'remember', 'consider', 'plan']
        }
        
        relevant_keywords = type_keywords.get(content_type, [])
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in text_lower)
        informativeness_score += min(0.4, keyword_matches * 0.05)
        
        # Descriptive adjectives
        descriptive_words = ['beautiful', 'stunning', 'amazing', 'excellent', 'perfect', 'wonderful', 
                           'charming', 'elegant', 'cozy', 'spacious', 'comfortable', 'convenient']
        descriptive_matches = sum(1 for word in descriptive_words if word in text_lower)
        informativeness_score += min(0.2, descriptive_matches * 0.03)
        
        # Practical information
        practical_words = ['located', 'available', 'offers', 'includes', 'features', 'provides']
        practical_matches = sum(1 for word in practical_words if word in text_lower)
        informativeness_score += min(0.3, practical_matches * 0.05)
        
        # Sentence structure (variety indicates better information)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 25:
            informativeness_score += 0.1
            
        return min(1.0, informativeness_score)
    
    def _analyze_relevance(self, text: str, job_context: Dict) -> float:
        """Analyze relevance to the job context."""
        if not job_context:
            return 0.5
            
        relevance_score = 0.0
        text_lower = text.lower()
        
        # Job-specific keywords
        job_keywords = job_context.get('keywords', [])
        if job_keywords:
            keyword_matches = sum(1 for keyword in job_keywords if keyword.lower() in text_lower)
            relevance_score += min(0.4, keyword_matches / len(job_keywords))
        
        # Persona alignment
        persona_type = job_context.get('persona_type', '')
        if persona_type:
            persona_relevance = self._calculate_persona_relevance(text_lower, persona_type)
            relevance_score += persona_relevance * 0.3
        
        # Context matching
        context_terms = job_context.get('context_terms', [])
        if context_terms:
            context_matches = sum(1 for term in context_terms if term.lower() in text_lower)
            relevance_score += min(0.3, context_matches * 0.1)
        
        return min(1.0, relevance_score + 0.2)  # Base relevance
    
    def _analyze_readability(self, text: str) -> float:
        """Analyze text readability."""
        if not text:
            return 0.0
            
        # Simple readability metrics
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Optimal ranges
        readability_score = 0.5  # Base score
        
        if 10 <= avg_sentence_length <= 20:
            readability_score += 0.2
        if 4 <= avg_word_length <= 6:
            readability_score += 0.2
            
        # Penalize very long or very short content
        if len(words) < 20:
            readability_score -= 0.1
        elif len(words) > 500:
            readability_score -= 0.1
            
        return max(0.0, min(1.0, readability_score))
    
    def _analyze_actionability(self, text: str) -> float:
        """Analyze how actionable the content is."""
        text_lower = text.lower()
        
        # Actionable verbs and phrases
        actionable_terms = ['visit', 'try', 'book', 'call', 'contact', 'go to', 'check out', 
                          'explore', 'experience', 'enjoy', 'taste', 'see', 'do']
        
        actionable_matches = sum(1 for term in actionable_terms if term in text_lower)
        actionability_score = min(0.8, actionable_matches * 0.1)
        
        # Specific instructions or recommendations
        instruction_terms = ['recommended', 'suggested', 'best', 'should', 'must', 'don\'t miss']
        instruction_matches = sum(1 for term in instruction_terms if term in text_lower)
        actionability_score += min(0.2, instruction_matches * 0.05)
        
        return min(1.0, actionability_score)
    
    def _calculate_persona_relevance(self, text_lower: str, persona_type: str) -> float:
        """Calculate relevance to specific persona types."""
        persona_keywords = {
            'student': ['budget', 'cheap', 'affordable', 'young', 'group', 'friends'],
            'family': ['family', 'children', 'kids', 'safe', 'suitable', 'activities'],
            'luxury': ['luxury', 'premium', 'exclusive', 'high-end', 'deluxe', 'finest'],
            'business': ['business', 'professional', 'conference', 'meeting', 'corporate'],
            'adventure': ['adventure', 'outdoor', 'active', 'sports', 'hiking', 'explore']
        }
        
        keywords = persona_keywords.get(persona_type.lower(), [])
        if not keywords:
            return 0.5
            
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(1.0, matches / len(keywords))
    
    def _generate_quality_breakdown(self, analysis: Dict) -> Dict:
        """Generate detailed quality breakdown."""
        return {
            'strengths': self._identify_strengths(analysis),
            'weaknesses': self._identify_weaknesses(analysis),
            'overall_assessment': self._generate_overall_assessment(analysis['overall_quality_score'])
        }
    
    def _identify_strengths(self, analysis: Dict) -> List[str]:
        """Identify content strengths."""
        strengths = []
        
        if analysis['specificity_score'] > 0.7:
            strengths.append("High specificity with detailed information")
        if analysis['informativeness_score'] > 0.7:
            strengths.append("Rich informative content")
        if analysis['relevance_score'] > 0.7:
            strengths.append("Highly relevant to job context")
        if analysis['actionability_score'] > 0.7:
            strengths.append("Contains actionable recommendations")
            
        return strengths if strengths else ["Basic content quality"]
    
    def _identify_weaknesses(self, analysis: Dict) -> List[str]:
        """Identify content weaknesses."""
        weaknesses = []
        
        if analysis['specificity_score'] < 0.4:
            weaknesses.append("Lacks specific details")
        if analysis['informativeness_score'] < 0.4:
            weaknesses.append("Limited informative value")
        if analysis['relevance_score'] < 0.4:
            weaknesses.append("Low relevance to job context")
        if analysis['actionability_score'] < 0.4:
            weaknesses.append("Few actionable recommendations")
            
        return weaknesses
    
    def _generate_improvement_suggestions(self, analysis: Dict, text: str) -> List[str]:
        """Generate suggestions for content improvement."""
        suggestions = []
        
        if analysis['specificity_score'] < 0.5:
            suggestions.append("Add more specific details like addresses, phone numbers, or hours")
        if analysis['informativeness_score'] < 0.5:
            suggestions.append("Include more practical information and descriptive details")
        if analysis['actionability_score'] < 0.5:
            suggestions.append("Add more actionable recommendations and specific suggestions")
            
        return suggestions
    
    def _generate_overall_assessment(self, score: float) -> str:
        """Generate overall quality assessment."""
        if score >= 0.8:
            return "Excellent quality content"
        elif score >= 0.6:
            return "Good quality content"
        elif score >= 0.4:
            return "Average quality content"
        else:
            return "Below average quality content"
    
    def _empty_quality_report(self) -> Dict:
        """Return empty quality report for invalid input."""
        return {
            'overall_quality_score': 0.0,
            'specificity_score': 0.0,
            'informativeness_score': 0.0,
            'relevance_score': 0.0,
            'readability_score': 0.0,
            'actionability_score': 0.0,
            'quality_breakdown': {
                'strengths': [],
                'weaknesses': ["No content to analyze"],
                'overall_assessment': "No content"
            },
            'improvement_suggestions': ["Provide content for analysis"]
        }

# Global instance
content_quality_analyzer = ContentQualityAnalyzer()
