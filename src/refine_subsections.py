# src/refine_subsections.py

from typing import List, Dict, Tuple
import logging
import re
import numpy as np
from src.utils import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubsectionRefiner:
    """
    Enhanced subsection refiner with intelligent text extraction and contextual understanding.
    """
    
    def __init__(self, embedder):
        self.embedder = embedder

    def refine(
        self,
        top_sections: List[Dict],
        persona_job_embedding: np.ndarray,
        max_sentences: int = 6,
        min_sentences: int = 3,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Enhanced refinement with contextual sentence selection and intelligent text extraction.

        Args:
            top_sections: Selected sections with full text
            persona_job_embedding: Persona-job embedding vector
            max_sentences: Maximum sentences to include in refined text
            min_sentences: Minimum sentences to ensure sufficient content
            context_window: Number of sentences around high-scoring ones to include for context

        Returns:
            List of refined sections with enhanced content extraction
        """
        refined_outputs = []

        for section in top_sections:
            try:
                refined_section = self._refine_single_section(
                    section, persona_job_embedding, max_sentences, min_sentences, context_window
                )
                if refined_section:
                    refined_outputs.append(refined_section)
            except Exception as e:
                logger.warning(f"Error refining section from {section.get('document', 'unknown')}: {str(e)}")
                continue

        return refined_outputs

    def _refine_single_section(
        self, 
        section: Dict, 
        persona_job_embedding: np.ndarray, 
        max_sentences: int, 
        min_sentences: int,
        context_window: int
    ) -> Dict:
        """
        Refine a single section with enhanced processing.
        """
        text = section.get("text") or ""
        cleaned = clean_text(text)
        
        # Enhanced sentence splitting
        sentences = self._smart_sentence_split(cleaned)
        
        if len(sentences) < min_sentences:
            logger.warning(f"Section has too few sentences ({len(sentences)}) - using full text")
            return {
                "document": section["document"],
                "refined_text": cleaned,
                "page_number": section["page_number"],
                "confidence_score": section.get("confidence_score", 0.0),
                "refinement_method": "full_text"
            }

        # Score sentences with enhanced logic
        sentence_scores = self._score_sentences_enhanced(sentences, persona_job_embedding, section)
        
        # Select best sentences with context awareness
        selected_sentences = self._select_sentences_with_context(
            sentences, sentence_scores, max_sentences, min_sentences, context_window
        )
        
        # Create refined text with improved flow
        refined_text = self._create_refined_text(selected_sentences, sentences)
        
        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(sentence_scores, selected_sentences, sentences)

        return {
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section["page_number"],
            "confidence_score": confidence_score,
            "refinement_method": "contextual_selection",
            "sentences_used": len(selected_sentences),
            "total_sentences": len(sentences)
        }

    def _smart_sentence_split(self, text: str) -> List[str]:
        """
        Enhanced sentence splitting with better handling of abbreviations and edge cases.
        """
        # Pre-process to handle common abbreviations
        abbrev_patterns = [
            (r'\bDr\.', 'Dr<DOT>'),
            (r'\bMr\.', 'Mr<DOT>'),
            (r'\bMrs\.', 'Mrs<DOT>'),
            (r'\bMs\.', 'Ms<DOT>'),
            (r'\be\.g\.', 'e<DOT>g<DOT>'),
            (r'\bi\.e\.', 'i<DOT>e<DOT>'),
            (r'\betc\.', 'etc<DOT>'),
            (r'\bvs\.', 'vs<DOT>'),
            (r'\bSt\.', 'St<DOT>')
        ]
        
        for pattern, replacement in abbrev_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Split on sentence endings with better regex
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        raw_sentences = sentence_pattern.split(text)
        
        # Post-process sentences
        sentences = []
        for sentence in raw_sentences:
            # Restore abbreviations
            for pattern, replacement in abbrev_patterns:
                sentence = sentence.replace(replacement, pattern.replace(r'\b', '').replace('\\', ''))
            
            sentence = sentence.strip()
            
            # Filter valid sentences
            if (len(sentence) >= 20 and  # Minimum length
                len(sentence.split()) >= 4 and  # Minimum word count
                any(c.isalpha() for c in sentence)):  # Contains letters
                sentences.append(sentence)
        
        return sentences

    def _score_sentences_enhanced(
        self, 
        sentences: List[str], 
        persona_job_embedding: np.ndarray, 
        section: Dict
    ) -> List[Tuple[float, int]]:
        """
        Enhanced sentence scoring with multiple factors.
        """
        sentence_scores = []
        content_type = section.get("content_type", "general")
        
        for i, sentence in enumerate(sentences):
            try:
                # Base semantic similarity
                sentence_embedding = self.embedder.embed_text(sentence)
                semantic_score = self.embedder.similarity(sentence_embedding, persona_job_embedding)
                
                # Content quality factors
                info_score = self._calculate_information_density(sentence)
                action_score = self._calculate_actionability_score(sentence)
                specificity_score = self._calculate_specificity_score(sentence)
                
                # Position bias (earlier sentences often more important)
                position_score = max(0.1, 1.0 - (i * 0.1))
                
                # Content type boosting
                type_boost = self._get_content_type_boost(sentence, content_type)
                
                # Combine scores
                final_score = (
                    0.4 * semantic_score +
                    0.2 * info_score +
                    0.15 * action_score +
                    0.1 * specificity_score +
                    0.1 * position_score +
                    0.05 * type_boost
                )
                
                sentence_scores.append((final_score, i))
                
            except Exception as e:
                logger.warning(f"Error scoring sentence {i}: {str(e)}")
                sentence_scores.append((0.0, i))
        
        return sentence_scores

    def _calculate_information_density(self, sentence: str) -> float:
        """
        Calculate information density of a sentence.
        """
        info_indicators = [
            'address', 'phone', 'hours', 'price', 'cost', 'location', 'website',
            'contact', 'booking', 'reservation', 'open', 'closed', 'monday',
            'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'euro', 'dollar', '$', '€', 'free', 'paid'
        ]
        
        sentence_lower = sentence.lower()
        info_count = sum(1 for indicator in info_indicators if indicator in sentence_lower)
        
        # Normalize by sentence length
        return min(info_count / 5.0, 1.0)

    def _calculate_actionability_score(self, sentence: str) -> float:
        """
        Calculate how actionable/recommendatory a sentence is.
        """
        action_words = [
            'visit', 'try', 'go', 'see', 'explore', 'discover', 'enjoy', 'experience',
            'recommended', 'must-see', 'best', 'top', 'popular', 'famous', 'excellent',
            'perfect', 'ideal', 'should', 'can', 'worth', 'amazing', 'beautiful'
        ]
        
        sentence_lower = sentence.lower()
        action_count = sum(1 for word in action_words if word in sentence_lower)
        
        return min(action_count / 3.0, 1.0)

    def _calculate_specificity_score(self, sentence: str) -> float:
        """
        Calculate how specific/detailed a sentence is.
        """
        # Look for specific details like numbers, proper nouns, specific times
        specificity_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d{1,2}:\d{2}\b',  # Times
            r'\b\d{1,2}(am|pm)\b',  # AM/PM times
            r'\b[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'  # Addresses
        ]
        
        specificity_count = 0
        for pattern in specificity_indicators:
            specificity_count += len(re.findall(pattern, sentence))
        
        return min(specificity_count / 5.0, 1.0)

    def _get_content_type_boost(self, sentence: str, content_type: str) -> float:
        """
        Apply content-type specific boosting.
        """
        sentence_lower = sentence.lower()
        
        type_keywords = {
            'activities': ['activity', 'tour', 'visit', 'explore', 'adventure'],
            'dining': ['restaurant', 'food', 'eat', 'cuisine', 'meal', 'dish'],
            'accommodation': ['hotel', 'room', 'stay', 'accommodation', 'lodge'],
            'nightlife': ['bar', 'club', 'night', 'music', 'dance', 'party'],
            'culture': ['museum', 'art', 'history', 'culture', 'heritage'],
            'transportation': ['transport', 'bus', 'train', 'metro', 'taxi']
        }
        
        if content_type in type_keywords:
            keyword_count = sum(1 for keyword in type_keywords[content_type] 
                              if keyword in sentence_lower)
            return min(keyword_count / 2.0, 1.0)
        
        return 0.0

    def _select_sentences_with_context(
        self, 
        sentences: List[str], 
        sentence_scores: List[Tuple[float, int]], 
        max_sentences: int, 
        min_sentences: int,
        context_window: int
    ) -> List[int]:
        """
        Select sentences with contextual awareness.
        """
        # Sort by score
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Start with top-scoring sentences
        selected_indices = set()
        
        # Add top sentences with their context
        for score, idx in sentence_scores[:max_sentences]:
            if len(selected_indices) >= max_sentences:
                break
                
            # Add the sentence itself
            selected_indices.add(idx)
            
            # Add context sentences if space allows
            for offset in range(1, context_window + 1):
                if len(selected_indices) >= max_sentences:
                    break
                    
                # Add previous sentence
                if idx - offset >= 0:
                    selected_indices.add(idx - offset)
                    
                # Add next sentence
                if idx + offset < len(sentences) and len(selected_indices) < max_sentences:
                    selected_indices.add(idx + offset)
        
        # Ensure minimum sentences
        if len(selected_indices) < min_sentences:
            for score, idx in sentence_scores[max_sentences:]:
                selected_indices.add(idx)
                if len(selected_indices) >= min_sentences:
                    break
        
        return sorted(list(selected_indices))

    def _create_refined_text(self, selected_indices: List[int], sentences: List[str]) -> str:
        """
        Create refined text maintaining natural flow with enhanced cleaning.
        """
        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Clean individual sentences first
        cleaned_sentences = []
        for sentence in selected_sentences:
            cleaned = self._clean_pdf_artifacts(sentence)
            cleaned_sentences.append(cleaned)
        
        # Join sentences with proper spacing
        refined_text = ' '.join(cleaned_sentences).strip()
        
        # Final cleanup
        refined_text = self._final_text_polish(refined_text)
        
        return refined_text

    def _clean_pdf_artifacts(self, text: str) -> str:
        """
        Clean PDF extraction artifacts and formatting issues.
        """
        # Fix common PDF ligature issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
        text = text.replace('–', '-').replace('"', '"').replace('"', '"')
        text = text.replace('…', '...')
        
        # Fix broken words (common PDF artifacts) 
        word_fixes = {
            'ther mal': 'thermal',
            'wate r': 'water', 
            'cent er': 'center',
            'appro priate': 'appropriate',
            'com fortable': 'comfortable',
            'res taurant': 'restaurant',
            'accom modation': 'accommodation',
            'Perpign an': 'Perpignan',
            'sy mbol': 'symbol',
            'coz y': 'cozy',
            'travel ers': 'travelers',
            'Mont pellier': 'Montpellier',
            'Marse ille': 'Marseille',
            'beauti ful': 'beautiful',
            'wonder ful': 'wonderful',
            'inter esting': 'interesting',
            'import ant': 'important',
            'excel lent': 'excellent'
        }
        
        for broken, fixed in word_fixes.items():
            text = text.replace(broken, fixed)
        
        # Fix spacing issues around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix spacing around colons and list formatting
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*-\s*', ' - ', text)
        
        return text.strip()

    def _final_text_polish(self, text: str) -> str:
        """
        Final polishing of the refined text.
        """
        # Clean multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix common formatting issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Ensure text ends properly
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()

    def _calculate_overall_confidence(
        self, 
        sentence_scores: List[Tuple[float, int]], 
        selected_indices: List[int], 
        sentences: List[str]
    ) -> float:
        """
        Calculate overall confidence score for the refined section.
        """
        if not sentence_scores or not selected_indices:
            return 0.0
        
        # Get scores for selected sentences
        score_dict = {idx: score for score, idx in sentence_scores}
        selected_scores = [score_dict.get(idx, 0.0) for idx in selected_indices]
        
        # Calculate weighted average (higher weight for better scores)
        if selected_scores:
            return sum(selected_scores) / len(selected_scores)
        
        return 0.0

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Legacy method maintained for compatibility.
        """
        return self._smart_sentence_split(text)
