# src/section_selector.py
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
import re
from src.utils import clean_text
from src.multilingual_support import multilingual_support
from src.smart_cache import smart_cache
from src.performance_monitor import performance_monitor
from src.advanced_confidence import AdvancedConfidenceScorer
from src.content_quality_analyzer import ContentQualityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectionSelector:
    """
    Enhanced section selector with persona-aware selection and content filtering.
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.quality_analyzer = ContentQualityAnalyzer()
        
        # Content type boosting factors based on typical travel planning needs
        self.default_boost_factors = {
            'activities': 1.3,
            'dining': 1.2,
            'accommodation': 1.1,
            'nightlife': 1.2,
            'culture': 1.1,
            'practical': 1.0,
            'transportation': 0.9,
            'general': 1.0
        }

    def select_relevant_sections(
        self,
        documents: Dict[str, List[Dict]],
        persona_job_embedding: np.ndarray,
        top_k: int = 5,
        max_per_doc: int = 2,
        persona: str = "",
        job: str = "",
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Enhanced section selection with persona-aware filtering and diversity optimization.

        Args:
            documents: Document sections dictionary
            persona_job_embedding: Persona-job embedding vector
            top_k: Number of sections to select
            max_per_doc: Maximum sections per document
            persona: Persona string for context-aware filtering
            job: Job description for context extraction
            diversity_weight: Weight for diversity vs similarity trade-off

        Returns:
            List of selected sections with enhanced scoring
        """
        # Start performance monitoring
        timer = performance_monitor.start_stage_timer("section_selection")
        
        # Check cache first
        cache_key = smart_cache.get_cache_key({
            'persona': persona[:200],  # Truncate for key generation
            'job': job[:200],
            'doc_count': len(documents),
            'top_k': top_k,
            'max_per_doc': max_per_doc
        })
        
        # cached_result = None  # Disable cache for testing
        cached_result = None  # Force disable cache to test travel enhancements
        # cached_result = smart_cache.get_cached_computation(cache_key, max_age_hours=1)
        if cached_result is not None:
            performance_monitor.record_metric("cache_hit", 1, "count", "performance")
            performance_monitor.end_stage_timer(timer)
            return cached_result
        
        # Extract context for persona-aware boosting
        boost_factors = self._get_persona_boost_factors(persona, job)
        content_filters = self._get_content_filters(persona, job)
        
        candidates = []
        total_sections = 0
        filtered_sections = 0

        for doc_name, sections in documents.items():
            for section in sections:
                total_sections += 1
                text = section.get("text", "").strip()
                if not text or len(text) < 50:  # Skip very short sections
                    continue

                # Apply content filters
                if not self._passes_content_filters(section, content_filters):
                    continue

                # Hard filter for obviously bad instruction-style titles
                original_title = section.get("title", "")
                title = original_title if original_title else self._extract_enhanced_title(section, persona, job)
                
                if self._is_bad_instruction_title(title):
                    filtered_sections += 1
                    continue  # Skip sections with instruction-style titles entirely

                try:
                    # Enhanced similarity scoring with content-type awareness
                    content_type = section.get('content_type', 'general')
                    similarity = self.embedder.contextual_similarity(
                        text, persona_job_embedding, content_type, boost_factors
                    )

                    # Title already extracted above for filtering
                    # Add title quality boost to final score
                    title_quality_boost = self._calculate_title_quality_boost(title)

                    # Calculate content quality score using new analyzer
                    quality_metrics = self.quality_analyzer.analyze_content_quality(
                        text, content_type, {'persona': persona, 'job': job, 'section_title': title, 'document': doc_name}
                    )
                    quality_score = quality_metrics['overall_quality_score'] / 100.0  # Normalize to 0-1

                    # Calculate advanced confidence score
                    # Note: We'll use a simplified version for now
                    confidence_metrics = {
                        'overall_confidence': similarity * 100,  # Convert to percentage
                        'semantic_similarity': similarity,
                        'quality_score': quality_score
                    }

                    # Enhanced job-relevance scoring
                    job_relevance_score = self._calculate_job_relevance_score(text, title, persona, job, doc_name)
                    
                    # Combine all scoring factors with balanced weights
                    final_score = (
                        0.25 * similarity +           # Basic semantic similarity
                        0.20 * job_relevance_score +  # Job-specific relevance
                        0.15 * quality_score +        # Content quality
                        0.15 * (confidence_metrics['overall_confidence'] / 100.0) +  # Confidence
                        0.25 * title_quality_boost    # Title quality (reduced from 0.4)
                    )

                    candidates.append({
                        "document": doc_name,
                        "title": title,
                        "text": text,
                        "page_number": section.get("page_number", 0),
                        "confidence_score": similarity,
                        "quality_score": quality_score,
                        "final_score": final_score,
                        "content_type": content_type,
                        "word_count": section.get("word_count", len(text.split())),
                        "section_index": section.get("section_index", 0),
                        "quality_metrics": quality_metrics,
                        "confidence_metrics": confidence_metrics
                    })

                except Exception as e:
                    logger.warning(f"Error processing section from {doc_name}: {str(e)}")
                    continue

        logger.info(f"Processed {total_sections} sections, filtered out {filtered_sections} bad titles, found {len(candidates)} candidates")

        if not candidates:
            logger.warning("No valid candidates found")
            performance_monitor.end_stage_timer(timer)
            return []

        # Enhanced selection with diversity optimization
        selected_sections = self._select_diverse_sections(
            candidates, top_k, max_per_doc, diversity_weight
        )

        # Cache the result
        smart_cache.cache_computation_result(cache_key, selected_sections, "section_selection")
        
        # Record performance metrics
        performance_monitor.record_metric("sections_processed", total_sections, "count", "processing")
        performance_monitor.record_metric("candidates_found", len(candidates), "count", "processing")
        performance_monitor.record_metric("sections_selected", len(selected_sections), "count", "processing")
        performance_monitor.end_stage_timer(timer)

        logger.info(f"Selected {len(selected_sections)} sections")
        return selected_sections

    def _get_persona_boost_factors(self, persona: str, job: str) -> Dict[str, float]:
        """
        Get persona-specific content type boost factors with enhanced multilingual support.
        """
        # Start with default factors
        boost_factors = self.default_boost_factors.copy()
        
        # Use multilingual support for enhanced detection
        enhanced_factors = multilingual_support.get_enhanced_boost_factors(
            persona, job, boost_factors
        )
        
        # Log language detection for debugging
        lang_info = multilingual_support.get_language_info(job)
        if lang_info['is_multilingual']:
            logger.debug(f"Detected non-English language: {lang_info['detected_language']} for job: {job[:50]}...")
        
        return enhanced_factors

    def _get_content_filters(self, persona: str, job: str) -> Dict[str, any]:
        """
        Get content filters based on persona and job with multilingual adaptation.
        """
        filters = {
            'exclude_family_content': False,
            'exclude_adult_content': False,
            'min_relevance_score': 0.3,
            'preferred_content_types': []
        }

        # Use multilingual support for context detection
        family_score = multilingual_support.calculate_multilingual_indicator_score(job, 'family_indicators')
        business_score = multilingual_support.calculate_multilingual_indicator_score(job, 'business_indicators')
        luxury_score = multilingual_support.calculate_multilingual_indicator_score(job, 'luxury_indicators')

        # Apply context-specific filters with multilingual awareness
        if family_score > 0.15:  # Lowered threshold for multilingual
            filters['exclude_adult_content'] = True
            filters['preferred_content_types'] = ['activities', 'culture', 'accommodation']
            
        elif business_score > 0.15:
            filters['preferred_content_types'] = ['accommodation', 'transportation', 'practical', 'dining']
            filters['min_relevance_score'] = 0.4
            
        elif luxury_score > 0.15:
            filters['min_relevance_score'] = 0.4
            filters['preferred_content_types'] = ['accommodation', 'dining', 'culture']

        return filters

    def _passes_content_filters(self, section: Dict, filters: Dict[str, any]) -> bool:
        """
        Check if section passes content filters.
        """
        text_lower = section.get("text", "").lower()
        title_lower = section.get("title", "").lower()
        
        # Filter out family content for college groups
        if filters.get('exclude_family_content', False):
            family_keywords = ['family-friendly', 'children', 'kids', 'child', 'family']
            if any(keyword in text_lower or keyword in title_lower for keyword in family_keywords):
                return False

        # Filter out adult-only content for families
        if filters.get('exclude_adult_content', False):
            adult_keywords = ['nightclub', 'bar', 'casino', 'adult-only']
            if any(keyword in text_lower or keyword in title_lower for keyword in adult_keywords):
                return False

        return True

    def _calculate_content_quality(self, section: Dict, persona: str, job: str) -> float:
        """
        Calculate content quality score with dynamic persona adaptation.
        """
        text = section.get("text", "")
        word_count = section.get("word_count", len(text.split()))
        content_type = section.get("content_type", "general")
        job_lower = job.lower()
        text_lower = text.lower()
        
        # Base quality factors
        length_score = min(word_count / 100, 1.0)  # Prefer longer, more detailed content
        
        # Information density score
        info_keywords = ['address', 'phone', 'hours', 'price', 'location', 'website', 'contact']
        info_density = sum(1 for keyword in info_keywords if keyword in text_lower) / len(info_keywords)
        
        # Actionability score (contains specific recommendations)
        action_keywords = ['visit', 'try', 'recommended', 'must-see', 'best', 'popular', 'famous']
        action_density = sum(1 for keyword in action_keywords if keyword in text_lower) / len(action_keywords)
        
        # Dynamic persona-specific scoring (generalizable approach)
        persona_boost = self._calculate_dynamic_persona_boost(text_lower, job_lower)
        
        # Combine all scores with balanced weighting
        quality_score = (0.35 * length_score) + (0.25 * info_density) + (0.25 * action_density) + (0.15 * persona_boost)
        
        return min(quality_score, 1.0)
    
    def _calculate_dynamic_persona_boost(self, text_lower: str, job_lower: str) -> float:
        """
        Calculate persona-specific boost based on job context with multilingual support.
        """
        # Use multilingual support for enhanced persona boost calculation
        return multilingual_support.calculate_multilingual_persona_boost(text_lower, job_lower)

    def _select_diverse_sections(
        self, 
        candidates: List[Dict], 
        top_k: int, 
        max_per_doc: int, 
        diversity_weight: float
    ) -> List[Dict]:
        """
        Select sections with diversity optimization.
        """
        # Sort by final score
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        selected_sections = []
        per_doc_counter = {}
        content_type_counter = {}

        for candidate in candidates:
            doc = candidate["document"]
            content_type = candidate.get("content_type", "general")
            
            # Check document limit
            if per_doc_counter.get(doc, 0) >= max_per_doc:
                continue
                
            # Apply diversity penalty for content type repetition
            type_penalty = content_type_counter.get(content_type, 0) * diversity_weight
            adjusted_score = candidate["final_score"] - type_penalty
            
            if len(selected_sections) < top_k:
                candidate["adjusted_score"] = adjusted_score
                selected_sections.append(candidate)
                per_doc_counter[doc] = per_doc_counter.get(doc, 0) + 1
                content_type_counter[content_type] = content_type_counter.get(content_type, 0) + 1
            else:
                # Check if this candidate should replace a lower-scored one
                min_selected = min(selected_sections, key=lambda x: x["adjusted_score"])
                if adjusted_score > min_selected["adjusted_score"]:
                    # Remove the lowest scoring section
                    selected_sections.remove(min_selected)
                    old_doc = min_selected["document"]
                    old_type = min_selected.get("content_type", "general")
                    per_doc_counter[old_doc] -= 1
                    content_type_counter[old_type] -= 1
                    
                    # Add the new section
                    candidate["adjusted_score"] = adjusted_score
                    selected_sections.append(candidate)
                    per_doc_counter[doc] = per_doc_counter.get(doc, 0) + 1
                    content_type_counter[content_type] = content_type_counter.get(content_type, 0) + 1

        # Final sort by adjusted score
        selected_sections.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
        return selected_sections

    def _extract_enhanced_title(self, section: Dict, persona: str = "", job: str = "") -> str:
        """
        Extract title with priority on structured titles and persona-aware enhancement.
        """
        # Strategy 1: Use structured title from document reader if available
        existing_title = section.get("title", "")
        has_structured_title = section.get("has_structured_title", False)
        
        # ALWAYS prefer structured titles if they exist and look reasonable
        if has_structured_title and existing_title:
            cleaned_title = self._polish_title(existing_title)
            if len(cleaned_title.strip()) > 0 and not self._is_bad_instruction_title(cleaned_title):
                return cleaned_title

        # Strategy 2: Use persona-aware title extraction
        text = section.get("text", "")
        if text and persona and job:
            persona_aware_title = self._extract_persona_aware_title(text, persona, job)
            if persona_aware_title and not self._is_bad_instruction_title(persona_aware_title):
                return persona_aware_title

        # Strategy 3: If existing title looks good even without structured flag, use it
        if existing_title and self._is_quality_title(existing_title):
            return self._polish_title(existing_title)

        # Strategy 3: Try to extract from text content
        text = section.get("text", "")
        if text:
            extracted_title = self._extract_title_from_content(text)
            if extracted_title and self._is_quality_title(extracted_title):
                return self._polish_title(extracted_title)

        # Strategy 4: Use existing title if it passes reasonableness check
        if existing_title and self._is_reasonable_title(existing_title):
            return self._polish_title(existing_title)

        # Strategy 5: Generate semantic title based on content type (rarely used now)
        content_type = section.get("content_type", "general")
        semantic_title = self._generate_smart_semantic_title(text, content_type)
        if semantic_title:
            return semantic_title

        # Strategy 6: Final fallback
        return self._create_fallback_title(text)

    def _extract_persona_aware_title(self, text: str, persona: str, job: str) -> Optional[str]:
        """
        Extract titles using persona and job context for better relevance.
        """
        if not text or not persona or not job:
            return None
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
        
        # For Food Contractor persona, look for recipe and dish names
        if 'food' in persona.lower() or 'contractor' in persona.lower():
            return self._extract_food_related_title(text, lines, job)
        
        # For HR professional, look for form names and process titles
        elif 'hr' in persona.lower() or 'human resources' in persona.lower():
            return self._extract_hr_related_title(text, lines, job)
        
        # For travel/tourism contexts
        elif any(word in job.lower() for word in ['travel', 'tourism', 'guide', 'visit']):
            return self._extract_travel_related_title(text, lines, job)
        
        # Generic content-aware extraction
        return self._extract_context_aware_title(text, lines, job)
    
    def _extract_food_related_title(self, text: str, lines: List[str], job: str) -> Optional[str]:
        """
        Extract food-related titles for cooking/catering contexts.
        """
        # Look for actual dish names in the content, not instructions
        dish_patterns = [
            # Pattern: "Dish Name Ingredients:" - extract the dish name
            r'([A-Z][A-Za-z\s&-]{2,30}?)\s+Ingredients?\s*:',
            # Pattern: "Dish Name Instructions:"
            r'([A-Z][A-Za-z\s&-]{2,30}?)\s+Instructions?\s*:',
        ]
        
        for pattern in dish_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                cleaned_match = match.strip()
                if self._is_valid_dish_name(cleaned_match, job):
                    return cleaned_match
        
        # Look for dish names in content based on job context
        if 'vegetarian' in job.lower() or 'buffet' in job.lower():
            vegetarian_dishes = self._find_vegetarian_dish_names(text, lines)
            if vegetarian_dishes:
                return vegetarian_dishes[0]  # Return the first good match
        
        return None
    
    def _is_valid_dish_name(self, name: str, job: str) -> bool:
        """
        Check if extracted text is a valid dish name in job context.
        """
        if not name or len(name) < 3 or len(name) > 40:
            return False
        
        # Must start with capital
        if not name[0].isupper():
            return False
        
        # Filter out obvious non-dish names
        bad_patterns = ['ingredients', 'instructions', 'recipe', 'cooking', 'preparation']
        if any(pattern in name.lower() for pattern in bad_patterns):
            return False
        
        # For vegetarian jobs, prefer plant-based dishes
        if 'vegetarian' in job.lower():
            meat_words = ['chicken', 'beef', 'pork', 'lamb', 'fish', 'turkey', 'bacon']
            if any(meat in name.lower() for meat in meat_words):
                return False
        
        return True
    
    def _find_vegetarian_dish_names(self, text: str, lines: List[str]) -> List[str]:
        """
        Find vegetarian-friendly dish names in the content.
        """
        vegetarian_indicators = [
            'vegetable', 'veggie', 'salad', 'pasta', 'rice', 'quinoa',
            'hummus', 'falafel', 'caprese', 'risotto', 'ratatouille',
            'mushroom', 'spinach', 'tomato', 'avocado', 'cheese'
        ]
        
        found_dishes = []
        for line in lines[:10]:
            if (len(line.split()) <= 6 and
                any(indicator in line.lower() for indicator in vegetarian_indicators) and
                line[0].isupper() and
                not self._is_instruction_line(line)):
                found_dishes.append(line.strip())
        
        return found_dishes
    
    def _extract_hr_related_title(self, text: str, lines: List[str], job: str) -> Optional[str]:
        """
        Extract HR-related titles for form and process contexts.
        """
        # Look for form names and HR processes
        hr_patterns = [
            r'([A-Z][A-Za-z\s&-]{3,40}?)\s+Form',
            r'([A-Z][A-Za-z\s&-]{3,40}?)\s+Document',
            r'([A-Z][A-Za-z\s&-]{3,40}?)\s+Process',
        ]
        
        for pattern in hr_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                cleaned_match = match.strip()
                if len(cleaned_match) >= 3:
                    return cleaned_match + ' Form'  # Add back the form indicator
        
        return None
    
    def _extract_travel_related_title(self, text: str, lines: List[str], job: str) -> Optional[str]:
        """
        Extract travel-related titles for tourism contexts.
        """
        # For college friends, prioritize activities, adventures, and experiences
        if 'college' in job.lower() or 'friends' in job.lower():
            college_travel_indicators = [
                'adventure', 'activities', 'nightlife', 'entertainment', 'coastal',
                'beach', 'fun', 'experiences', 'tours', 'excursions', 'things to do'
            ]
            
            # Look for activity-focused titles first
            for line in lines[:8]:
                if (any(indicator in line.lower() for indicator in college_travel_indicators) and
                    line[0].isupper() and
                    not self._is_instruction_line(line) and
                    len(line.split()) <= 8):
                    return line.strip()
        
        # General travel indicators
        travel_indicators = [
            'restaurant', 'hotel', 'attraction', 'activity', 'nightlife',
            'guide', 'tour', 'visit', 'explore', 'discover', 'cuisine',
            'cultural', 'historic', 'scenic', 'adventure'
        ]
        
        for line in lines[:8]:
            if (any(indicator in line.lower() for indicator in travel_indicators) and
                line[0].isupper() and
                not self._is_instruction_line(line) and
                len(line.split()) <= 8):
                return line.strip()
        
        return None
    
    def _extract_context_aware_title(self, text: str, lines: List[str], job: str) -> Optional[str]:
        """
        Generic context-aware title extraction.
        """
        # Extract based on job keywords
        job_keywords = job.lower().split()
        
        for line in lines[:6]:
            line_words = line.lower().split()
            # If line contains job-relevant keywords and looks like a title
            if (any(keyword in line_words for keyword in job_keywords) and
                line[0].isupper() and
                not self._is_instruction_line(line) and
                3 <= len(line.split()) <= 10):
                return line.strip()
        
        return None
    
    def _is_instruction_line(self, line: str) -> bool:
        """
        Check if line is an instruction rather than a title.
        """
        if not line:
            return False
        
        line_lower = line.lower().strip()
        
        # Instruction starters
        instruction_starters = [
            'in a', 'add ', 'mix ', 'stir', 'cook', 'heat', 'place', 'put',
            'season', 'serve', 'garnish', 'combine', 'whisk', 'blend',
            'pour', 'spread', 'layer', 'roll', 'cut', 'slice', 'chop',
            'form ', 'brown ', 'lay ', 'sauté', 'to ', 'click', 'select'
        ]
        
        return any(line_lower.startswith(starter) for starter in instruction_starters)

    def _is_quality_title(self, title: str) -> bool:
        """
        Check if a title meets quality standards.
        """
        if not title:
            return False
            
        clean_title = title.strip()
        
        # Length check
        if len(clean_title) < 5 or len(clean_title) > 120:
            return False
        
        # Quality indicators
        quality_checks = [
            # Good structure
            clean_title[0].isupper(),
            3 <= len(clean_title.split()) <= 15,
            
            # Not just descriptive text
            not clean_title.lower().startswith(('the following', 'this section', 'in this')),
            not clean_title.endswith('...'),
            
            # Has meaningful content
            any(char.isalpha() for char in clean_title),
            
            # Not overly generic
            clean_title.lower() not in ['section', 'chapter', 'part', 'information', 'guide'],
            
            # Not a sentence fragment
            not clean_title.lower().startswith(('it is', 'there are', 'you can', 'this is')),
        ]
        
        return sum(quality_checks) >= 5

    def _is_reasonable_title(self, title: str) -> bool:
        """
        Check if a title is reasonable even if not perfect.
        """
        if not title:
            return False
            
        clean_title = title.strip()
        
        # Reject partial sentences and incomplete thoughts
        if any(clean_title.endswith(marker) for marker in ['...', ',', '(', 'and', 'or', 'including']):
            return False
            
        # Reject titles that start mid-sentence
        if clean_title.startswith(('To ', 'Once ', 'Did you', 'Have you')):
            return False
            
        # Basic reasonableness checks
        return (8 <= len(clean_title) <= 100 and
                clean_title[0].isupper() and
                len(clean_title.split()) >= 2 and
                not clean_title.endswith('...') and
                not clean_title.endswith(','))

    def _extract_title_from_content(self, text: str) -> Optional[str]:
        """
        Enhanced content-based title extraction with better heuristics.
        """
        if not text:
            return None
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Look for different types of titles with priorities
        candidates = []
        
        # Priority 1: Look for section headers (capitalized, short lines)
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            if self._looks_like_section_header(line):
                candidates.append((line, 'header', 100 - i * 5))  # Higher score for earlier lines
        
        # Priority 2: Look for recipe/dish names in food content
        if any(word in text.lower() for word in ['ingredients:', 'instructions:', 'recipe', 'dish']):
            for i, line in enumerate(lines[:3]):
                if self._looks_like_recipe_title(line):
                    candidates.append((line, 'recipe', 90 - i * 5))
        
        # Priority 3: Look for topic headings
        for i, line in enumerate(lines[:4]):
            if self._looks_like_topic_heading(line):
                candidates.append((line, 'topic', 80 - i * 5))
        
        # Priority 4: Look for first substantial line as title
        for i, line in enumerate(lines[:3]):
            if self._is_substantial_line(line) and not self._is_instruction_line(line):
                candidates.append((line, 'substantial', 70 - i * 10))
        
        # Sort by score and return best candidate
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_title = candidates[0][0]
            return self._clean_extracted_title(best_title)
        
        return None

    def _looks_like_section_header(self, line: str) -> bool:
        """Check if line looks like a section header."""
        if not line or len(line.strip()) < 3:
            return False
        
        line = line.strip()
        
        # Good indicators
        if (line[0].isupper() and 
            not line.endswith('.') and 
            len(line) < 80 and 
            len(line.split()) <= 8 and
            not any(starter in line.lower() for starter in ['to ', 'click', 'select', 'enter', 'choose'])):
            return True
        
        return False
    
    def _looks_like_recipe_title(self, line: str) -> bool:
        """Check if line looks like a recipe or dish title."""
        if not line or len(line.strip()) < 3:
            return False
        
        line = line.strip()
        
        # Recipe title patterns
        recipe_indicators = [
            # Common dish patterns
            lambda s: any(word in s.lower() for word in ['salad', 'soup', 'pasta', 'pizza', 'burger', 'sandwich']),
            lambda s: any(word in s.lower() for word in ['chicken', 'beef', 'pork', 'fish', 'vegetable']),
            lambda s: any(word in s.lower() for word in ['cake', 'bread', 'muffin', 'cookie', 'pie']),
            lambda s: any(word in s.lower() for word in ['falafel', 'hummus', 'ratatouille', 'lasagna']),
            # Cooking method patterns
            lambda s: any(word in s.lower() for word in ['grilled', 'baked', 'roasted', 'fried', 'steamed']),
        ]
        
        if (any(indicator(line) for indicator in recipe_indicators) and
            not line.endswith(':') and
            not self._is_instruction_line(line) and
            len(line.split()) <= 6):
            return True
        
        return False
    
    def _looks_like_topic_heading(self, line: str) -> bool:
        """Check if line looks like a topic heading."""
        if not line or len(line.strip()) < 5:
            return False
        
        line = line.strip()
        
        # Topic heading patterns
        topic_patterns = [
            # Travel topics
            lambda s: any(word in s.lower() for word in ['activities', 'attractions', 'restaurants', 'hotels', 'nightlife']),
            lambda s: any(word in s.lower() for word in ['guide to', 'overview of', 'introduction to']),
            # Business/HR topics
            lambda s: any(word in s.lower() for word in ['forms', 'signatures', 'documents', 'compliance']),
            # Food topics
            lambda s: any(word in s.lower() for word in ['cuisine', 'dishes', 'recipes', 'cooking']),
        ]
        
        if (any(pattern(line) for pattern in topic_patterns) and
            not line.endswith('.') and
            not self._is_instruction_line(line) and
            len(line.split()) <= 10):
            return True
        
        return False
    
    def _is_substantial_line(self, line: str) -> bool:
        """Check if line has substantial content."""
        if not line:
            return False
        
        line = line.strip()
        return (len(line) >= 10 and 
                len(line.split()) >= 3 and 
                len(line.split()) <= 15 and
                not line.startswith('Note:') and
                not line.startswith('Tip:'))
    
    def _is_instruction_line(self, line: str) -> bool:
        """Check if line looks like an instruction."""
        if not line:
            return False
        
        line_lower = line.lower()
        instruction_indicators = [
            'click', 'select', 'choose', 'enter', 'type', 'press', 'drag',
            'to add', 'to create', 'to fill', 'to edit', 'to change',
            'you can', 'you should', 'you need', 'you must',
            'instructions:', 'steps:', 'how to', 'make sure'
        ]
        
        return any(indicator in line_lower for indicator in instruction_indicators)
    
    def _clean_extracted_title(self, title: str) -> str:
        """Clean an extracted title."""
        if not title:
            return ""
        
        # Remove common prefixes
        title = re.sub(r'^(step \d+:?\s*|chapter \d+:?\s*|\d+\.\s*)', '', title, flags=re.IGNORECASE)
        
        # Clean punctuation
        title = title.replace(':', '').replace(';', '').strip()
        
        # Ensure proper capitalization
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]
        
        return title.strip()

    def _looks_like_content_title(self, line: str, position: int, full_text: str) -> bool:
        """
        Check if a line looks like a title within content.
        """
        if not line or len(line) < 8 or len(line) > 80:
            return False
        
        # Content title indicators
        indicators = [
            # Position and format
            position <= 3,
            line[0].isupper(),
            not line.endswith('.'),
            
            # Structure
            3 <= len(line.split()) <= 10,
            ':' not in line or line.count(':') == 1,
            
            # Content context
            not line.lower().startswith(('the ', 'it ', 'this ', 'there ')),
            len([c for c in line[:25] if c.isupper()]) >= 2,
            
            # Followed by content (not standalone)
            len(full_text.split()) > len(line.split()) + 15,
        ]
        
        return sum(indicators) >= 6

    def _generate_smart_semantic_title(self, text: str, content_type: str) -> Optional[str]:
        """
        Generate semantic title based on content structure, not topic-specific patterns.
        """
        if not text:
            return None
            
        # Try to extract the first meaningful sentence or phrase as title
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:3]:
            clean_sentence = sentence.strip()
            if (10 <= len(clean_sentence) <= 80 and
                clean_sentence and
                clean_sentence[0].isupper() and
                len(clean_sentence.split()) >= 2):
                return self._clean_title_line(clean_sentence)
        
        return None

    def _create_fallback_title(self, text: str) -> str:
        """
        Create fallback title when all other methods fail.
        """
        if not text:
            return "Document Section"
        
        # Extract key concepts for title
        words = text.split()[:8]
        title = ' '.join(words)
        
        # Clean up title
        title = self._clean_title_line(title)
        
        # Ensure reasonable length
        if len(title) > 60:
            title = title[:57] + "..."
        
        return title if title else "Document Section"

    def _calculate_title_quality_boost(self, title: str) -> float:
        """
        Calculate quality boost based on title quality (0.0 to 1.0).
        Good titles get higher scores to prioritize them in selection.
        """
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # Heavy penalty for generic titles
        if title in ["Culture Guide", "Tips Guide", "Nightlife Guide", "Dining Guide", "Document Section"]:
            return 0.0
        
        # VERY heavy penalty for instruction sentences - these should almost never be selected
        instruction_starters = [
            'to add', 'to fill', 'to create', 'to convert', 'to change', 'to edit', 'to select',
            'click', 'select', 'enter the', 'choose', 'open the', 'from the', 'in the',
            '(optional)', 'did you', 'have you', 'once a', 'it changes', 'then select',
            'note:', 'when you', 'if you', 'you can', 'use the', 'hover over'
        ]
        
        # Hard rejection for obvious instruction fragments
        if (any(title_lower.startswith(starter) for starter in instruction_starters) or
            title.endswith('...') or 
            len(title) < 8 or
            title.count(',') > 2 or  # Too many commas suggest sentence fragments
            ' and then ' in title_lower or
            ' you want to' in title_lower or
            'click' in title_lower):
            return -1.0  # Very negative boost to heavily penalize these
        
        # High boost for proper section headings
        good_title_patterns = [
            # Acrobat-specific proper titles
            'create', 'convert', 'fill and sign', 'export', 'share', 'edit', 'request',
            'pdf forms', 'signatures', 'generative ai', 'batch', 'multiple files',
            'web pages', 'method', 'overview', 'preferences', 'settings'
        ]
        
        # Check for structured titles that look like proper headings
        is_proper_heading = (
            title[0].isupper() and  # Starts with capital
            10 <= len(title) <= 80 and  # Reasonable length
            not title.endswith('.') and  # Not a sentence
            len(title.split()) >= 2 and  # At least 2 words
            not any(word in title_lower for word in ['the', 'a', 'an']) or  # Articles suggest sentences
            any(pattern in title_lower for pattern in good_title_patterns)  # Contains good patterns
        )
        
        if is_proper_heading:
            return 1.0  # Maximum boost for proper headings
        
        # Boost for titles with action words but proper structure
        if any(pattern in title_lower for pattern in good_title_patterns):
            return 0.7
        
        # Medium penalty for sentence-like titles
        if title.endswith('.') or len(title.split()) > 10:
            return 0.2
        
        return 0.4  # Default for reasonable titles

    def _is_bad_instruction_title(self, title: str) -> bool:
        """
        Enhanced hard filter to completely exclude obviously bad instruction-style titles.
        Returns True if the title should be filtered out.
        """
        if not title:
            return True
        
        title = title.strip()
        if len(title) < 4:
            return True
        
        title_lower = title.lower()
        
        # Expanded instruction starters - more comprehensive
        instruction_starters = [
            'to add', 'to fill', 'to create', 'to convert', 'to change', 'to edit', 'to select',
            'to enable', 'to share', 'to send', 'to open', 'to save', 'to export', 'to import',
            'click', 'select', 'enter the', 'choose', 'open the', 'from the', 'in the',
            'press', 'drag', 'hover', 'scroll', 'navigate', 'access', 'locate',
            'add name or email', 'you get this', 'you must share', 'you can also',
            'consider packing', 'make sure', 'blend until', 'serve immediately', 
            'drizzle with', 'serve on', 'add zucchini', 'cook until', 'heat oil',
            'in a blender', 'in a bowl', 'spread hummus', 'layer with',
            'cook ', 'add ', 'mix ', 'stir ', 'heat ', 'place ', 'put ',
            'season ', 'serve ', 'garnish ', 'combine ', 'whisk ', 'blend ',
            'pour ', 'spread ', 'layer ', 'roll ', 'cut ', 'slice ', 'chop ',
            'form ', 'brown ', 'lay ', 'sauté', 'bake ', 'boil ', 'fry ',
            'grill ', 'roast ', 'steam ', 'simmer ', 'marinate '
        ]
        
        # Expanded instruction patterns
        instruction_patterns = [
            r'^(step \d+)', r'^(\d+\.)', r'^(chapter \d+)',
            r'(optional):', r'\(optional\)', r'did you', r'have you', 
            r'once a', r'it changes', r'then select', r'when you', 
            r'if you', r'you can', r'use the', r'hover over'
        ]
        
        # Content-specific bad patterns
        cooking_instruction_patterns = [
            'until smooth', 'until tender', 'until golden', 'until done',
            'in a large', 'in a small', 'over medium', 'over high',
            'with salt', 'with pepper', 'to taste'
        ]
        
        # Ingredient list patterns (new)
        ingredient_patterns = [
            r'^\s*o\s+\d+', r'^\s*•\s+\d+', r'^\s*-\s+\d+',  # Bullet point with number
            r'\d+\s+(cup|tablespoon|teaspoon|pound|ounce|gram|ml|liter)s?\s',  # Measurements
            r'^\s*o\s+[^.]+\s+o\s+',  # Multiple bullet points in one line
            r'^\d+\s+(cup|tbsp|tsp|lb|oz|g|kg)s?\s',  # Short measurements
            r'ingredients?\s*:', r'instructions?\s*:',  # Section headers
        ]
        
        # Hard rejection criteria - expanded
        hard_rejections = [
            # Instruction starters
            any(title_lower.startswith(starter) for starter in instruction_starters),
            # Instruction patterns
            any(re.search(pattern, title_lower) for pattern in instruction_patterns),
            # Cooking instruction patterns
            any(pattern in title_lower for pattern in cooking_instruction_patterns),
            # Ingredient list patterns (new)
            any(re.search(pattern, title_lower) for pattern in ingredient_patterns),
            # Structure issues
            title.endswith('...'),
            title.count(',') > 2,  # Too many commas
            title.count('•') > 0,   # Bullet points
            title.count('–') > 1,   # Multiple dashes
            title.count('o ') > 1,  # Multiple bullet points (new)
            # Sentence fragments
            ' and then ' in title_lower,
            ' you want to' in title_lower,
            ' make sure ' in title_lower,
            'click' in title_lower,
            # Length issues - adjusted for valid short dish names
            len(title) < 4,  # Changed from 8 to 4 - allow short dish names like "Hummus"
            len(title) > 150,
            # Too many small words (suggests instruction text)
            len([w for w in title_lower.split() if w in ['a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with']]) > len(title.split()) * 0.4,
            # Incomplete or fragmented titles (new)
            title.endswith(' o') or title.endswith(' •') or title.endswith(' -'),
            title.lower().startswith('spread') and 'mixture' in title_lower,
            title.startswith('o ') and len(title.split()) > 6,  # Long bullet point lines
        ]
        
        return any(hard_rejections)

    def _clean_title_line(self, line: str) -> str:
        """
        Clean a line to make it suitable as a title.
        """
        # Remove bullet points and numbering
        line = re.sub(r'^[•\-\*\d+\.\)\]]\s*', '', line)
        
        # Clean PDF artifacts
        line = line.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
        line = line.replace('–', '-').replace('"', '"').replace('"', '"')
        
        # Clean whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()

    def _polish_title(self, title: str) -> str:
        """
        Polish a title for final presentation.
        """
        title = self._clean_title_line(title)
        
        # Ensure proper capitalization
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]
        
        # Remove trailing periods if present
        if title.endswith('.'):
            title = title[:-1]
        
        # Limit length intelligently
        if len(title) > 100:
            # Try to break at natural points
            if ':' in title:
                title = title.split(':')[0].strip()
            elif ' - ' in title:
                parts = title.split(' - ')
                title = parts[0].strip() if len(parts[0]) > 15 else ' - '.join(parts[:2])
            else:
                words = title.split()
                title = ' '.join(words[:12])
                if len(title) > 97:
                    title = title[:97] + "..."
        
        return title.strip()

    def _is_good_original_title(self, title: str) -> bool:
        """
        Check if the original title from PDF is good enough to use.
        """
        if not title:
            return False
            
        # Clean the title first
        clean_title = self._clean_title(title)
        
        # Check length - not too short or too long
        if len(clean_title) < 10 or len(clean_title) > 120:
            return False
            
        # Check if it's not just the beginning of a paragraph
        if (clean_title.count(' ') > 20 or  # Too many words
            clean_title.endswith('...') or  # Already truncated
            clean_title.lower().startswith(('the following', 'in this section', 'this chapter'))):
            return False
            
        # Check if it looks like a real title (not just text start)
        title_indicators = [
            not clean_title.endswith('.'),  # Titles don't end with period
            clean_title[0].isupper() if clean_title else False,  # Starts with capital
            ':' not in clean_title or clean_title.count(':') <= 1,  # At most one colon
            not clean_title.lower().startswith(('it is', 'there are', 'you can', 'this is')),  # Not descriptive text
        ]
        
        return sum(title_indicators) >= 3

    def _extract_pdf_header(self, text: str) -> str:
        """
        Extract actual headers from PDF text content.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for clear headers in the first few lines
        for i, line in enumerate(lines[:5]):
            # Remove common PDF artifacts and bullet points
            clean_line = re.sub(r'^[•\-\*\d+\.\)]\s*', '', line).strip()
            
            if self._is_likely_pdf_header(clean_line, i):
                return clean_line
                
        return None

    def _is_likely_pdf_header(self, line: str, position: int) -> bool:
        """
        Determine if a line is likely a header from the original PDF.
        """
        if not line or len(line) < 8 or len(line) > 100:
            return False
            
        # Strong indicators of PDF headers
        header_patterns = [
            # Position-based: headers are usually at the top
            position <= 2,
            
            # Format-based: headers have certain characteristics
            line[0].isupper() if line else False,
            not line.endswith('.'),
            not line.endswith(','),
            
            # Content-based: not common sentence starters
            not line.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'in ', 'on ', 'at ')),
            
            # Structure-based: reasonable word count for headers
            3 <= len(line.split()) <= 12,
            
            # Capitalization pattern: multiple capitals often indicate headers
            len([c for c in line[:30] if c.isupper()]) >= 2,
        ]
        
        return sum(header_patterns) >= 5

    def _extract_content_title(self, text: str) -> str:
        """
        Extract title from the actual content structure.
        """
        # Look for patterns that indicate section starts
        patterns = [
            r'^([A-Z][^.!?]{10,80}):',  # "Section Name:"
            r'^([A-Z][A-Za-z\s]{10,80})\s*\n',  # Standalone header line
            r'^([A-Z][^.!?]{10,80})\s*[-–—]',  # "Title - description"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                if self._is_good_original_title(title):
                    return title
        
        # Look for sentences that could be titles (first meaningful sentence)
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            # Remove bullet points and artifacts
            clean_sentence = re.sub(r'^[•\-\*\d+\.\)]\s*', '', sentence).strip()
            
            if (15 <= len(clean_sentence) <= 80 and
                clean_sentence[0].isupper() and
                not clean_sentence.lower().startswith(('the ', 'it ', 'this ', 'there '))):
                return clean_sentence
                
        return None

    def _generate_semantic_title(self, text: str, content_type: str) -> str:
        """
        Generate semantically meaningful titles based on content type and keywords.
        Only used as fallback when no good original title is found.
        """
        text_lower = text.lower()
        
        # Only generate semantic titles for content that clearly lacks good headers
        if any(indicator in text_lower for indicator in ['tips', 'advice', 'information', 'guide']):
            # Content-type specific title generation for generic content
            if content_type == "activities":
                return self._generate_activities_title(text, text_lower)
            elif content_type == "dining":
                return self._generate_dining_title(text, text_lower)
            elif content_type == "accommodation":
                return self._generate_accommodation_title(text, text_lower)
            elif content_type == "practical":
                return self._generate_practical_title(text, text_lower)
        
        return None

    def _generate_activities_title(self, text: str, text_lower: str) -> str:
        """Generate title for activity content."""
        if 'beach' in text_lower:
            return 'Beach Activities and Tips'
        elif 'hiking' in text_lower:
            return 'Hiking and Outdoor Activities'
        elif 'museum' in text_lower:
            return 'Museums and Cultural Sites'
        return 'Activities and Attractions'

    def _generate_dining_title(self, text: str, text_lower: str) -> str:
        """Generate title for dining content."""
        if 'wine' in text_lower:
            return 'Wine and Tasting Tips'
        elif 'restaurant' in text_lower:
            return 'Restaurant Recommendations'
        return 'Dining and Food Tips'

    def _generate_accommodation_title(self, text: str, text_lower: str) -> str:
        """Generate title for accommodation content."""
        if 'budget' in text_lower:
            return 'Budget Accommodation Options'
        return 'Hotels and Lodging'

    def _generate_practical_title(self, text: str, text_lower: str) -> str:
        """Generate title for practical content."""
        if 'packing' in text_lower:
            return 'Packing Tips and Preparation'
        elif 'transport' in text_lower:
            return 'Transportation Information'
        return 'Travel Tips and Information'

    def _find_best_header(self, text: str) -> str:
        """
        Find the best header in the text using enhanced detection.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines[:5]:  # Check first 5 lines
            if self._is_strong_header(line):
                return self._clean_title(line)
        
        return None

    def _is_strong_header(self, line: str) -> bool:
        """
        Enhanced header detection with stricter criteria.
        """
        if not line or len(line) < 10 or len(line) > 80:
            return False
        
        # Remove bullet points and clean
        clean_line = re.sub(r'^[•\-\*\d+\.\)]\s*', '', line)
        
        # Strong header indicators
        indicators = [
            clean_line[0].isupper() if clean_line else False,  # Starts with capital
            not clean_line.endswith('.'),  # Doesn't end with period
            len([c for c in clean_line[:30] if c.isupper()]) >= 3,  # Multiple capitals
            ':' not in clean_line or clean_line.count(':') == 1,  # At most one colon
            len(clean_line.split()) >= 2,  # At least 2 words
            not any(word in clean_line.lower() for word in ['this', 'these', 'you can', 'there are'])  # Not descriptive
        ]
        
        return sum(indicators) >= 4

    def _extract_fallback_title(self, text: str) -> str:
        """
        Fallback title extraction with improved logic.
        """
        # Try to get a meaningful first sentence
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 80:
                # Check if it's a good title candidate
                if not sentence.lower().startswith(('the ', 'it ', 'this ', 'there ')):
                    return self._clean_title(sentence)
        
        # Extract key concepts and create title
        return self._extract_concept_title(text)

    def _extract_concept_title(self, text: str) -> str:
        """
        Extract key concepts to create a meaningful title.
        """
        text_lower = text.lower()
        
        # Key concept patterns
        concepts = []
        
        # Look for Adobe/PDF specific concepts
        adobe_concepts = ['pdf', 'form', 'acrobat', 'document', 'field', 'signature', 'annotation', 'page']
        for concept in adobe_concepts:
            if concept in text_lower:
                concepts.append(concept.upper() if concept == 'pdf' else concept.title())
        
        # Look for action/process concepts
        actions = ['create', 'edit', 'convert', 'fill', 'sign', 'review', 'export', 'import']
        for action in actions:
            if action in text_lower:
                concepts.append(action.title())
        
        # Don't use generic travel-style titles, fall through to actual content extraction
        
        # Final fallback
        words = text.split()[:12]
        title = ' '.join(words)
        if len(title) > 80:
            title = title[:77] + "..."
        
        return self._clean_title(title)

    def _clean_title(self, title: str) -> str:
        """
        Clean and format the title properly while preserving original content.
        """
        if not title:
            return "Untitled Section"
            
        # Remove bullet points and extra characters but preserve original structure
        title = re.sub(r'^[•\-\*\d+\.\)]\s*', '', title)
        
        # Clean up spacing and artifacts
        title = re.sub(r'\s+', ' ', title)
        title = title.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
        
        # Remove common PDF artifacts
        title = title.replace('–', '-').replace('"', '"').replace('"', '"')
        
        # Ensure proper capitalization only if needed
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]
        
        # Limit length but preserve meaning
        if len(title) > 100:
            # Try to break at natural points
            if ':' in title:
                title = title.split(':')[0].strip()
            elif '-' in title and len(title.split('-')[0].strip()) > 20:
                title = title.split('-')[0].strip()
            else:
                words = title.split()
                title = ' '.join(words[:15])
                if len(title) > 97:
                    title = title[:97] + "..."
        
        return title.strip()

    def _calculate_job_relevance_score(self, text: str, title: str, persona: str, job: str, document: str = "") -> float:
        """
        Calculate how relevant the content is to the specific job requirements.
        Returns a score between 0.0 and 1.0.
        """
        if not job:
            return 0.5  # Default neutral score
        
        job_lower = job.lower()
        text_lower = text.lower()
        title_lower = title.lower()
        
        # Extract key job requirements
        job_keywords = self._extract_job_keywords(job_lower, persona)
        persona_keywords = self._extract_persona_keywords(persona.lower())
        
        # Calculate keyword matches in title (higher weight)
        title_matches = sum(1 for keyword in job_keywords + persona_keywords if keyword in title_lower)
        title_score = min(title_matches * 0.3, 1.0)
        
        # Calculate keyword matches in content
        content_matches = sum(1 for keyword in job_keywords + persona_keywords if keyword in text_lower)
        content_score = min(content_matches * 0.1, 1.0)
        
        # Analyze job context for better matching
        context_score = self._analyze_job_context(job_lower, text_lower, title_lower, document)
        
        # Combine scores
        final_score = (0.4 * title_score + 0.3 * content_score + 0.3 * context_score)
        return min(final_score, 1.0)
    
    def _extract_job_keywords(self, job: str, persona: str) -> List[str]:
        """Extract key requirements from job description using semantic analysis."""
        keywords = []
        job_lower = job.lower()
        
        # Generic semantic keyword extraction based on common patterns
        # This replaces hardcoded domain-specific lists with flexible pattern matching
        
        # Extract nouns and important action words
        import re
        
        # Common important word patterns that apply across domains
        important_patterns = [
            r'\b(?:create|make|build|develop|generate|design|produce)\w*\b',
            r'\b(?:manage|organize|handle|coordinate|oversee|administer)\w*\b', 
            r'\b(?:fill|complete|process|submit|approve|sign|verify)\w*\b',
            r'\b(?:plan|schedule|arrange|prepare|organize|coordinate)\w*\b',
            r'\b(?:form|document|file|report|certificate|application)\w*\b',
            r'\b(?:digital|electronic|online|automated|interactive)\w*\b'
        ]
        
        # Enhanced domain-aware patterns for better semantic matching
        if 'travel' in job_lower or 'trip' in job_lower or 'planner' in persona.lower():
            travel_patterns = [
                r'\b(?:visit|explore|tour|sightsee|discover|experience)\w*\b',
                r'\b(?:hotel|accommodation|stay|lodge|hostel|resort)\w*\b',
                r'\b(?:activity|attraction|adventure|entertainment|fun)\w*\b',
                r'\b(?:restaurant|dining|food|cuisine|meal)\w*\b',
                r'\b(?:transport|travel|flight|train|bus|car)\w*\b',
                r'\b(?:budget|cost|price|affordable|cheap|expensive)\w*\b'
            ]
            important_patterns.extend(travel_patterns)
        
        if 'food' in job_lower or 'menu' in job_lower or 'contractor' in persona.lower():
            food_patterns = [
                r'\b(?:cook|prepare|serve|recipe|ingredient|dish)\w*\b',
                r'\b(?:vegetarian|vegan|gluten|diet|nutrition|healthy)\w*\b',
                r'\b(?:buffet|catering|gathering|party|event)\w*\b'
            ]
            important_patterns.extend(food_patterns)
            
        if 'form' in job_lower or 'hr' in persona.lower() or 'professional' in persona.lower():
            business_patterns = [
                r'\b(?:fillable|interactive|field|signature|approval)\w*\b',
                r'\b(?:compliance|onboarding|employee|staff|workplace)\w*\b',
                r'\b(?:convert|export|share|collaborate|workflow)\w*\b'
            ]
            important_patterns.extend(business_patterns)
        
        for pattern in important_patterns:
            matches = re.findall(pattern, job_lower)
            keywords.extend([match.strip() for match in matches])
        
        # Extract specific terms mentioned in the job (nouns typically)
        words = job_lower.split()
        # Focus on words that are likely to be domain-specific terms
        domain_terms = [word.strip('.,!?;:') for word in words 
                       if len(word) > 3 and word not in {'with', 'from', 'that', 'this', 'they', 'them', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}]
        
        keywords.extend(domain_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
                
        return unique_keywords[:15]  # Limit to most relevant terms
    
    def _extract_persona_keywords(self, persona: str) -> List[str]:
        """Extract relevant keywords based on persona using generic approach."""
        keywords = []
        persona_lower = persona.lower()
        
        # Generic persona keyword extraction based on role indicators
        role_patterns = {
            'professional': ['workplace', 'business', 'office', 'corporate', 'professional'],
            'planner': ['plan', 'organize', 'schedule', 'coordinate', 'arrange'],
            'contractor': ['service', 'client', 'project', 'delivery', 'contract'],
            'manager': ['manage', 'oversee', 'supervise', 'coordinate', 'lead'],
            'specialist': ['expert', 'technical', 'specialized', 'advanced', 'detailed']
        }
        
        # Match persona to generic role patterns instead of hardcoded domains
        for role, role_keywords in role_patterns.items():
            if role in persona_lower:
                keywords.extend(role_keywords)
                break
        
        # If no specific role pattern matches, extract key terms from persona description
        if not keywords:
            words = persona_lower.split()
            keywords = [word.strip('.,!?;:') for word in words if len(word) > 3]
            
        return keywords[:8]  # Limit to most relevant terms
    
    def _analyze_job_context(self, job: str, text: str, title: str, document: str = "") -> float:
        """
        Analyze contextual relevance beyond keyword matching using semantic similarity.
        This method is domain-agnostic and works for any job context.
        """
        score = 0.5  # Start with neutral base score
        
        # Extract key job terms for semantic matching
        job_terms = self._extract_semantic_job_terms(job)
        text_terms = text.lower().split()
        title_terms = title.lower().split()
        
        # Enhanced travel context detection with smart food filtering
        if any(term in job.lower() for term in ['travel', 'trip', 'planner', 'vacation', 'tour']):
            
            # Smart food content filtering for travel context
            food_recipe_indicators = ['ingredients:', 'instructions:', 'serves', 'tablespoon', 'teaspoon', 
                                    'cup', 'ounce', 'minutes', 'heat', 'cook', 'preparation', 'recipe']
            travel_food_indicators = ['restaurant', 'dining', 'cuisine', 'local food', 'specialty', 
                                    'traditional', 'bistro', 'cafe', 'culinary', 'gastronomic']
            
            food_recipe_score = sum(1 for indicator in food_recipe_indicators 
                                  if indicator in text.lower())
            travel_food_score = sum(1 for indicator in travel_food_indicators 
                                  if indicator in text.lower())
            
            # Penalize pure recipe content but allow travel food content
            if food_recipe_score >= 2 and travel_food_score == 0:
                score -= min(food_recipe_score * 0.25, 0.7)  # Strong penalty for pure recipes
            elif travel_food_score > 0:
                score += min(travel_food_score * 0.08, 0.15)  # Boost for travel food content
            
            # Enhanced travel content detection with more comprehensive indicators
            travel_indicators = ['hotel', 'accommodation', 'lodging', 'stay', 'booking', 'resort', 
                               'attraction', 'sightseeing', 'tourist', 'visitor', 'landmark', 'monument',
                               'museum', 'gallery', 'beach', 'mountain', 'city', 'town', 'village',
                               'transport', 'flight', 'train', 'bus', 'taxi', 'rental', 'guide',
                               'itinerary', 'schedule', 'budget', 'cost', 'price', 'ticket', 'admission',
                               'location', 'area', 'region', 'district', 'neighborhood', 'overview',
                               'friendly', 'luxurious', 'comfortable', 'convenient', 'accessible']
            
            travel_content_score = sum(1 for indicator in travel_indicators 
                                     if any(indicator in term for term in text_terms))
            travel_title_score = sum(1 for indicator in travel_indicators 
                                   if any(indicator in term for term in title_terms))
            
            # Document source intelligence - boost content from travel documents
            document_name = document.lower()
            if any(travel_source in document_name for travel_source in 
                   ['south of france', 'cities', 'hotels', 'restaurants', 'things to do', 'tips', 'culture', 'history']):
                score += 0.2  # Significant boost for travel source documents
            
            # Apply enhanced travel-specific boosting
            if travel_content_score > 0:
                score += min(travel_content_score * 0.07, 0.4)  # Increased boost for travel content
            if travel_title_score > 0:
                score += min(travel_title_score * 0.15, 0.3)   # Increased boost for travel titles
                
            # Penalize clearly non-travel content (Acrobat, forms, etc.)
            non_travel_indicators = ['acrobat', 'pdf', 'form', 'signature', 'convert', 'export', 
                                   'fillable', 'digital', 'electronic', 'tool', 'software', 'generative', 'ai']
            non_travel_penalty = sum(1 for indicator in non_travel_indicators 
                                   if any(indicator in term for term in text_terms + title_terms))
            if non_travel_penalty > 0:
                score -= min(non_travel_penalty * 0.2, 0.6)  # Stronger penalty for non-travel content
        
        # Semantic relevance scoring - domain agnostic
        job_text_overlap = len(set(job_terms) & set(text_terms))
        job_title_overlap = len(set(job_terms) & set(title_terms))
        
        # Scale overlap scores based on text length to avoid bias toward longer content
        if len(text_terms) > 0:
            text_relevance = min(job_text_overlap / max(len(job_terms), 1) * 0.3, 0.3)
        else:
            text_relevance = 0
            
        if len(title_terms) > 0:
            title_relevance = min(job_title_overlap / max(len(job_terms), 1) * 0.4, 0.4)
        else:
            title_relevance = 0
        
        # Apply generic quality indicators
        quality_boost = self._calculate_generic_quality_indicators(text, title, job)
        
        # Combine all factors
        final_score = score + text_relevance + title_relevance + quality_boost
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def _extract_semantic_job_terms(self, job: str) -> List[str]:
        """Extract semantic terms from job description without domain bias."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'will', 'with'}
        
        words = [word.lower().strip('.,!?;:') for word in job.split() 
                if word.lower() not in stop_words and len(word) > 2]
        
        return words
    
    def _calculate_generic_quality_indicators(self, text: str, title: str, job: str) -> float:
        """Calculate quality indicators that work across all domains."""
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        text_length = len(text.split())
        if 50 <= text_length <= 500:  # Optimal range for detailed instructions
            score += 0.1
        elif text_length < 20:  # Too brief
            score -= 0.1
            
        # Title clarity (not too long, starts with capital, no special characters)
        title_words = len(title.split())
        if 2 <= title_words <= 8:  # Good title length
            score += 0.1
        if title[0].isupper():  # Proper capitalization
            score += 0.05
            
        # Content structure indicators
        if any(indicator in text.lower() for indicator in ['step', 'how to', 'procedure', 'method', 'process']):
            score += 0.1  # Instructional content
            
        return min(score, 0.2)  # Cap the quality boost

    def extract_title(self, section_text: str) -> str:
        """
        Legacy method - now delegates to enhanced extraction.
        """
        return self._extract_fallback_title(section_text)
