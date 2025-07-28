import os
import logging
from typing import Dict, List, Optional
from PyPDF2 import PdfReader
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentReader:
    """
    Enhanced document reader with better section extraction and error handling.
    """

    def __init__(self, min_section_length: int = 50, max_section_length: int = 3000):
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length

    def load_documents(self, folder_path: str) -> Dict[str, List[Dict]]:
        """
        Loads all PDFs and extracts structured sections with enhanced processing.

        Args:
            folder_path: Path to the folder containing PDF files.

        Returns:
            Dictionary mapping file name to list of sections with enhanced metadata.
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder path does not exist: {folder_path}")
            return {}

        documents = {}
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for filename in pdf_files:
            try:
                filepath = os.path.join(folder_path, filename)
                sections = self.extract_sections_from_pdf(filepath)
                if sections:
                    documents[filename] = sections
                    logger.info(f"Extracted {len(sections)} sections from {filename}")
                else:
                    logger.warning(f"No valid sections extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

        return documents

    def extract_sections_from_pdf(self, filepath: str) -> List[Dict]:
        """
        Enhanced PDF section extraction with better title detection using formatting analysis.

        Args:
            filepath: Full path to the PDF file.

        Returns:
            List of section dicts with enhanced metadata and proper title extraction.
        """
        try:
            reader = PdfReader(filepath)
        except Exception as e:
            logger.error(f"Failed to read PDF {filepath}: {str(e)}")
            return []

        sections = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if not text or len(text.strip()) < self.min_section_length:
                    continue

                # Enhanced text cleaning and preprocessing
                cleaned_text = self._preprocess_text(text)
                if len(cleaned_text) < self.min_section_length:
                    continue

                # Try to extract title using hierarchical approach
                page_title = self._extract_hierarchical_title(cleaned_text, i + 1)

                # Smart section detection and splitting
                page_sections = self._split_into_logical_sections_with_titles(
                    cleaned_text, i + 1, page_title
                )
                sections.extend(page_sections)

            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1} in {filepath}: {str(e)}")
                continue

        return sections

    def _extract_hierarchical_title(self, text: str, page_number: int) -> Optional[str]:
        """
        Extract title using hierarchical approach: bold/uppercase -> secondary -> content-based
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None

        # Strategy 1: Look for bold/uppercase titles (primary titles)
        primary_title = self._find_primary_title(lines)
        if primary_title and self._is_valid_title(primary_title):
            return primary_title

        # Strategy 2: Look for secondary titles (formatted but not bold)
        secondary_title = self._find_secondary_title(lines)
        if secondary_title and self._is_valid_title(secondary_title):
            return secondary_title

        # Strategy 3: Content-based title generation (only if no structured titles found)
        content_title = self._generate_content_based_title(text, lines)
        if content_title:
            return content_title

        return None

    def _find_primary_title(self, lines: List[str]) -> Optional[str]:
        """
        Find primary titles - these are typically bold, all caps, or heavily formatted.
        """
        for i, line in enumerate(lines[:5]):  # Check first 5 lines for primary titles
            # Clean line from artifacts
            clean_line = self._clean_line_artifacts(line)
            
            if self._is_primary_title_candidate(clean_line, i):
                return clean_line
                
        return None

    def _find_secondary_title(self, lines: List[str]) -> Optional[str]:
        """
        Find secondary titles - these have title-like characteristics but aren't bold.
        """
        for i, line in enumerate(lines[:8]):  # Check first 8 lines for secondary titles
            clean_line = self._clean_line_artifacts(line)
            
            if self._is_secondary_title_candidate(clean_line, i):
                return clean_line
                
        return None

    def _is_primary_title_candidate(self, line: str, position: int) -> bool:
        """
        Check if line is a primary title candidate (bold/uppercase).
        """
        if not line or len(line) < 5 or len(line) > 100:
            return False

        # Primary title indicators (bold-like characteristics)
        indicators = [
            # Position: primary titles are usually at the very top
            position <= 2,
            
            # Uppercase characteristics (simulating bold detection)
            len([c for c in line if c.isupper()]) >= len(line) * 0.6,  # Mostly uppercase
            
            # Format characteristics
            line[0].isupper(),
            not line.endswith('.'),
            not line.endswith(','),
            
            # Content characteristics
            not line.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'we ')),
            3 <= len(line.split()) <= 10,  # Reasonable word count for titles
            
            # No common sentence patterns
            not any(pattern in line.lower() for pattern in [
                'is located', 'can be found', 'are available', 'include', 'such as'
            ]),
            
            # Structure: not too wordy
            line.count(' ') <= 8,
        ]
        
        return sum(indicators) >= 6

    def _is_secondary_title_candidate(self, line: str, position: int) -> bool:
        """
        Check if line is a secondary title candidate.
        """
        if not line or len(line) < 8 or len(line) > 80:
            return False

        # Secondary title indicators
        indicators = [
            # Position: secondary titles in first part of page
            position <= 6,
            
            # Format characteristics
            line[0].isupper(),
            not line.endswith('.'),
            not line.endswith(','),
            
            # Title-like structure
            ':' in line and line.count(':') == 1,  # Often have colons
            len(line.split()) >= 3,  # Multi-word
            
            # Not typical sentence starters
            not line.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'we ')),
            
            # Has some uppercase letters (title case)
            len([c for c in line[:20] if c.isupper()]) >= 2,
            
            # Not too descriptive
            not any(pattern in line.lower() for pattern in [
                'located in', 'can be', 'are known', 'is famous', 'you will find'
            ])
        ]
        
        return sum(indicators) >= 5

    def _is_valid_title(self, title: str) -> bool:
        """
        Validate if the extracted title is actually a good title.
        """
        if not title:
            return False
            
        clean_title = title.strip()
        
        # Length check
        if len(clean_title) < 5 or len(clean_title) > 120:
            return False
        
        # Content validation
        invalid_patterns = [
            # Not just page numbers or artifacts
            clean_title.isdigit(),
            len(clean_title.split()) == 1 and clean_title.isdigit(),
            
            # Not common PDF artifacts
            clean_title.lower() in ['page', 'chapter', 'section', 'part'],
            
            # Not generic descriptive text
            clean_title.lower().startswith(('this section', 'in this chapter', 'the following')),
            
            # Not questions or incomplete sentences
            clean_title.endswith('?'),
            clean_title.count('...') > 0,
        ]
        
        if any(invalid_patterns):
            return False
        
        # Positive validation
        valid_indicators = [
            # Good title characteristics
            clean_title[0].isupper(),
            not clean_title.endswith('.'),
            3 <= len(clean_title.split()) <= 12,
            
            # Contains meaningful content
            any(char.isalpha() for char in clean_title),
            
            # Not too many articles/prepositions (indicates descriptive text)
            len([word for word in clean_title.lower().split() 
                 if word in ['the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with']]) <= len(clean_title.split()) * 0.4
        ]
        
        return sum(valid_indicators) >= 4

    def _generate_content_based_title(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Enhanced content-based title generation with recipe name detection.
        """
        # First try to find recipe names within the content
        recipe_title = self._extract_recipe_name_from_content(text, lines)
        if recipe_title:
            return recipe_title
        
        # Try to find dish names or food items
        dish_title = self._extract_dish_name_from_content(text, lines)
        if dish_title:
            return dish_title
        
        # Fall back to general title extraction
        return self._extract_general_content_title(text, lines)
    
    def _extract_recipe_name_from_content(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Extract recipe names from food content using pattern recognition.
        """
        # Look for recipe name patterns in text
        recipe_patterns = [
            # Pattern: "RecipeName Ingredients:" - common in recipe documents
            r'([A-Z][A-Za-z\s&-]{3,40}?)\s+Ingredients?\s*:',
            # Pattern: "RecipeName Instructions:" 
            r'([A-Z][A-Za-z\s&-]{3,40}?)\s+Instructions?\s*:',
            # Pattern: Lines that look like dish names before ingredients/instructions
            r'^([A-Z][A-Za-z\s&-]{3,40}?)(?=\s*(?:Ingredients?|Instructions?|•|\d+\s+(?:cup|tablespoon|teaspoon)))',
        ]
        
        for pattern in recipe_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                cleaned_match = match.strip()
                if self._is_valid_recipe_name(cleaned_match):
                    return cleaned_match
        
        # Look in individual lines for recipe names
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            cleaned_line = self._clean_line_artifacts(line)
            if self._looks_like_recipe_name(cleaned_line, i, text):
                return cleaned_line
        
        return None
    
    def _extract_dish_name_from_content(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Extract dish names using food-specific patterns.
        """
        # Common dish name indicators
        dish_indicators = [
            'salad', 'soup', 'pasta', 'pizza', 'burger', 'sandwich', 'wrap',
            'chicken', 'beef', 'pork', 'fish', 'vegetable', 'rice', 'noodles',
            'cake', 'bread', 'muffin', 'cookie', 'pie', 'tart',
            'falafel', 'hummus', 'risotto', 'stroganoff', 'stir-fry', 'curry'
        ]
        
        # Look for lines containing dish indicators that could be titles
        for line in lines[:8]:
            cleaned_line = self._clean_line_artifacts(line)
            if (len(cleaned_line.split()) <= 6 and 
                any(indicator in cleaned_line.lower() for indicator in dish_indicators) and
                not self._is_instruction_line(cleaned_line) and
                cleaned_line[0].isupper()):
                return cleaned_line
        
        return None
    
    def _extract_general_content_title(self, text: str, lines: List[str]) -> Optional[str]:
        """
        General title extraction for non-recipe content.
        """
        # Try to find a meaningful first sentence or phrase
        if lines:
            first_line = self._clean_line_artifacts(lines[0])
            if (10 <= len(first_line) <= 80 and 
                first_line[0].isupper() and
                not first_line.endswith('.') and
                not self._is_instruction_line(first_line)):
                return first_line
        
        # Try to extract from first sentence
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:2]:
            clean_sentence = self._clean_line_artifacts(sentence.strip())
            if (15 <= len(clean_sentence) <= 80 and
                clean_sentence[0].isupper() and
                not clean_sentence.lower().startswith(('the ', 'it ', 'this ', 'there ')) and
                not self._is_instruction_line(clean_sentence)):
                return clean_sentence
        
        return None
    
    def _is_valid_recipe_name(self, name: str) -> bool:
        """
        Check if extracted text looks like a valid recipe name.
        """
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        # Must start with capital letter
        if not name[0].isupper():
            return False
        
        # Shouldn't be an instruction
        if self._is_instruction_line(name):
            return False
        
        # Shouldn't contain measurement indicators
        measurement_words = ['cup', 'tablespoon', 'teaspoon', 'pound', 'ounce', 'gram', 'ml', 'liter']
        if any(word in name.lower() for word in measurement_words):
            return False
        
        # Should be reasonable word count for a dish name
        word_count = len(name.split())
        if word_count > 6:  # Too long for a dish name
            return False
        
        return True
    
    def _looks_like_recipe_name(self, line: str, position: int, full_text: str) -> bool:
        """
        Check if a line looks like a recipe name based on position and context.
        """
        if not line or len(line) < 3:
            return False
        
        # Must start with capital
        if not line[0].isupper():
            return False
        
        # Check if followed by ingredients or instructions
        text_after = full_text[full_text.find(line) + len(line):200].lower()
        has_recipe_context = any(keyword in text_after for keyword in 
                               ['ingredients:', 'instructions:', '• 1', '• 2', '1 cup', '2 cups'])
        
        # Good characteristics for recipe names
        good_characteristics = [
            position <= 2,  # Early in content
            not line.endswith('.'),  # Not a sentence
            2 <= len(line.split()) <= 5,  # Reasonable word count
            has_recipe_context,  # Followed by recipe content
            not self._is_instruction_line(line),  # Not an instruction
        ]
        
        return sum(good_characteristics) >= 3
    
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
            'form ', 'brown ', 'lay ', 'sauté'
        ]
        
        return any(line_lower.startswith(starter) for starter in instruction_starters)

    def _clean_line_artifacts(self, line: str) -> str:
        """
        Clean PDF extraction artifacts from a line while preserving structure.
        """
        # Remove bullet points and numbering
        line = re.sub(r'^[•\-\*\d+\.\)\]]\s*', '', line)
        
        # Clean common PDF artifacts
        line = line.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
        line = line.replace('–', '-').replace('"', '"').replace('"', '"')
        
        # Clean excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()

    def _split_into_logical_sections_with_titles(
        self, text: str, page_number: int, page_title: Optional[str]
    ) -> List[Dict]:
        """
        Split page text into logical sections with proper title assignment.
        """
        sections = []
        
        # Try to identify section breaks (headers, topic changes)
        potential_breaks = self._find_section_breaks(text)
        
        if not potential_breaks:
            # Single section for the whole page
            section_dict = self._create_section_dict_with_title(
                text, page_number, 0, page_title
            )
            sections.append(section_dict)
        else:
            # Multiple sections based on identified breaks
            for i, (start, end, section_title_hint) in enumerate(potential_breaks):
                section_text = text[start:end].strip()
                if len(section_text) >= self.min_section_length:
                    
                    # Use section-specific title or fall back to page title
                    final_title = section_title_hint if section_title_hint else page_title
                    
                    section_dict = self._create_section_dict_with_title(
                        section_text, page_number, i, final_title
                    )
                    sections.append(section_dict)

        return sections

    def _create_section_dict_with_title(
        self, text: str, page_number: int, section_index: int, title: Optional[str]
    ) -> Dict:
        """
        Create section dictionary with proper title handling.
        """
        # Use provided title or generate fallback
        final_title = title if title else self._generate_fallback_title(text)
        
        content_type = self._analyze_content_type(text)
        
        return {
            "title": final_title,
            "text": text,
            "page_number": page_number,
            "section_index": section_index,
            "content_type": content_type,
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_structured_title": title is not None
        }

    def _generate_fallback_title(self, text: str) -> str:
        """
        Generate fallback title when no structured title is found.
        """
        # Try to extract first meaningful sentence
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:2]:
            sentence = self._clean_line_artifacts(sentence.strip())
            if (10 <= len(sentence) <= 60 and 
                sentence[0].isupper() and
                not sentence.lower().startswith(('the ', 'it ', 'this '))):
                return sentence
        
        # Ultimate fallback: descriptive title based on content
        words = text.split()[:10]
        return ' '.join(words) + "..." if len(' '.join(words)) > 50 else ' '.join(words)

    def _preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing with better cleaning.
        """
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction artifacts
        text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = text.replace("–", "-").replace("—", "-").replace(""", '"').replace(""", '"')
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\b\d+\s*$', '', text)  # Remove trailing page numbers
        text = re.sub(r'^\s*\d+\s*', '', text)  # Remove leading page numbers
        
        return text.strip()

    def _split_into_logical_sections(self, text: str, page_number: int) -> List[Dict]:
        """
        Split page text into logical sections based on content structure.
        """
        sections = []
        
        # Try to identify section breaks (headers, topic changes)
        potential_breaks = self._find_section_breaks(text)
        
        if not potential_breaks:
            # If no clear breaks, treat as single section but check length
            if len(text) > self.max_section_length:
                # Split long text into smaller chunks
                chunks = self._split_by_sentences(text, self.max_section_length)
                for i, chunk in enumerate(chunks):
                    sections.append(self._create_section_dict(chunk, page_number, i))
            else:
                sections.append(self._create_section_dict(text, page_number, 0))
        else:
            # Split based on identified breaks
            for i, (start, end, title_hint) in enumerate(potential_breaks):
                section_text = text[start:end].strip()
                if len(section_text) >= self.min_section_length:
                    section_dict = self._create_section_dict(section_text, page_number, i)
                    if title_hint:
                        section_dict['title_hint'] = title_hint
                    sections.append(section_dict)

        return sections

    def _find_section_breaks(self, text: str) -> List[tuple]:
        """
        Find potential section breaks based on formatting patterns.
        """
        breaks = []
        lines = text.split('\n')
        
        # Look for potential headers (short lines, title case, etc.)
        for i, line in enumerate(lines):
            line = line.strip()
            if (len(line) > 10 and len(line) < 80 and 
                line[0].isupper() and 
                not line.endswith('.') and
                sum(c.isupper() for c in line[:20]) > 3):
                
                # Calculate position in original text
                start_pos = text.find(line)
                if start_pos != -1:
                    breaks.append((start_pos, line))

        # Convert to (start, end, title) tuples
        structured_breaks = []
        for i, (pos, title) in enumerate(breaks):
            start = pos
            end = breaks[i + 1][0] if i + 1 < len(breaks) else len(text)
            structured_breaks.append((start, end, title))

        return structured_breaks

    def _split_by_sentences(self, text: str, max_length: int) -> List[str]:
        """
        Split text by sentences while respecting max length.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _create_section_dict(self, text: str, page_number: int, section_index: int) -> Dict:
        """
        Create standardized section dictionary with enhanced metadata.
        """
        title = self.generate_enhanced_title(text)
        content_type = self._analyze_content_type(text)
        
        return {
            "title": title,
            "text": text,
            "page_number": page_number,
            "section_index": section_index,
            "content_type": content_type,
            "word_count": len(text.split()),
            "char_count": len(text)
        }

    def generate_enhanced_title(self, text: str, max_len: int = 100) -> str:
        """
        Enhanced title generation that prioritizes original PDF content.
        """
        if not text:
            return "Untitled Section" 
            
        # Strategy 1: Look for clear headers at the beginning (original PDF structure)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Check first few lines for original headers
        for i, line in enumerate(lines[:4]):
            # Clean the line but preserve original structure
            clean_line = re.sub(r'^[•\-\*\d+\.\)]\s*', '', line).strip()
            
            if self._is_original_header(clean_line, i, text):
                if len(clean_line) <= max_len:
                    return clean_line
                else:
                    # Truncate at natural break points
                    return self._smart_truncate(clean_line, max_len)
        
        # Strategy 2: Look for section patterns in original text
        section_title = self._extract_section_pattern(text, max_len)
        if section_title:
            return section_title
            
        # Strategy 3: Use first meaningful sentence from original content
        first_sentence = self._extract_first_sentence(text, max_len)
        if first_sentence:
            return first_sentence
            
        # Strategy 4: Fallback to smart truncation
        return self._smart_truncate(text, max_len)

    def _is_original_header(self, line: str, position: int, full_text: str) -> bool:
        """
        Check if a line is likely an original header from the PDF.
        """
        if not line or len(line) < 8:
            return False
            
        # Original header indicators
        indicators = [
            # Position: headers are usually at the top
            position <= 2,
            
            # Format: proper header formatting
            line[0].isupper(),
            not line.endswith('.'),
            not line.endswith(','),
            
            # Content: not typical sentence starters
            not line.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'we ', 'they ')),
            
            # Structure: reasonable length for headers
            8 <= len(line) <= 80,
            3 <= len(line.split()) <= 12,
            
            # Style: header-like capitalization
            len([c for c in line[:30] if c.isupper()]) >= 2,
            
            # Context: followed by content (not standalone)
            len(full_text.split()) > len(line.split()) + 10
        ]
        
        return sum(indicators) >= 6

    def _extract_section_pattern(self, text: str, max_len: int) -> Optional[str]:
        """
        Extract titles using common section patterns from PDFs.
        """
        # Pattern 1: "Section Name:" followed by content  
        pattern1 = re.search(r'^([A-Z][^:]{8,60}):\s', text, re.MULTILINE)
        if pattern1:
            title = pattern1.group(1).strip()
            if len(title) <= max_len:
                return title
        
        # Pattern 2: "Section Name -" or "Section Name –"
        pattern2 = re.search(r'^([A-Z][^-–]{8,60})\s*[-–]\s', text, re.MULTILINE)
        if pattern2:
            title = pattern2.group(1).strip() 
            if len(title) <= max_len:
                return title
                
        # Pattern 3: Standalone title line (common in PDFs)
        lines = text.split('\n')
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            # Remove bullet points but preserve structure
            clean_line = re.sub(r'^[•\-\*\d+\.\)]\s*', '', line)
            
            if (8 <= len(clean_line) <= max_len and 
                clean_line[0].isupper() and
                not clean_line.endswith('.') and
                i + 1 < len(lines) and len(lines[i + 1].strip()) > 20):  # Followed by content
                return clean_line
                
        return None

    def _extract_first_sentence(self, text: str, max_len: int) -> Optional[str]:
        """
        Extract first meaningful sentence if it can serve as a title.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            # Remove bullet points
            clean_sentence = re.sub(r'^[•\-\*\d+\.\)]\s*', '', sentence).strip()
            
            if (15 <= len(clean_sentence) <= max_len and
                clean_sentence[0].isupper() and
                not clean_sentence.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'we '))):
                return clean_sentence
                
        return None

    def _smart_truncate(self, text: str, max_len: int) -> str:
        """
        Smart truncation that preserves meaning and original structure.
        """
        if len(text) <= max_len:
            return text.strip()
            
        # Try to break at natural points for titles
        natural_breaks = []
        
        # Find parentheses completion
        paren_count = 0
        for i, char in enumerate(text[:max_len + 20]):  # Look a bit beyond max_len
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0 and i < max_len + 15:
                    # Complete parentheses found
                    potential_end = i + 1
                    # Check if next chars are part of title or start of content
                    next_chars = text[potential_end:potential_end + 10].strip()
                    if (next_chars.startswith(('You can', 'This ', 'The ', 'To ', 'It ')) or
                        next_chars == '' or next_chars[0].islower()):
                        natural_breaks.append(potential_end)
                        break
        
        # Find colon breaks
        if ':' in text[:max_len]:
            colon_pos = text.find(':', 0, max_len)
            if len(text[:colon_pos].strip()) >= 15:
                natural_breaks.append(colon_pos)
        
        # Find dash breaks
        for char in ['-', '–']:
            if char in text[:max_len]:
                pos = text.find(char, 0, max_len)
                truncated = text[:pos].strip()
                if len(truncated) >= 15:
                    natural_breaks.append(pos)
        
        # Use the best natural break
        if natural_breaks:
            best_break = min(natural_breaks)
            result = text[:best_break].strip()
            if len(result) >= 10:
                return result
        
        # Fallback: Break at word boundary
        truncated = text[:max_len]
        last_space = truncated.rfind(' ')
        if last_space > max_len * 0.7:  # Don't break too early
            return truncated[:last_space].strip()
            
        return truncated.strip() + "..."

    def _analyze_content_type(self, text: str) -> str:
        """
        Return general content type - domain agnostic approach.
        """
        return 'general'
