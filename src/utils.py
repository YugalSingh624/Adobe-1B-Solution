# src/utils.py

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR artifacts from text, including incorrectly split words and formatting issues.
    """
    if not text:
        return text
    
    # Fix escaped quotes and formatting issues
    text = text.replace('\\"', '"')  # Fix escaped quotes
    text = text.replace('"', '"').replace('"', '"')  # Normalize quotes
    
    # Handle truncated titles that end with incomplete patterns
    truncation_patterns = [
        r',\s*or\s*"[^"]*$',  # ", or "Le -> remove incomplete quote reference
        r',\s*or\s*$',        # ", or -> remove hanging conjunction
        r',\s*and\s*$',       # ", and -> remove hanging conjunction  
        r'\s*"[^"]*$',        # hanging quote at end
        r'\s*\([^)]*$',       # hanging parenthesis
        r'\s*\[[^\]]*$',      # hanging bracket
    ]
    
    cleaned_text = text
    for pattern in truncation_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    # Common OCR word splitting fixes
    ocr_fixes = {
        'ther mal': 'thermal',
        'comfor t': 'comfort',
        't oys': 'toys',
        'Outdo or': 'Outdoor',
        'outdo or': 'outdoor',
        'garlic ky': 'garlicky',
        'wit h': 'with',
        'yo ur': 'your',
        'ho me': 'home',
        'tim e': 'time',
        'som e': 'some',
        'mor e': 'more',
        'wher e': 'where',
        'ther e': 'there',
        'her e': 'here',
        'befor e': 'before',
        'abov e': 'above',
        'belo w': 'below',
        'aroun d': 'around',
        'throug h': 'through',
        'withou t': 'without',
        'ver y': 'very',
        'an d': 'and',
        'the m': 'them',
        'fro m': 'from',
        'int o': 'into',
        'ove r': 'over',
        'unde r': 'under',
        'afte r': 'after'
    }
    
    # Apply OCR fixes
    for incorrect, correct in ocr_fixes.items():
        cleaned_text = cleaned_text.replace(incorrect, correct)
    
    # Fix excessive spacing (multiple spaces to single space)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Fix spacing around punctuation
    cleaned_text = re.sub(r'\s+([,.!?;:])', r'\1', cleaned_text)
    cleaned_text = re.sub(r'([,.!?;:])\s+', r'\1 ', cleaned_text)
    
    return cleaned_text.strip()


def generate_output_json(
    documents: List[str],
    persona: str,
    job: str,
    top_sections: List[Dict],
    refined_sections: List[Dict],
    timestamp: str,
    performance_metrics: Dict = None,
    cache_statistics: Dict = None
) -> Dict:
    """
    Generate clean output JSON in the specified format.
    """
    extracted = []
    for i, section in enumerate(top_sections, 1):
        # Use the actual title from the document section - this has already been properly filtered and extracted
        actual_title = section.get("title", "Untitled Section")
        
        # Clean OCR artifacts from title
        cleaned_title = clean_ocr_text(actual_title)
        
        section_entry = {
            "document": section["document"],
            "section_title": cleaned_title,
            "importance_rank": i,
            "page_number": section["page_number"]
        }
        extracted.append(section_entry)

    refined = []
    for refined_sec in refined_sections:
        # Clean OCR artifacts from refined text
        cleaned_refined_text = clean_ocr_text(refined_sec["refined_text"])
        
        refined_entry = {
            "document": refined_sec["document"],
            "refined_text": cleaned_refined_text,
            "page_number": refined_sec["page_number"]
        }
        refined.append(refined_entry)

    # Clean output format matching the exact structure requested
    output = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted,
        "subsection_analysis": refined
    }

    return output


def save_json_output(output_dict: Dict, output_dir: str = "outputs") -> str:
    """
    Save JSON output with clean logging.
    
    Returns:
        Path to the saved file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Validate output structure
        _validate_output_structure(output_dict)

        # Save JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False, default=_json_serializer)

        return filepath
        
    except Exception as e:
        logger.error(f"Error saving output: {str(e)}")
        raise


def _validate_output_structure(output_dict: Dict) -> None:
    """
    Validate the output dictionary structure.
    """
    required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
    for key in required_keys:
        if key not in output_dict:
            raise ValueError(f"Missing required key in output: {key}")
    
    if not isinstance(output_dict["extracted_sections"], list):
        raise ValueError("extracted_sections must be a list")
    
    if not isinstance(output_dict["subsection_analysis"], list):
        raise ValueError("subsection_analysis must be a list")


def _log_output_summary(output_dict: Dict, filepath: str) -> None:
    """
    Log summary statistics about the output.
    """
    metadata = output_dict.get("metadata", {})
    extracted = output_dict.get("extracted_sections", [])
    refined = output_dict.get("subsection_analysis", [])
    
    logger.info(f"Output Summary for {filepath}:")
    logger.info(f"  - Persona: {metadata.get('persona', 'Unknown')}")
    logger.info(f"  - Job: {metadata.get('job_to_be_done', 'Unknown')}")
    logger.info(f"  - Documents processed: {len(metadata.get('input_documents', []))}")
    logger.info(f"  - Sections extracted: {len(extracted)}")
    logger.info(f"  - Sections refined: {len(refined)}")
    
    if metadata.get("average_confidence"):
        logger.info(f"  - Average confidence: {metadata['average_confidence']:.3f}")


def _json_serializer(obj: Any) -> Any:
    """
    Enhanced JSON serializer for complex objects.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (int, float)):
        return float(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def clean_text(text: str) -> str:
    """
    Enhanced text cleaning with better artifact removal.
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction artifacts
    artifact_replacements = [
        ("•", ""),
        ("–", "-"),
        ("—", "-"),
        (""", '"'),
        (""", '"'),
        ("'", "'"),
        ("'", "'"),
        ("ﬀ", "ff"),
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
        ("ﬃ", "ffi"),
        ("ﬄ", "ffl"),
    ]
    
    for old, new in artifact_replacements:
        text = text.replace(old, new)
    
    # Remove excessive punctuation
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'-{2,}', '-', text)      # Multiple dashes to single
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s*([,;:!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    
    # Remove trailing/leading whitespace
    text = text.strip()
    
    return text


def _extract_proper_title_from_text(text: str, max_len: int = 100) -> str:
    """
    Extract meaningful titles from PDF text content, specifically designed for Adobe Acrobat documentation.
    """
    if not text:
        return "Untitled Section"
    
    # Clean the text first
    text = text.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Strategy 1: Look for clear section headers in the first few lines
    for line in lines[:3]:
        clean_line = line.strip()
        
        # Remove common PDF artifacts and bullet points
        clean_line = re.sub(r'^[•\-\*\d+\.\)]\s*', '', clean_line)
        clean_line = re.sub(r'^\W+', '', clean_line)
        clean_line = clean_line.strip()
        
        # Check if this looks like a good title
        if (8 <= len(clean_line) <= max_len and 
            clean_line[0].isupper() and
            not clean_line.endswith('.') and
            not clean_line.lower().startswith(('the ', 'it ', 'this ', 'there ', 'you ', 'when ', 'if ', 'note:', 'tip:'))):
            
            # Additional checks for Adobe Acrobat content
            if any(keyword in clean_line.lower() for keyword in [
                'pdf', 'acrobat', 'form', 'sign', 'fill', 'create', 'convert', 'edit', 'export', 'share', 'request'
            ]):
                return clean_line
            
            # Generic good title characteristics
            word_count = len(clean_line.split())
            if 2 <= word_count <= 10:
                return clean_line
    
    # Strategy 2: Look for action-oriented sentences
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences[:3]:
        sentence = sentence.strip()
        if (10 <= len(sentence) <= max_len and
            sentence[0].isupper()):
            
            # Look for instructional patterns common in Adobe docs
            if any(pattern in sentence.lower() for pattern in [
                'to ', 'how to', 'you can', 'create', 'use', 'choose', 'select', 'click', 'open'
            ]):
                # Clean up the sentence to make it title-like
                if sentence.lower().startswith('to '):
                    title = sentence[3:].strip()
                    title = title[0].upper() + title[1:] if title else sentence
                    return title if len(title) <= max_len else sentence
                return sentence
    
    # Strategy 3: Extract key phrases from first paragraph
    first_paragraph = text.split('\n\n')[0] if '\n\n' in text else text[:200]
    
    # Look for Adobe Acrobat specific patterns
    adobe_patterns = [
        r'(Create|Convert|Fill|Sign|Edit|Export|Share|Request)\s+[A-Za-z\s]{5,40}',
        r'(PDF|Acrobat)\s+[A-Za-z\s]{5,40}',
        r'How to\s+[A-Za-z\s]{5,40}',
        r'Using\s+[A-Za-z\s]{5,40}'
    ]
    
    for pattern in adobe_patterns:
        matches = re.findall(pattern, first_paragraph, re.IGNORECASE)
        if matches:
            title = matches[0].strip()
            if len(title) <= max_len:
                return title
    
    # Strategy 4: Fallback - use first meaningful sentence
    words = text.split()[:15]
    title = ' '.join(words)
    
    # Clean up the title
    title = re.sub(r'^[•\-\*\d+\.\)]\s*', '', title)
    
    if len(title) > max_len:
        # Find good break point
        truncated = title[:max_len-3]
        last_space = truncated.rfind(' ')
        if last_space > max_len // 2:
            title = truncated[:last_space] + "..."
        else:
            title = truncated + "..."
    
    return title if title else "Untitled Section"
def get_project_statistics(documents: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Generate project statistics for analysis and debugging.
    """
    total_sections = sum(len(sections) for sections in documents.values())
    total_documents = len(documents)
    
    # Content type distribution
    content_types = {}
    word_counts = []
    
    for doc_name, sections in documents.items():
        for section in sections:
            content_type = section.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            word_count = section.get("word_count", 0)
            if word_count > 0:
                word_counts.append(word_count)
    
    stats = {
        "total_documents": total_documents,
        "total_sections": total_sections,
        "average_sections_per_document": total_sections / total_documents if total_documents > 0 else 0,
        "content_type_distribution": content_types,
        "word_count_stats": {
            "average": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "total": sum(word_counts)
        }
    }
    
    return stats


def create_debug_output(
    documents: Dict[str, List[Dict]],
    top_sections: List[Dict],
    refined_sections: List[Dict],
    persona: str,
    job: str
) -> Dict[str, Any]:
    """
    Create debug output for analysis and troubleshooting.
    """
    debug_info = {
        "persona": persona,
        "job": job,
        "timestamp": datetime.now().isoformat(),
        "project_stats": get_project_statistics(documents),
        "selection_analysis": {
            "total_candidates": sum(len(sections) for sections in documents.values()),
            "selected_count": len(top_sections),
            "selection_rate": len(top_sections) / sum(len(sections) for sections in documents.values()) if documents else 0,
            "document_distribution": {}
        },
        "refinement_analysis": {
            "sections_refined": len(refined_sections),
            "average_confidence": 0,
            "refinement_methods": {}
        }
    }
    
    # Analyze document distribution in selection
    for section in top_sections:
        doc = section["document"]
        debug_info["selection_analysis"]["document_distribution"][doc] = \
            debug_info["selection_analysis"]["document_distribution"].get(doc, 0) + 1
    
    # Analyze refinement
    if refined_sections:
        confidences = [s.get("confidence_score", 0) for s in refined_sections if s.get("confidence_score")]
        if confidences:
            debug_info["refinement_analysis"]["average_confidence"] = sum(confidences) / len(confidences)
        
        for section in refined_sections:
            method = section.get("refinement_method", "unknown")
            debug_info["refinement_analysis"]["refinement_methods"][method] = \
                debug_info["refinement_analysis"]["refinement_methods"].get(method, 0) + 1
    
    return debug_info
    return text
