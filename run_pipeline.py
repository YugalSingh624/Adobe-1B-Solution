from src.embedder import PersonaEmbedder
from src.document_reader import DocumentReader
from src.section_selector import SectionSelector
from src.refine_subsections import SubsectionRefiner
from src.utils import (
    generate_output_json, 
    save_json_output
)

from datetime import datetime
import os
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- ENHANCED CONFIGURABLE RUNTIME INPUTS ---
def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from JSON file, with fallbacks to defaults.
    """
    default_config = {
        "persona": "HR professional",
        "job": "Create and manage fillable forms for onboarding and compliance.",
        "docs_folder": "docs",
        "top_k": 5,
        "max_per_doc": 2,
        "max_sentences": 6,
        "min_sentences": 3,
        "context_window": 2,
        "diversity_weight": 0.3,
        "output_dir": "outputs",
        "save_debug_info": True
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            # Merge with defaults
            config = {**default_config, **user_config}
        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {str(e)}")
            logger.info("Using default configuration")
            config = default_config
    else:
        logger.info("No config file found, using default configuration")
        config = default_config
    
    return config

def validate_inputs(config: dict) -> bool:
    """
    Validate configuration inputs.
    """
    if not os.path.exists(config["docs_folder"]):
        logger.error(f"Documents folder does not exist: {config['docs_folder']}")
        return False
    
    if config["top_k"] <= 0:
        logger.error("top_k must be positive")
        return False
        
    if config["max_per_doc"] <= 0:
        logger.error("max_per_doc must be positive")
        return False
    
    return True

def main():
    """
    Simplified main pipeline with clean output format.
    """
    try:
        # Load configuration (support command line config file)
        import sys
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
        config = load_config(config_file)
        
        # Validate inputs
        if not validate_inputs(config):
            logger.error("Configuration validation failed")
            return
        
        logger.info("Starting document processing pipeline")
        logger.info(f"Persona: {config['persona']}")
        logger.info(f"Job: {config['job']}")
        
        # --- STEP 1: Load Documents ---
        logger.info("Step 1: Loading documents...")
        reader = DocumentReader(
            min_section_length=50,
            max_section_length=3000
        )
        documents = reader.load_documents(config["docs_folder"])
        
        if not documents:
            logger.error("No documents loaded successfully")
            return

        logger.info(f"Loaded {len(documents)} documents with {sum(len(sections) for sections in documents.values())} total sections")
        
        # --- STEP 2: Embed Persona & Job ---
        logger.info("Step 2: Creating persona-job embedding...")
        embedder = PersonaEmbedder(cache_embeddings=True)
        persona_job_embedding = embedder.embed_persona_job(config["persona"], config["job"])
        
        # --- STEP 3: Select Top Sections ---
        logger.info("Step 3: Selecting relevant sections...")
        selector = SectionSelector(embedder)
        top_sections = selector.select_relevant_sections(
            documents,
            persona_job_embedding,
            top_k=config["top_k"],
            max_per_doc=config["max_per_doc"],
            persona=config["persona"],
            job=config["job"],
            diversity_weight=config["diversity_weight"]
        )
        
        if not top_sections:
            logger.error("No sections selected")
            return
        
        logger.info(f"Selected {len(top_sections)} sections")
        
        # --- STEP 4: Refine Subsections ---
        logger.info("Step 4: Refining subsections...")
        refiner = SubsectionRefiner(embedder)
        refined_sections = refiner.refine(
            top_sections, 
            persona_job_embedding,
            max_sentences=config["max_sentences"],
            min_sentences=config["min_sentences"],
            context_window=config["context_window"]
        )
        
        logger.info(f"Refined {len(refined_sections)} sections")
        
        # --- STEP 5: Generate Clean Output ---
        logger.info("Step 5: Generating output...")
        timestamp = datetime.now().isoformat()
        
        output = generate_output_json(
            documents=list(documents.keys()),
            persona=config["persona"],
            job=config["job"],
            top_sections=top_sections,
            refined_sections=refined_sections,
            timestamp=timestamp
        )
        
        # Save main output
        output_path = save_json_output(output, config["output_dir"])
        
        # Final summary
        logger.info("Pipeline completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Processed {len(documents)} documents, selected {len(top_sections)} sections, refined {len(refined_sections)} sections")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
