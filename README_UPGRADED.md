# 🚀 Advanced Document Section Selection System

A sophisticated AI-powered document processing pipeline that intelligently extracts and selects the most relevant content sections from PDF documents based on specific personas and job requirements.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://docker.com)
[![BGE Model](https://img.shields.io/badge/model-BGE--small--en--v1.5-green.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![CPU Optimized](https://img.shields.io/badge/optimized-CPU--only-orange.svg)](https://pytorch.org)

## 🌟 Overview

This system uses advanced natural language processing and semantic similarity matching to analyze large collections of PDF documents and extract the most relevant sections for specific user personas and job roles. **Successfully tested** with diverse content domains including culinary, travel, technical documentation, and business workflows.

### ✨ Key Features

- 🎯 **Persona-Aware Content Selection**: Tailors content extraction based on specific job roles and requirements
- 🚀 **Universal Domain Support**: Works across food/recipes, business/Adobe Acrobat, travel, and mixed domains
- 🧠 **Advanced Semantic Analysis**: Uses BGE-small-en-v1.5 embeddings for intelligent content matching
- 🔍 **Smart Content Filtering**: Eliminates instruction fragments, navigation elements, and low-quality sections
- 📊 **Quality Assurance**: Multi-layered content validation and confidence scoring
- 🌐 **Multi-language Support**: Handles diverse content with multilingual capabilities
- ⚡ **Performance Optimization**: Intelligent caching and batch processing
- 📈 **Comprehensive Analytics**: Detailed processing metrics and performance monitoring
- 🐋 **Docker Support**: Production-ready containerized deployment
- 🔧 **Flexible Configuration**: Custom persona and job configuration support

---

## 🏗️ System Architecture

### Core Components

```
├── run_pipeline.py              # Main execution pipeline
├── config.json                  # Default configuration file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Multi-service orchestration
├── entrypoint.sh               # Container startup script
├── src/
│   ├── document_reader.py       # PDF extraction and preprocessing
│   ├── embedder.py             # Semantic embedding generation
│   ├── section_selector.py     # Core selection logic with persona awareness
│   ├── refine_subsections.py   # Content refinement and optimization
│   ├── utils.py                # Utility functions and output generation
│   ├── multilingual_support.py # Multi-language processing
│   ├── batch_processor.py      # Batch processing capabilities
│   ├── advanced_confidence.py  # Confidence scoring algorithms
│   └── content_quality_analyzer.py # Content quality assessment
├── docs/                       # Source PDF documents (31 files included)
├── models/                     # BGE embedding model
│   └── bge-small-en-v1.5/     # Pre-downloaded BGE model
├── outputs/                    # Generated results with timestamps
└── cache/                      # Processing cache for performance
```

### Processing Pipeline

1. **Document Loading**: Extracts text and structure from PDF documents (supports 31+ files)
2. **Persona Analysis**: Analyzes job requirements and persona characteristics
3. **Semantic Embedding**: Generates vector representations using BGE-small-en-v1.5
4. **Intelligent Filtering**: Removes low-quality sections (filters ~65% of irrelevant content)
5. **Relevance Scoring**: Ranks sections based on persona-job semantic alignment
6. **Content Refinement**: Optimizes selected sections for quality and coherence
7. **Output Generation**: Produces structured JSON results with complete metadata

---

## 🚀 Getting Started

### 📋 Prerequisites

Before you begin, ensure you have one of the following setups:

**Option A: Docker Setup (Recommended for Easy Setup)**
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop/))
- Docker Compose (included with Docker Desktop)
- 4GB+ RAM available for Docker

**Option B: Python Setup (For Development)**
- Python 3.8 or higher ([Download here](https://www.python.org/downloads/))
- pip (package installer for Python)
- 4GB+ RAM available

### 🚀 Quick Start

#### Method 1: Docker (Recommended for Beginners)

**Step 1: Clone or Download the Project**
```bash
# If you have git installed
git clone <repository-url>
cd Adobe-1B-solution

# OR download and extract the ZIP file to your desired location
```

**Step 2: Start the Application**
```bash
# Navigate to the project directory
cd path/to/Adobe-1B-solution

# Start the Docker container
docker compose up -d
```

**Step 3: Verify Everything is Working**
```bash
# Check if the container is running
docker compose ps

# View the startup logs
docker compose logs
```
Expected output should show:
```
🚀 Advanced Document Section Selection System (CPU-ONLY)
✅ BGE model: Ready
✅ Application: Ready
Ready to process documents! 🎯
```

**Step 4: Run Your First Analysis**
```bash
# Run with the default configuration (Executive Chef persona)
docker compose exec document-processor python run_pipeline.py
```

**Step 5: View Your Results**
```bash
# List all output files
docker compose exec document-processor ls -la outputs/

# View the latest results (replace with actual filename)
docker compose exec document-processor cat outputs/output_YYYYMMDD_HHMMSS.json
```

**Step 6: Copy Results to Your Computer**
```bash
# Copy all results to your local machine
docker cp advanced-doc-selector-cpu-container:/app/outputs/ ./my_results/
```

#### Method 2: Direct Python Installation

**Step 1: Download and Setup**
```bash
# Navigate to the project directory
cd path/to/Adobe-1B-solution

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Step 2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

**Step 3: Verify BGE Model**
```bash
# Check if the BGE model directory exists
# The model should be in: models/bge-small-en-v1.5/
# If missing, the application will download it automatically on first run
```

**Step 4: Run Your First Analysis**
```bash
# Run with default configuration
python run_pipeline.py

# OR specify a custom configuration
python run_pipeline.py your_config.json
```

**Step 5: Check Results**
```bash
# View generated results
dir outputs\  # Windows
ls outputs/   # macOS/Linux
```

### 🎯 Testing Different Personas

The system comes with examples for different use cases. Try these:

**For Cooking/Recipe Content:**
```bash
# Docker
docker compose exec document-processor python run_pipeline.py config.json

# Python
python run_pipeline.py config.json
```

**For Travel Content (create custom config):**
```json
{
    "persona": "Travel Blogger",
    "job": "Create engaging travel content about South of France destinations, culture, and cuisine for blog readers planning their vacation",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

**For Technical Documentation:**
```json
{
    "persona": "Software Trainer",
    "job": "Develop comprehensive Adobe Acrobat training materials covering all features from basic to advanced",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

### 🛠️ Common Commands

**Docker Users:**
```bash
# Start the application
docker compose up -d

# Run with default settings
docker compose exec document-processor python run_pipeline.py

# Run with custom config
docker compose exec document-processor python run_pipeline.py my_config.json

# Copy custom config to container
docker cp my_config.json advanced-doc-selector-cpu-container:/app/

# View logs
docker compose logs -f

# Stop the application
docker compose down

# Restart the application
docker compose restart
```

**Python Users:**
```bash
# Activate virtual environment (if using)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run analysis
python run_pipeline.py

# Run with custom config
python run_pipeline.py my_config.json

# Check if everything is working
python verify_cpu_only.py
```

---

## 📋 Configuration Guide

### Creating Custom Persona Configurations

Create JSON files for different use cases:

#### Executive Chef Configuration
```json
{
    "persona": "Executive Chef",
    "job": "Create diverse seasonal menu offerings for a high-end restaurant featuring breakfast, lunch, and dinner options with creative side dishes and regional specialties.",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

#### Travel Marketing Manager Configuration
```json
{
    "persona": "Travel Marketing Manager",
    "job": "Develop comprehensive marketing materials for South of France tourism, highlighting attractions, cultural experiences, dining, and accommodation options for luxury and budget travelers.",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

#### IT Training Specialist Configuration  
```json
{
    "persona": "IT Training Specialist",
    "job": "Design comprehensive Adobe Acrobat training curriculum covering document creation, editing, sharing, e-signatures, and advanced features like generative AI for corporate users.",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `persona` | Professional role/identity | Required | String |
| `job` | Specific job requirements and focus | Required | String |
| `docs_folder` | Directory containing PDF documents | "docs" | String |
| `top_k` | Number of sections to select | 5 | 1-20 |
| `max_per_doc` | Max sections per document | 2 | 1-10 |
| `max_sentences` | Max sentences per section | 6 | 3-15 |
| `min_sentences` | Min sentences per section | 3 | 1-10 |
| `context_window` | Context sentences around selection | 2 | 1-5 |
| `diversity_weight` | Content diversity factor | 0.3 | 0.0-1.0 |
| `output_dir` | Output directory | "outputs" | String |

---

## 💼 Validated Use Cases & Results

### 🍽️ Executive Chef Results
**Configuration**: High-end restaurant menu development  
**Selected Content**:
- Breakfast: Smoothie Bowl recipes with detailed ingredients
- Dinner: Beef Bourguignon and Mushroom Risotto with techniques
- Lunch: Veggie Sushi Rolls with step-by-step instructions
- Context: Luxurious hotels with Michelin-starred restaurants

### 🌍 Travel Marketing Manager Results  
**Configuration**: South of France tourism marketing
**Selected Content**:
- Regional Overview: Comprehensive South of France introduction
- Accommodations: Budget-friendly and luxury hotel options
- Cultural Context: Local attractions and experiences
- Dining: Regional cuisine and restaurant recommendations

### 💻 IT Training Specialist Results
**Configuration**: Adobe Acrobat corporate training
**Selected Content**:
- Generative AI: Advanced AI features in Acrobat
- Document Creation: Best practices for PDF creation
- File Conversion: Menu-based conversion processes
- Skills Assessment: Exporting capabilities testing

### 📊 Performance Metrics
- **Documents Processed**: 31 PDF files
- **Total Sections Analyzed**: 446 sections
- **Content Filtering Efficiency**: ~65% irrelevant content filtered
- **Processing Time**: ~2-3 minutes per run (Docker)
- **Selection Accuracy**: High relevance across all tested personas

---

## 📂 Available Document Corpus

The system includes a diverse collection of 31 PDF documents:

### Food & Culinary (9 documents)
- Breakfast Ideas.pdf
- Lunch Ideas.pdf  
- Dinner Ideas - Mains (3 files)
- Dinner Ideas - Sides (4 files)

### Travel & Tourism (7 documents)
- South of France - Cities.pdf
- South of France - Cuisine.pdf
- South of France - History.pdf
- South of France - Restaurants and Hotels.pdf
- South of France - Things to Do.pdf
- South of France - Tips and Tricks.pdf
- South of France - Traditions and Culture.pdf

### Technical Documentation (13 documents)
- Learn Acrobat - Create and Convert (2 files)
- Learn Acrobat - Edit (2 files)
- Learn Acrobat - Export (2 files)
- Learn Acrobat - Fill and Sign.pdf
- Learn Acrobat - Generative AI (2 files)
- Learn Acrobat - Request e-signatures (2 files)
- Learn Acrobat - Share (2 files)

### Assessment & References (2 documents)
- Test Your Acrobat Exporting Skills.pdf  
- The Ultimate PDF Sharing Checklist.pdf

---

## 🐋 Docker Deployment

### Container Architecture
- **Base Image**: Python 3.11-slim (CPU-optimized)
- **Multi-stage Build**: Optimized for production deployment
- **Security**: Non-root user execution
- **Model**: Pre-installed BGE-small-en-v1.5
- **Networking**: Isolated container network

### Docker Commands

**Build and Start**:
```bash
docker compose up -d
```

**View Logs**:
```bash
docker compose logs -f
```

**Execute Commands**:
```bash
# Run with default config
docker compose exec document-processor python run_pipeline.py

# Run with custom config
docker compose exec document-processor python run_pipeline.py custom_config.json

# Access container shell
docker compose exec document-processor bash
```

**Copy Files**:
```bash
# Copy config to container
docker cp my_config.json advanced-doc-selector-cpu-container:/app/

# Copy results from container
docker cp advanced-doc-selector-cpu-container:/app/outputs/ ./local_outputs/
```

**Stop and Cleanup**:
```bash
docker compose down
```

### Production Deployment
```bash
# Build for production
docker compose build --no-cache

# Run in production mode
docker compose up -d --restart unless-stopped
```

---

## 📊 Output Format

### JSON Structure
```json
{
    "metadata": {
        "input_documents": ["list", "of", "processed", "files"],
        "persona": "Your Persona",
        "job_to_be_done": "Your specific job requirements",
        "processing_timestamp": "2025-07-28T13:18:32.457432"
    },
    "extracted_sections": [
        {
            "document": "Source_Document.pdf",
            "section_title": "Relevant Section Title",
            "importance_rank": 1,
            "page_number": 7
        }
    ],
    "subsection_analysis": [
        {
            "document": "Source_Document.pdf",
            "refined_text": "Optimized content text with context...",
            "page_number": 7
        }
    ]
}
```

### Output Features
- **Timestamp-based Filenames**: `output_YYYYMMDD_HHMMSS.json`
- **Complete Metadata**: Persona, job, processing details
- **Ranked Results**: Sections ordered by relevance
- **Source Attribution**: Document and page references
- **Refined Content**: Optimized text with context

---

## 🛠️ Troubleshooting

### 🆘 Beginner-Friendly Solutions

**❓ "I don't know if Docker is installed"**
```bash
# Test if Docker is installed
docker --version

# Test if Docker Compose is available
docker compose version

# If not installed, download from: https://www.docker.com/products/docker-desktop/
```

**❓ "The container won't start"**
```bash
# Make sure Docker Desktop is running first
# Then try:
docker compose down
docker compose up -d

# Check what went wrong:
docker compose logs
```

**❓ "I get 'command not found' errors"**
```bash
# Make sure you're in the right directory
cd path/to/Adobe-1B-solution

# Check if the files exist
dir  # Windows
ls   # macOS/Linux

# You should see files like: docker-compose.yml, requirements.txt, run_pipeline.py
```

**❓ "Python says 'module not found'"**
```bash
# Make sure you installed the requirements
pip install -r requirements.txt

# If using virtual environment, make sure it's activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

**❓ "The output folder is empty"**
```bash
# Check if the process completed
# Look for success messages in the output

# For Docker users:
docker compose exec document-processor ls -la outputs/

# For Python users:
dir outputs\  # Windows
ls outputs/   # macOS/Linux
```

**❓ "I want to use my own documents"**
```bash
# Replace files in the docs/ folder with your PDF files
# Make sure they're readable PDF files (not scanned images)

# For Docker users, copy files to container:
docker cp my_document.pdf advanced-doc-selector-cpu-container:/app/docs/
```

### Advanced Issues

**1. Container Startup Issues**
```bash
# Check container status
docker compose ps

# View detailed logs
docker compose logs document-processor

# Restart container
docker compose restart
```

**2. Missing BGE Model**
```bash
# Verify model exists in container
docker compose exec document-processor ls -la models/bge-small-en-v1.5/

# If missing, rebuild container
docker compose build --no-cache
```

**3. Configuration File Errors**
```bash
# Validate JSON syntax
python -m json.tool your_config.json

# Check required fields
# Ensure "persona" and "job" are specified
```

**4. Permission Issues**
```bash
# Fix file permissions (Linux/Mac)
chmod 644 your_config.json

# For Windows, ensure files are accessible
```

### Performance Optimization

**Memory Usage**:
- System uses ~2-4GB RAM during processing
- BGE model requires ~400MB memory
- Consider increasing Docker memory limits if needed

**Processing Speed**:
- First run: ~3-4 minutes (model loading + processing)
- Subsequent runs: ~2-3 minutes (cached embeddings)
- 31 documents → 446 sections → 5 selected sections

---

## 🔧 Development & Customization

### Adding New Documents
1. Place PDF files in `docs/` folder
2. Ensure files are readable and contain extractable text
3. Run pipeline - new documents are automatically detected

### Custom Embedder Models
Currently supports BGE-small-en-v1.5. To use different models:
1. Update `src/embedder.py`
2. Modify model path in configuration
3. Rebuild Docker container if needed

### Extending Persona Types
The system is domain-agnostic. Test with personas like:
- **Healthcare Professional**: Medical document analysis
- **Legal Consultant**: Contract and legal document review  
- **Financial Analyst**: Business and financial report processing
- **Education Specialist**: Academic content curation

---

## 📈 System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+ available
- **Storage**: 2GB+ free space
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### Recommended Requirements  
- **CPU**: 4+ cores
- **RAM**: 8GB+ available
- **Storage**: 5GB+ free space
- **Network**: Internet connection for initial setup

### Docker Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ Docker memory allocation

---

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Clone your fork locally
3. Install development dependencies
4. Create feature branch
5. Test with provided document corpus
6. Submit pull request

### Testing New Personas
1. Create configuration file
2. Test with existing document corpus
3. Validate output quality and relevance
4. Document results and add to examples

---

## 📄 License

This project is developed for Adobe's Advanced Document Processing Challenge. Please refer to the original challenge guidelines for usage and distribution terms.

---

---

## 📚 Quick Reference Card

### 🐋 Docker Users (Copy-Paste Ready)

```bash
# 1. Start the application
docker compose up -d

# 2. Run analysis with default settings
docker compose exec document-processor python run_pipeline.py

# 3. Copy custom config and run
docker cp my_config.json advanced-doc-selector-cpu-container:/app/
docker compose exec document-processor python run_pipeline.py my_config.json

# 4. Get your results
docker cp advanced-doc-selector-cpu-container:/app/outputs/ ./my_results/

# 5. Stop when done
docker compose down
```

### 🐍 Python Users (Copy-Paste Ready)

```bash
# 1. Setup (one time only)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 2. Run analysis
python run_pipeline.py

# 3. Run with custom config
python run_pipeline.py my_config.json

# 4. Check results
dir outputs\  # Windows
# ls outputs/   # macOS/Linux
```

### 📝 Sample Configuration Template

```json
{
    "persona": "Your Professional Role",
    "job": "Describe what you want to achieve with the documents",
    "top_k": 5,
    "max_per_doc": 2,
    "max_sentences": 6,
    "min_sentences": 3,
    "context_window": 2,
    "diversity_weight": 0.3,
    "output_dir": "outputs"
}
```

---

## 🆘 Support

### Quick Help
- **Configuration Issues**: Check JSON syntax and required fields
- **Docker Problems**: Ensure Docker is running and has sufficient resources
- **Performance Issues**: Monitor system resources and consider optimization
- **Output Quality**: Experiment with different `diversity_weight` values

### Advanced Support
For technical issues, performance optimization, or custom implementations, please refer to the comprehensive documentation in the `docs/` folder or contact the development team.

---

## 🎉 Acknowledgments

- **BGE Model**: BAAI's BGE-small-en-v1.5 for semantic embeddings
- **Docker Community**: For containerization best practices
- **PyTorch Team**: For CPU-optimized machine learning framework
- **Open Source Community**: For various dependencies and tools

---

**🚀 Ready to process your documents with AI-powered intelligence!** 

Whether you're a chef looking for recipe inspiration, a travel marketer crafting destination guides, or an IT trainer developing software curricula, this system adapts to your specific needs and delivers precisely relevant content from your document collections.
