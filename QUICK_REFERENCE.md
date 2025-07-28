# 🎯 Quick Reference Guide

## 📋 Common Commands

### Docker Usage (Recommended)

**Start System:**
```bash
docker compose up -d
```

**Run Default Configuration:**
```bash
docker compose exec document-processor python run_pipeline.py
```

**Run with Custom Persona:**
```bash
# 1. Copy your config file
docker cp your_config.json advanced-doc-selector-cpu-container:/app/

# 2. Run with config
docker compose exec document-processor python run_pipeline.py your_config.json

# 3. View results  
docker compose exec document-processor cat outputs/output_*.json
```

### Python Direct Usage

**Install and Run:**
```bash
pip install -r requirements.txt
python run_pipeline.py
python run_pipeline.py your_config.json
```

---

## 🧑‍🍳 Persona Templates

### Executive Chef
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

### Travel Marketing Manager  
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

### IT Training Specialist
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

---

## 🛠️ Parameter Quick Guide

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `top_k` | Sections to select | 3-8 |
| `max_per_doc` | Max per document | 1-3 |
| `max_sentences` | Content length | 4-10 |
| `diversity_weight` | Content variety | 0.2-0.5 |

---

## 📊 Expected Performance

- **Documents**: 31 PDFs processed
- **Processing Time**: 2-3 minutes  
- **Content Filtering**: ~65% irrelevant content removed
- **Output Quality**: High relevance for tested personas

---

## 🆘 Quick Troubleshooting

**Container Won't Start:**
```bash
docker compose restart
docker compose logs
```

**Config File Issues:**
```bash
python -m json.tool your_config.json  # Validate JSON
```

**No Results:**
- Check `persona` and `job` are specified
- Ensure PDF files exist in `docs/` folder
- Verify BGE model is present
