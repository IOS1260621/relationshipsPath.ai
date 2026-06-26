# relationshipsPath.ai

RelationshipPath AI is a Streamlit MVP for relationship coaching support.
It is designed for communication help, reflection, message rewriting, and weekly check-ins.
It is not therapy, emergency support, or clinical diagnosis.

## Features

- Safety-first coaching chat with guardrails for harm, abuse, and manipulation requests
- Message rewrite helper (Calm, Loving, Boundary, Apology, Short Text)
- Weekly relationship check-in summary generator
- Session journal with JSON export
- Situation-aware response focus (argument, hurt, trust rebuild, weekly check-in, etc.)

## Requirements

- Python 3.9+
- pip
- dependencies listed in requirements.txt

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python -m streamlit run app.py
```

3. Open the local URL shown by Streamlit in your browser.
If you are using Codespaces, open the forwarded port 8501 URL.

## Project Structure

- app.py: main Streamlit application
- relationship_config.py: keywords, templates, and coaching configuration
- requirements.txt: pinned Python dependencies
- .streamlit/config.toml: Streamlit server settings for Codespaces
- README.md: project documentation
