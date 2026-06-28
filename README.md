# RelationshipPath AI - OpenAI Powered Version

## Run locally / Codespaces

1. Make sure your OpenAI API key is available as an environment variable:

```bash
echo $OPENAI_API_KEY
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Optional model override

The app uses `gpt-5.5` by default. To use a different model, set:

```bash
export OPENAI_MODEL="gpt-5.5"
```

## Streamlit Cloud

Add `OPENAI_API_KEY` to Streamlit app secrets, not to GitHub.
