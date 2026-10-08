# 📄 AI Resume ATS Checker

Upload your resume (PDF, DOCX or TXT) and get:

- An **ATS score** out of 100 (70% AI evaluation + 30% rule-based checks)
- A score breakdown: keywords, formatting, impact, clarity
- Strengths and weaknesses
- Prioritized, specific **improvement suggestions**
- Before/after **bullet rewrite examples**
- **Missing keywords** (paste a job description for best results)
- A downloadable JSON report

Built with [Streamlit](https://streamlit.io) and the Google Gemini API.

## How it works

1. Text is extracted from your file (`pypdf` / `python-docx`).
2. Rule-based checks run locally: contact info, standard sections, length, action verbs, metrics, bullets.
3. Gemini evaluates the resume (optionally against a job description) and returns structured JSON.
4. The two scores are blended into the final ATS score.

> No ATS publishes its exact algorithm, so this score is an estimate of ATS-readiness, not a guarantee.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/app/apikey>, then either:

- paste it into the app sidebar, **or**
- set it as an environment variable:
  - macOS/Linux: `export GEMINI_API_KEY="your_key"`
  - Windows (PowerShell): `$env:GEMINI_API_KEY="your_key"`
- or create `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your_key"
  ```

Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | How | Default |
|---|---|---|
| API key | Sidebar, `GEMINI_API_KEY` env var, or Streamlit secrets | none |
| Model | Sidebar or `GEMINI_MODEL` env var | `gemini-flash-latest` |

`gemini-flash-latest` always points to Google's current Flash model. Google retires older model names over time; if you see a "model not found" error, change the model name in the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

## Project structure

```
├── app.py            # Streamlit app
├── requirements.txt  # Dependencies
├── README.md
└── .gitignore
```

## Limitations

- Scanned/image-only PDFs can't be read (an ATS can't read them either).
- Max upload size is 5 MB; only the first ~20,000 characters are analyzed.
- Your resume text is sent to Google's Gemini API. Don't upload documents you aren't comfortable sharing.
- Always verify AI rewrites; only keep claims that are true.

## License

MIT
