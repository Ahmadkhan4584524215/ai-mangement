"""AI Resume ATS Checker - Streamlit + Gemini Flash."""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
MAX_FILE_MB = 5
MAX_CHARS = 20000  # cap text sent to the model

st.set_page_config(page_title="AI Resume ATS Checker", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------------
def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password protected.")
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:  # many resumes use tables for layout
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts).strip()


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()
    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# ----------------------------------------------------------------------------
# Rule-based ATS checks (fast, deterministic, no API call)
# ----------------------------------------------------------------------------
SECTION_PATTERNS = {
    "Experience": r"\b(work\s+experience|experience|employment|professional\s+background)\b",
    "Education": r"\b(education|academic|qualifications)\b",
    "Skills": r"\b(skills|technical\s+skills|core\s+competencies|technologies)\b",
    "Summary": r"\b(summary|objective|profile|about\s+me)\b",
    "Projects": r"\b(projects|portfolio)\b",
}
ACTION_VERBS = {
    "led", "managed", "built", "developed", "designed", "created", "improved",
    "increased", "reduced", "launched", "implemented", "delivered", "optimized",
    "automated", "achieved", "analyzed", "deployed", "architected", "mentored",
    "coordinated", "streamlined", "generated", "established", "drove", "owned",
}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
LINK_RE = re.compile(r"(linkedin\.com|github\.com|portfolio|https?://)", re.I)
NUMBER_RE = re.compile(r"\d+(\.\d+)?\s?(%|k\b|m\b|x\b|\+)|\$\s?\d+|\b\d{2,}\b", re.I)


def rule_based_checks(text: str) -> dict:
    """Return a 0-100 score plus a list of (label, passed, detail) checks."""
    lower = text.lower()
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    checks = []  # (label, passed, detail, points_earned, points_max)

    def add(label, passed, detail, pts):
        checks.append((label, passed, detail, pts if passed else 0, pts))

    add("Email address found", bool(EMAIL_RE.search(text)), "Recruiters and ATS need a contact email.", 8)
    add("Phone number found", bool(PHONE_RE.search(text)), "Add a phone number in the header.", 7)
    add("LinkedIn / GitHub / portfolio link", bool(LINK_RE.search(text)), "Links add credibility and are parsed by most ATS.", 5)

    for section, pattern in SECTION_PATTERNS.items():
        pts = 10 if section in ("Experience", "Education", "Skills") else 3
        found = bool(re.search(pattern, lower))
        add(f"'{section}' section", found, f"Use a clear, standard '{section}' heading.", pts)

    add("Good length (250-1000 words)", 250 <= word_count <= 1000,
        f"Your resume has {word_count} words. Aim for 1-2 pages.", 10)

    verb_hits = sum(1 for w in words if w.lower() in ACTION_VERBS)
    add("Strong action verbs used (5+)", verb_hits >= 5, f"Found {verb_hits}. Start bullets with verbs like 'Led', 'Built'.", 10)

    numbers = len(NUMBER_RE.findall(text))
    add("Quantified achievements (3+ metrics)", numbers >= 3, f"Found about {numbers}. Add %, $, or counts.", 12)

    bullets = len(re.findall(r"^\s*[-•*▪●◦]", text, flags=re.M))
    add("Bullet points used", bullets >= 5, f"Found {bullets}. Bullets are easier for ATS and humans to scan.", 10)

    total = sum(c[4] for c in checks)
    earned = sum(c[3] for c in checks)
    score = round(100 * earned / total) if total else 0
    return {"score": score, "checks": checks, "word_count": word_count}


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and senior technical recruiter.
Evaluate the resume honestly and strictly. Do not inflate scores.
Base every finding ONLY on the resume text provided. Never invent experience.
If a job description is provided, judge keyword match and relevance against it; otherwise judge general ATS-readiness.
Return ONLY valid JSON matching this schema:
{
  "ats_score": <integer 0-100>,
  "score_breakdown": {
    "keywords_and_relevance": <integer 0-100>,
    "formatting_and_structure": <integer 0-100>,
    "impact_and_achievements": <integer 0-100>,
    "clarity_and_grammar": <integer 0-100>
  },
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<string>", ...],
  "weaknesses": ["<string>", ...],
  "missing_keywords": ["<string>", ...],
  "improvements": [
    {"priority": "High|Medium|Low", "section": "<resume section>", "issue": "<what is wrong>", "suggestion": "<specific fix>"}
  ],
  "rewrite_examples": [
    {"original": "<weak bullet taken from the resume>", "improved": "<stronger version, no fabricated facts>"}
  ]
}
Give 5-8 improvements, 3-5 rewrite_examples, and at most 15 missing_keywords."""


def clean_json(raw: str) -> dict:
    """Parse model output into a dict, tolerating code fences or extra text."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError("The AI returned an unreadable response. Please try again.")


def _clamp(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def normalize_result(data: dict) -> dict:
    """Make sure every expected key exists with the right type."""
    bd = data.get("score_breakdown") or {}
    return {
        "ats_score": _clamp(data.get("ats_score")),
        "score_breakdown": {
            "Keywords & relevance": _clamp(bd.get("keywords_and_relevance")),
            "Formatting & structure": _clamp(bd.get("formatting_and_structure")),
            "Impact & achievements": _clamp(bd.get("impact_and_achievements")),
            "Clarity & grammar": _clamp(bd.get("clarity_and_grammar")),
        },
        "summary": str(data.get("summary") or ""),
        "strengths": [str(x) for x in (data.get("strengths") or [])],
        "weaknesses": [str(x) for x in (data.get("weaknesses") or [])],
        "missing_keywords": [str(x) for x in (data.get("missing_keywords") or [])],
        "improvements": [i for i in (data.get("improvements") or []) if isinstance(i, dict)],
        "rewrite_examples": [r for r in (data.get("rewrite_examples") or []) if isinstance(r, dict)],
    }


def analyze_with_gemini(api_key: str, model: str, resume_text: str, job_description: str) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"RESUME:\n\"\"\"\n{resume_text[:MAX_CHARS]}\n\"\"\"\n\n"
    if job_description.strip():
        prompt += f"JOB DESCRIPTION:\n\"\"\"\n{job_description[:MAX_CHARS]}\n\"\"\"\n"
    else:
        prompt += "JOB DESCRIPTION: (not provided - evaluate general ATS-readiness)\n"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    return normalize_result(clean_json(response.text))


def final_score(ai_score: int, rule_score: int) -> int:
    """Blend: AI judgement (70%) + deterministic checks (30%)."""
    return round(0.7 * ai_score + 0.3 * rule_score)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def score_label(score: int) -> str:
    if score >= 80:
        return "🟢 Excellent"
    if score >= 60:
        return "🟡 Good, needs polish"
    if score >= 40:
        return "🟠 Needs work"
    return "🔴 Poor"


def get_api_key(sidebar_key: str) -> str:
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # no secrets file locally
    return os.getenv("GEMINI_API_KEY", "")


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "Your Gemini API key looks invalid or lacks permission. Please check it."
    if "429" in msg or "quota" in low or "rate" in low or "resource_exhausted" in low:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    return f"Something went wrong: {msg}"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.title("📄 AI Resume ATS Checker")
    st.caption("Upload your resume and get an ATS score with specific, actionable improvements.")

    with st.sidebar:
        st.header("⚙️ Settings")
        sidebar_key = st.text_input("Gemini API key", type="password",
                                    help="Get a free key at aistudio.google.com/app/apikey")
        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        st.markdown("---")
        st.markdown("**Privacy:** your resume is sent to Google's Gemini API for analysis and is not stored by this app.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded = st.file_uploader("Upload resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    with col_right:
        job_desc = st.text_area("Job description (optional, improves keyword matching)", height=160,
                                placeholder="Paste the job description here...")

    if st.button("🔍 Analyze Resume", type="primary", disabled=uploaded is None):
        api_key = get_api_key(sidebar_key)
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()

        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is too large. Max size is {MAX_FILE_MB} MB.")
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                text = extract_text(uploaded.name, data)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        if len(text.split()) < 50:
            st.error("Very little text was found. If your PDF is a scanned image, "
                     "an ATS cannot read it either - export a text-based PDF or DOCX instead.")
            st.stop()

        rules = rule_based_checks(text)
        try:
            with st.spinner("Gemini is analyzing your resume..."):
                ai = analyze_with_gemini(api_key, model.strip() or DEFAULT_MODEL, text, job_desc)
        except Exception as e:
            st.error(friendly_error(e))
            st.stop()

        st.session_state["result"] = {"ai": ai, "rules": rules}

    result = st.session_state.get("result")
    if not result:
        st.info("👆 Upload a resume and click **Analyze Resume** to get started.")
        return

    ai, rules = result["ai"], result["rules"]
    overall = final_score(ai["ats_score"], rules["score"])

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall ATS score", f"{overall}/100")
    m2.metric("AI evaluation", f"{ai['ats_score']}/100")
    m3.metric("Formatting checks", f"{rules['score']}/100")
    st.subheader(score_label(overall))
    st.progress(overall / 100)
    if ai["summary"]:
        st.write(ai["summary"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Breakdown", "🛠️ Improvements", "✍️ Rewrite examples", "🔑 Keywords", "✅ Checks"]
    )

    with tab1:
        for name, val in ai["score_breakdown"].items():
            st.write(f"**{name}** - {val}/100")
            st.progress(val / 100)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 💪 Strengths")
            for s in ai["strengths"] or ["-"]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("#### ⚠️ Weaknesses")
            for w in ai["weaknesses"] or ["-"]:
                st.markdown(f"- {w}")

    with tab2:
        order = {"high": 0, "medium": 1, "low": 2}
        items = sorted(ai["improvements"], key=lambda i: order.get(str(i.get("priority", "")).lower(), 3))
        icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        if not items:
            st.write("No improvements returned.")
        for i in items:
            pr = str(i.get("priority", "")).lower()
            with st.expander(f"{icons.get(pr, '⚪')} {i.get('priority', '')} - {i.get('section', 'General')}"):
                st.markdown(f"**Issue:** {i.get('issue', '')}")
                st.markdown(f"**Fix:** {i.get('suggestion', '')}")

    with tab3:
        if not ai["rewrite_examples"]:
            st.write("No rewrite examples returned.")
        for r in ai["rewrite_examples"]:
            st.markdown(f"❌ **Before:** {r.get('original', '')}")
            st.markdown(f"✅ **After:** {r.get('improved', '')}")
            st.markdown("---")
        st.caption("Check every rewrite for accuracy - only keep claims that are true for you.")

    with tab4:
        if ai["missing_keywords"]:
            st.write("Consider adding these (only if they honestly apply to you):")
            st.write(" ".join(f"`{k}`" for k in ai["missing_keywords"]))
        else:
            st.write("No missing keywords identified.")

    with tab5:
        for label, passed, detail, _, _ in rules["checks"]:
            st.write(f"{'✅' if passed else '❌'} **{label}**" + ("" if passed else f" - {detail}"))

    report = {"overall_score": overall, "ai_analysis": ai,
              "rule_checks": [{"check": c[0], "passed": c[1]} for c in rules["checks"]]}
    st.download_button("⬇️ Download report (JSON)", json.dumps(report, indent=2),
                       file_name="ats_report.json", mime="application/json")


if __name__ == "__main__":
    main()
