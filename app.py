import streamlit as st
from transformers import pipeline
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  /* ── global reset ── */
  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* ── hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── page background ── */
  .stApp {
      background: #0d0d0f;
      background-image:
          radial-gradient(ellipse 80% 60% at 50% -10%, rgba(100,60,255,.25) 0%, transparent 70%),
          radial-gradient(ellipse 50% 40% at 80% 90%, rgba(255,60,120,.12) 0%, transparent 60%);
  }

  /* ── hero title ── */
  .hero-title {
      font-family: 'Space Mono', monospace;
      font-size: clamp(2rem, 5vw, 3.2rem);
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -1px;
      line-height: 1.1;
      margin-bottom: 6px;
  }
  .hero-sub {
      color: #6b6b7a;
      font-size: .95rem;
      font-weight: 300;
      letter-spacing: .04em;
      margin-bottom: 40px;
  }
  .accent { color: #7c5cfc; }

  /* ── card wrapper ── */
  .card {
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 18px;
      padding: 32px 36px;
      backdrop-filter: blur(12px);
      margin-bottom: 24px;
  }

  /* ── textarea override ── */
  textarea {
      background: rgba(255,255,255,.06) !important;
      border: 1px solid rgba(255,255,255,.12) !important;
      border-radius: 12px !important;
      color: #e8e8ef !important;
      font-family: 'DM Sans', sans-serif !important;
      font-size: .97rem !important;
      resize: vertical !important;
  }
  textarea:focus {
      border-color: #7c5cfc !important;
      box-shadow: 0 0 0 3px rgba(124,92,252,.18) !important;
  }

  /* ── button ── */
  .stButton > button {
      width: 100%;
      background: linear-gradient(135deg, #7c5cfc 0%, #b44ff5 100%);
      color: #fff;
      font-family: 'Space Mono', monospace;
      font-size: .88rem;
      font-weight: 700;
      letter-spacing: .08em;
      border: none;
      border-radius: 12px;
      padding: 14px 0;
      cursor: pointer;
      transition: opacity .2s, transform .15s;
  }
  .stButton > button:hover {
      opacity: .88;
      transform: translateY(-2px);
  }
  .stButton > button:active { transform: translateY(0); }

  /* ── result cards ── */
  .result-positive {
      background: linear-gradient(135deg, rgba(34,197,94,.15), rgba(16,185,129,.08));
      border: 1px solid rgba(34,197,94,.35);
      border-radius: 16px;
      padding: 28px 32px;
      text-align: center;
      animation: popIn .4s cubic-bezier(.34,1.56,.64,1) both;
  }
  .result-negative {
      background: linear-gradient(135deg, rgba(239,68,68,.15), rgba(220,38,38,.08));
      border: 1px solid rgba(239,68,68,.35);
      border-radius: 16px;
      padding: 28px 32px;
      text-align: center;
      animation: popIn .4s cubic-bezier(.34,1.56,.64,1) both;
  }
  .result-neutral {
      background: linear-gradient(135deg, rgba(148,163,184,.15), rgba(100,116,139,.08));
      border: 1px solid rgba(148,163,184,.30);
      border-radius: 16px;
      padding: 28px 32px;
      text-align: center;
      animation: popIn .4s cubic-bezier(.34,1.56,.64,1) both;
  }
  .result-emoji { font-size: 3.5rem; margin-bottom: 8px; }
  .result-label {
      font-family: 'Space Mono', monospace;
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 4px;
  }
  .result-score { color: #a0a0b8; font-size: .9rem; }
  .result-score strong { color: #e8e8ef; }

  /* ── confidence bar ── */
  .conf-bar-bg {
      background: rgba(255,255,255,.1);
      border-radius: 99px;
      height: 8px;
      margin-top: 16px;
      overflow: hidden;
  }
  .conf-bar-fill-pos {
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg,#22c55e,#10b981);
      transition: width 1s ease;
  }
  .conf-bar-fill-neg {
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg,#ef4444,#dc2626);
      transition: width 1s ease;
  }
  .conf-bar-fill-neu {
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg,#94a3b8,#64748b);
      transition: width 1s ease;
  }

  /* ── history items ── */
  .history-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: rgba(255,255,255,.03);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 10px;
      margin-bottom: 8px;
      font-size: .87rem;
      color: #9090a8;
  }
  .history-badge-pos {
      background: rgba(34,197,94,.2);
      color: #4ade80;
      font-family: 'Space Mono', monospace;
      font-size: .7rem;
      padding: 2px 8px;
      border-radius: 99px;
      white-space: nowrap;
  }
  .history-badge-neg {
      background: rgba(239,68,68,.2);
      color: #f87171;
      font-family: 'Space Mono', monospace;
      font-size: .7rem;
      padding: 2px 8px;
      border-radius: 99px;
      white-space: nowrap;
  }
  .history-badge-neu {
      background: rgba(148,163,184,.2);
      color: #cbd5e1;
      font-family: 'Space Mono', monospace;
      font-size: .7rem;
      padding: 2px 8px;
      border-radius: 99px;
      white-space: nowrap;
  }
  .history-text {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
  }

  /* ── divider ── */
  .divider {
      border: none;
      border-top: 1px solid rgba(255,255,255,.07);
      margin: 28px 0;
  }

  /* ── section label ── */
  .section-label {
      font-family: 'Space Mono', monospace;
      font-size: .72rem;
      letter-spacing: .12em;
      color: #4a4a60;
      text-transform: uppercase;
      margin-bottom: 14px;
  }

  /* ── pop-in animation ── */
  @keyframes popIn {
      from { opacity:0; transform: scale(.9) translateY(10px); }
      to   { opacity:1; transform: scale(1) translateY(0); }
  }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


# ─── Session State ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Helper ─────────────────────────────────────────────────────────────────
def classify(label: str, score: float):
    if label == "POSITIVE":
        return "POSITIVE", "😊", "positive", score
    elif label == "NEGATIVE":
        return "NEGATIVE", "😞", "negative", score
    else:
        return "NEUTRAL", "😐", "neutral", score


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Sentiment<br><span class='accent'>Analyzer</span></div>
<div class='hero-sub'>Powered by DistilBERT · Instant emotion detection from text</div>
""", unsafe_allow_html=True)


# ─── Input Card ──────────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Your Text</div>", unsafe_allow_html=True)

text_input = st.text_area(
    label="",
    placeholder="Type or paste any text here — a tweet, review, sentence …",
    height=140,
    label_visibility="collapsed",
)

analyze_btn = st.button("⚡  ANALYZE SENTIMENT", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ─── Analysis ────────────────────────────────────────────────────────────────
if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Loading model & analyzing …"):
            classifier = load_model()
            result = classifier(text_input[:512])[0]

        label, emoji, css_class, score = classify(result["label"], result["score"])
        pct = round(score * 100, 1)

        # ── result card ──
        fill_class = f"conf-bar-fill-{css_class}"
        st.markdown(f"""
        <div class='result-{css_class}'>
            <div class='result-emoji'>{emoji}</div>
            <div class='result-label'>{label}</div>
            <div class='result-score'>Confidence: <strong>{pct}%</strong></div>
            <div class='conf-bar-bg'>
                <div class='{fill_class}' style='width:{pct}%'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── save to history ──
        st.session_state.history.insert(0, {
            "text": text_input.strip(),
            "label": label,
            "score": pct,
            "css": css_class,
        })
        if len(st.session_state.history) > 8:
            st.session_state.history = st.session_state.history[:8]


# ─── History ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 0.18])
    with col1:
        st.markdown("<div class='section-label'>Recent Analyses</div>", unsafe_allow_html=True)
    with col2:
        if st.button("Clear", key="clear_history"):
            st.session_state.history = []
            st.rerun()

    for item in st.session_state.history:
        badge_class = f"history-badge-{item['css']}"
        preview = item["text"][:70] + ("…" if len(item["text"]) > 70 else "")
        st.markdown(f"""
        <div class='history-item'>
            <span class='{badge_class}'>{item['label']} {item['score']}%</span>
            <span class='history-text'>{preview}</span>
        </div>
        """, unsafe_allow_html=True)