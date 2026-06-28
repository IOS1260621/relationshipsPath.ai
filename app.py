import streamlit as st
from datetime import datetime
import json
import os
import re
import html

from openai import OpenAI

try:
    from relationship_config import (
        APP_NAME,
        MODALITY_NAME,
        MAX_CHAT_MESSAGES,
        MAX_JOURNAL_ENTRIES,
        SAFETY_KEYWORDS,
        MANIPULATION_KEYWORDS,
        DIAGNOSIS_KEYWORDS,
        SITUATION_OPTIONS,
        SITUATION_FOCUS,
        REWRITE_TEMPLATES,
    )
except (ModuleNotFoundError, ImportError):
    # Fallback config so this file can run by itself if relationship_config.py is missing or incomplete.
    APP_NAME = "RelationshipPath AI"
    MODALITY_NAME = "Relationships"
    MAX_CHAT_MESSAGES = 50
    MAX_JOURNAL_ENTRIES = 100
    SAFETY_KEYWORDS = [
        "suicide", "kill myself", "end my life", "self harm", "self-harm",
        "hurt myself", "hurt her", "hurt him", "hurt them", "kill her", "kill him",
        "violence", "hit me", "hit her", "hit him", "choked", "strangled",
        "threatened", "threat", "weapon", "gun", "knife",
        "abuse", "abusive", "domestic violence", "scared of him", "scared of her",
        "stalking", "stalk", "unsafe", "danger", "emergency"
    ]
    MANIPULATION_KEYWORDS = [
        "make them jealous", "force them", "control them", "manipulate", "revenge",
        "spy", "stalk", "track them", "make them stay", "gaslight", "punish them"
    ]
    DIAGNOSIS_KEYWORDS = [
        "narcissist", "bipolar", "borderline", "psychopath", "sociopath",
        "diagnose", "mental disorder", "personality disorder"
    ]
    SITUATION_OPTIONS = [
        "Argument or conflict",
        "Rewrite a difficult text",
        "Apology",
        "Boundary",
        "Trust repair",
        "Prepare for a hard conversation",
        "Understand a pattern",
    ]
    SITUATION_FOCUS = {
        "Argument or conflict": "Lower emotional intensity and turn the argument into one calm conversation.",
        "Rewrite a difficult text": "Make the message clearer, calmer, and harder to misread.",
        "Apology": "Take responsibility without over-explaining or demanding forgiveness.",
        "Boundary": "State one clear boundary without threats, punishment, or control.",
        "Trust repair": "Name the hurt, clarify the repair action, and avoid rushing trust.",
        "Prepare for a hard conversation": "Prepare one point, one feeling, and one request before the conversation.",
        "Understand a pattern": "Focus on repeated behaviors, triggers, and healthier choices you can control.",
    }
    REWRITE_TEMPLATES = {
        "Calm": "I want to say this more calmly: {base}",
        "Loving": "I care about us, and I want to say this with love: {base}",
        "Boundary": "I want to be clear about my boundary: {base}",
        "Apology": "I want to take responsibility and apologize: {base}",
        "Short Text": "Short version: {base}",
    }

# ============================================================
# RelationshipPath AI - MVP
# First modality: Relationships
# Positioning: AI relationship coach, not licensed therapy.
# ============================================================

st.set_page_config(
    page_title="RelationshipPath AI",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Helper Functions
# -----------------------------

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    if "safety_acknowledged" not in st.session_state:
        st.session_state.safety_acknowledged = False

    if "current_modality" not in st.session_state:
        st.session_state.current_modality = MODALITY_NAME


def get_query_param_value(name, default=""):
    """Safely read one query parameter value across Streamlit versions."""
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default

    if isinstance(value, list):
        return value[0] if value else default

    return value if value is not None else default


def remember_safety_acknowledgment_from_url():
    """Keep the safety gate passed when top-nav links reload the page on mobile."""
    if get_query_param_value("ack", "") == "1":
        st.session_state.safety_acknowledged = True


def mark_safety_acknowledged():
    st.session_state.safety_acknowledged = True
    try:
        current_page = get_query_param_value("page", "coach") or "coach"
        st.query_params["ack"] = "1"
        st.query_params["page"] = current_page
    except Exception:
        pass


def append_capped_state_list(state_key, item, max_items):
    st.session_state[state_key].append(item)
    if len(st.session_state[state_key]) > max_items:
        st.session_state[state_key] = st.session_state[state_key][-max_items:]


# -----------------------------
# OpenAI / LLM Helpers
# -----------------------------

def get_secret_or_env(name, default=""):
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def get_openai_model():
    return get_secret_or_env("OPENAI_MODEL", "gpt-5.4-mini")


def get_openai_client():
    api_key = get_secret_or_env("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def has_openai_key():
    return bool(get_secret_or_env("OPENAI_API_KEY", ""))


@st.cache_data(ttl=300, show_spinner=False)
def check_openai_availability_cached(model_name, key_present):
    """Check OpenAI availability at most once every 5 minutes."""
    checked_at = datetime.now().strftime("%I:%M:%S %p")

    if not key_present:
        return {
            "available": False,
            "label": "AI Offline",
            "detail": "OPENAI_API_KEY not found.",
            "checked_at": checked_at,
        }

    try:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client could not be created.")

        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "low"},
            instructions="You are a connection check. Reply with only: OK",
            input="Reply with OK.",
            max_output_tokens=20,
        )
        output_text = (response.output_text or "").strip()

        return {
            "available": True,
            "label": "AI Available",
            "detail": f"Model: {model_name}. Last check: {checked_at}.",
            "checked_at": checked_at,
            "test_response": output_text,
        }
    except Exception as error:
        return {
            "available": False,
            "label": "AI Offline",
            "detail": f"{type(error).__name__}: {error}",
            "checked_at": checked_at,
        }


def render_ai_status_light():
    """Render a one-line AI status indicator: AI + green/red dot."""
    status = check_openai_availability_cached(get_openai_model(), has_openai_key())
    is_available = bool(status.get("available"))

    dot = "#22c55e" if is_available else "#ef4444"
    ring = "rgba(34, 197, 94, 0.22)" if is_available else "rgba(239, 68, 68, 0.22)"
    label = "Connected" if is_available else "Offline"

    status_html = f"""
        <div class="ai-status-one-line" title="AI {label}">
            <span class="ai-status-text">AI</span>
            <span class="ai-status-dot" style="background:{dot}; box-shadow:0 0 0 5px {ring};"></span>
        </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


def call_openai_response(system_prompt, user_prompt, max_output_tokens=900):
    """Call OpenAI Responses API and return text. Raises errors to the caller."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it as a Codespaces secret, environment variable, "
            "or Streamlit secret before using AI mode."
        )

    response = client.responses.create(
        model=get_openai_model(),
        reasoning={"effort": "low"},
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )

    return (response.output_text or "").strip()


def relationship_system_prompt():
    return f"""
You are {APP_NAME}, an AI relationship coach for the {MODALITY_NAME} modality.

Positioning:
- You are AI relationship coaching, not licensed therapy.
- Do not present yourself as a therapist, clinician, doctor, lawyer, crisis line, or emergency service.
- Do not diagnose the user, their partner, or anyone else.

Safety rules:
- If the user describes immediate danger, abuse, violence, threats, self-harm, harm to others, stalking, coercion, or weapon involvement, pause coaching and recommend immediate human support, emergency services, or a trusted person nearby.
- Do not help with manipulation, revenge, spying, coercion, emotional control, or forcing someone to stay. Redirect to honesty, boundaries, and respectful communication.
- Avoid legal or medical advice.

Coaching style:
- Warm, clear, practical, and emotionally intelligent.
- Help users think before they text.
- Turn arguments into conversations.
- Use short sections and give the user something immediately usable.
- When helpful, include a message draft the user can copy.
- End with one realistic next step.
"""


def build_recent_chat_context(limit=6):
    messages = st.session_state.get("chat_history", [])[-limit:]
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content[:1200]}")
    return "\n\n".join(lines)


def normalize_text(text):
    normalized = text.lower().replace("’", "'").replace("`", "'")
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_keywords(keyword_list):
    return [normalize_text(keyword) for keyword in keyword_list if normalize_text(keyword)]


NORMALIZED_SAFETY_KEYWORDS = normalize_keywords(SAFETY_KEYWORDS)
NORMALIZED_MANIPULATION_KEYWORDS = normalize_keywords(MANIPULATION_KEYWORDS)
NORMALIZED_DIAGNOSIS_KEYWORDS = normalize_keywords(DIAGNOSIS_KEYWORDS)


def contains_any_keyword(text, normalized_keywords):
    text_normalized = normalize_text(text)
    if not text_normalized:
        return False

    bounded_text = f" {text_normalized} "
    return any(f" {keyword} " in bounded_text for keyword in normalized_keywords)


def detect_safety_issue(text):
    return contains_any_keyword(text, NORMALIZED_SAFETY_KEYWORDS)


def detect_manipulation_request(text):
    return contains_any_keyword(text, NORMALIZED_MANIPULATION_KEYWORDS)


def detect_diagnosis_request(text):
    return contains_any_keyword(text, NORMALIZED_DIAGNOSIS_KEYWORDS)


def simple_emotion_guess(text):
    text_lower = text.lower()

    emotion_map = {
        "hurt": ["hurt", "ignored", "dismissed", "unimportant", "betrayed"],
        "angry": ["angry", "mad", "furious", "pissed", "annoyed"],
        "sad": ["sad", "lonely", "alone", "unloved", "cry"],
        "anxious": ["anxious", "worried", "afraid", "nervous", "insecure"],
        "confused": ["confused", "lost", "mixed signals", "don’t know", "don't know"],
        "guilty": ["guilty", "my fault", "apologize", "sorry"],
    }

    detected = []

    for emotion, words in emotion_map.items():
        if any(word in text_lower for word in words):
            detected.append(emotion)

    if not detected:
        return "emotionally activated or uncertain"

    return ", ".join(detected)


def compute_formula_scores(user_input, happened_text, felt_text, need_text):
    combined_text = " ".join([
        user_input.strip(),
        happened_text.strip(),
        felt_text.strip(),
        need_text.strip(),
    ]).strip()

    words = [w for w in normalize_text(combined_text).split(" ") if w]
    word_count = len(words)

    structure_points = 0
    structure_points += 1 if happened_text.strip() else 0
    structure_points += 1 if felt_text.strip() else 0
    structure_points += 1 if need_text.strip() else 0

    # Formula 1: Clarity score (0-10) rewards structure + enough context.
    clarity_score = min(
        10.0,
        round((structure_points * 2.2) + min(word_count / 15.0, 3.4), 1)
    )

    intense_words = [
        "always", "never", "furious", "betrayed", "hate", "done", "can't", "cant"
    ]
    intensity_hits = sum(1 for word in intense_words if f" {word} " in f" {normalize_text(combined_text)} ")

    # Formula 2: Emotional intensity score (0-10).
    intensity_score = min(10.0, round((intensity_hits * 1.8) + (3.0 if detect_safety_issue(combined_text) else 0.0), 1))

    return clarity_score, intensity_score


def build_ai_step_outputs(user_input, happened_text, felt_text, need_text):
    clarity_score, intensity_score = compute_formula_scores(
        user_input,
        happened_text,
        felt_text,
        need_text,
    )

    safety_scope_text = " ".join([
        user_input.strip(),
        happened_text.strip(),
        felt_text.strip(),
        need_text.strip(),
    ])

    if detect_safety_issue(safety_scope_text):
        step_4 = (
            "Safety override triggered. The formula detected potential risk language, so coaching is paused and immediate human support is recommended."
        )
        step_5 = "Next step: pause messaging and contact trusted or professional support now."
        return step_4, step_5

    if detect_manipulation_request(safety_scope_text):
        step_4 = (
            "Manipulation override triggered. The formula blocks control/revenge framing and switches to healthy communication only."
        )
        step_5 = "Next step: restate your goal as honesty + respect + one clear boundary."
        return step_4, step_5

    if intensity_score >= 7.0:
        next_step = "Take a 20-minute pause first, then send only one calm sentence and one request."
    elif clarity_score < 5.0:
        next_step = "Add one factual event, one feeling, and one specific need before sending any message."
    else:
        next_step = "Send the calm draft and ask for a short 10-minute conversation window."

    step_4 = (
        f"Clarity score = min(10, 2.2 x structure_points + min(word_count/15, 3.4)) = {clarity_score}/10; "
        f"Intensity score = min(10, 1.8 x trigger_hits + safety_bonus) = {intensity_score}/10."
    )
    step_5 = f"Formula-based next step: {next_step}"
    return step_4, step_5


def generate_relationship_coaching_response(
    user_input,
    situation_type=None,
    happened_text="",
    felt_text="",
    need_text="",
):
    """
    OpenAI-powered relationship coaching response.
    Safety / manipulation / diagnosis checks run before the LLM call.
    If OpenAI is unavailable, the app falls back to the local rule-based coach.
    """

    safety_scope_text = " ".join([
        user_input.strip(),
        happened_text.strip(),
        felt_text.strip(),
        need_text.strip(),
    ])

    if detect_safety_issue(safety_scope_text):
        return """
I’m concerned there may be a safety issue here.

This app is not designed for emergencies, abuse, violence, threats, self-harm, or situations where someone may be in danger. Please pause relationship coaching for now and contact emergency services, a trusted person nearby, or a qualified professional support resource.

Your immediate safety matters more than resolving the relationship conflict in this chat.
"""

    if detect_manipulation_request(safety_scope_text):
        return """
I can’t help with manipulation, revenge, spying, emotional control, or trying to force someone to stay.

What I can help with is a healthier version of this: clearly expressing your feelings, setting a boundary, asking for honesty, or deciding what you need to do next.

A better starting point could be:

“I don’t want to control you or pressure you. I want to understand where we stand and what we both honestly want.”
"""

    if detect_diagnosis_request(safety_scope_text):
        return """
I can’t diagnose your partner or label them with a mental-health condition.

What we can do instead is focus on observable behavior:

- What happened?
- How did it affect you?
- What boundary or request do you want to communicate?
- What pattern are you noticing?

A healthy script could be:

“When this happens, I feel hurt and disconnected. I’m not trying to label you. I want to talk about the behavior and how we can handle it differently.”
"""

    _hidden_formula_check, ai_next_step = build_ai_step_outputs(user_input, happened_text, felt_text, need_text)
    ai_next_step = ai_next_step.replace("Formula-based next step: ", "").strip()

    llm_prompt = f"""
Task: Relationship coach chat response.

Situation type:
{situation_type}

Recent chat context:
{build_recent_chat_context()}

Current user message:
{user_input}

Respond like a calm, supportive relationship coach in a natural conversation.
Do NOT use a fixed template or headings like "What I hear", "Slow it down", "Say it for me", or "Next step".

Write 2 to 4 short phone-friendly paragraphs. Sound human, warm, and direct.
Acknowledge the user's feelings without overclaiming. Help them slow down, see the situation more clearly, and choose a healthier response.

If useful, include one copy-ready message using natural wording like: "You could say:"
End with one simple next move. Internal next-step suggestion you may use if helpful: {ai_next_step}

Important:
- Do not show formulas.
- Do not show scores.
- Do not mention clarity score, intensity score, structure points, trigger hits, or AI formula check.
- Do not sound clinical or robotic.
- Do not force a section-by-section format.
"""

    try:
        return call_openai_response(relationship_system_prompt(), llm_prompt)
    except Exception as error:
        # Safe fallback: if OpenAI is unavailable, use the original local MVP logic.
        emotion = simple_emotion_guess(user_input)
        situation_focus = SITUATION_FOCUS.get(
            situation_type,
            "Focus on one specific recent moment and what you want to communicate calmly."
        )

        happened_value = happened_text.strip()
        felt_value = felt_text.strip()
        need_value = need_text.strip()

        happened_line = happened_value if happened_value else "Describe only the facts, without blame."
        felt_line = felt_value if felt_value else "Use I feel language instead of you always language."
        need_line = need_value if need_value else "Be specific about the relationship need."

        if happened_value or felt_value or need_value:
            scripted_message = (
                f"I want to share this calmly. {happened_line} I felt {felt_line if felt_value else 'emotionally affected by this'}. "
                f"What I need is {need_line if need_value else 'a calmer conversation where both sides feel heard'}."
            )
        else:
            scripted_message = (
                "Hey, I do not want us to keep arguing. I care about us, and I want to understand each other better. "
                "When we talked earlier, I felt hurt and unheard. Can we take a few minutes to talk calmly and each explain what we were feeling?"
            )

        _hidden_formula_check, ai_next_step = build_ai_step_outputs(user_input, happened_text, felt_text, need_text)
        ai_next_step = ai_next_step.replace("Formula-based next step: ", "").strip()

        return f"""
⚠️ **AI fallback mode:** OpenAI did not respond, so the app used its built-in coach.

It sounds like you may be feeling **{emotion}**. {situation_focus}

Try to keep this about one specific moment, one feeling, and one clear request. Avoid stacking every past argument into the same message.

You could say: "{scripted_message}"

For now, the next move is simple: {ai_next_step}

Technical note: `{error}`
"""


def rewrite_message(original_message, style):
    if not original_message.strip():
        return "Type a message first."

    if detect_safety_issue(original_message):
        return """
This message appears to involve possible safety concerns. This app should not be used to manage emergencies, abuse, threats, or self-harm situations. Please seek immediate human support.
"""

    if detect_manipulation_request(original_message):
        return """
I can’t rewrite this into a manipulative, threatening, controlling, or revenge-based message.

A healthier version would be:

“I want to be honest about how I feel without trying to control you. Can we talk openly about where we stand?”
"""

    base = original_message.strip()

    llm_prompt = f"""
Task: Rewrite a relationship message.

Rewrite style: {style}

Original message:
{base}

Create a copy-ready rewritten version. Requirements:
- Keep the user's core meaning.
- Make it calmer, clearer, and relationship-focused.
- Do not add threats, diagnosis, manipulation, or guilt pressure.
- For Boundary style, state one clear boundary and one respectful next step.
- For Apology style, include responsibility without begging or demanding forgiveness.
- For Short Text style, make it text-message length.
Return only the rewritten message, with no long explanation.
"""

    try:
        return call_openai_response(relationship_system_prompt(), llm_prompt, max_output_tokens=350)
    except Exception as error:
        template = REWRITE_TEMPLATES.get(style)
        fallback = template.format(base=base) if template else base
        return f"⚠️ AI fallback mode: `{error}`\n\n{fallback}"


def create_weekly_summary(connection_score, communication_score, trust_score, conflict_score, notes):
    avg_score = round((connection_score + communication_score + trust_score + conflict_score) / 4, 1)

    if detect_safety_issue(notes):
        return """
I’m concerned your weekly notes may include safety-related concerns.

This app is not designed for emergencies, abuse, violence, threats, self-harm, or situations where someone may be in danger. Please pause relationship coaching for now and contact emergency services, a trusted person nearby, or a qualified professional support resource.
"""

    if detect_manipulation_request(notes):
        return """
I can’t help with manipulation, revenge, spying, emotional control, or trying to force someone to stay.

A healthier weekly goal is to focus on honesty, respect, boundaries, and what you can control.
"""

    if detect_diagnosis_request(notes):
        return """
I can’t diagnose your partner or label them with a mental-health condition.

For this check-in, focus on observable behavior, how it affected you, and what boundary or request you need.
"""

    llm_prompt = f"""
Task: Generate an AI weekly relationship check-in summary.

Scores, 1 to 10:
- Connection: {connection_score}
- Communication: {communication_score}
- Trust: {trust_score}
- Conflict handling: {conflict_score}
- Average score: {avg_score}

User notes:
{notes if notes.strip() else "No additional notes provided."}

Create a helpful weekly check-in summary with:
1. Overall read of the week
2. Strengths
3. Main risk area
4. One calm conversation to have
5. A copy-ready starter message

Do not diagnose. Do not call this therapy. Keep it practical and relationship-coaching focused.
"""

    try:
        return call_openai_response(relationship_system_prompt(), llm_prompt, max_output_tokens=700)
    except Exception as error:
        if avg_score >= 8:
            tone = "Your relationship check-in looks strong this week."
        elif avg_score >= 5:
            tone = "Your relationship check-in looks mixed. There may be some strengths and some areas needing attention."
        else:
            tone = "Your relationship check-in suggests this may be a difficult week emotionally."

        return f"""
⚠️ **AI fallback mode:** OpenAI did not respond, so the app used its built-in weekly summary.

Technical note: `{error}`

## Weekly Relationship Check-In Summary

**Average score:** {avg_score}/10

{tone}

### Scores
- Connection: {connection_score}/10
- Communication: {communication_score}/10
- Trust: {trust_score}/10
- Conflict level: {conflict_score}/10

### Reflection
{notes if notes.strip() else "No additional notes added."}

### Suggested next step
Choose one calm conversation to have this week. Keep it short and focus on one topic only.

A useful starter:

“Can we check in for 10 minutes this week? I don’t want to argue. I just want us to understand each other better.”
"""


def export_journal_as_json():
    return json.dumps(st.session_state.journal_entries, indent=2)


# -----------------------------
# UI
# -----------------------------

initialize_session_state()
remember_safety_acknowledgment_from_url()

# -----------------------------
# Phone-first visual polish
# -----------------------------

st.markdown(
    """
    <style>
    /* Hide Streamlit's top toolbar/header/menu so it does not cover the app title on phones. */
    header,
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .stAppHeader,
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Clean phone-first layout with no sidebar. */
    section[data-testid="stSidebar"],
    button[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }

    .stApp {
        margin-left: 0 !important;
    }

    .block-container {
        max-width: 760px;
        padding-top: 0.75rem;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
        padding-bottom: 6rem;
    }

    h1 {
        font-size: clamp(1.75rem, 7vw, 2.45rem) !important;
        line-height: 1.05 !important;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem !important;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    div[data-testid="stCaptionContainer"] {
        font-size: 1rem;
    }

    /* Four-button top nav. */
    .top-nav-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.42rem;
        margin: 0.75rem 0 0.55rem 0;
        width: 100%;
    }

    .top-nav-button {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 54px;
        padding: 0.55rem 0.35rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.38);
        background: rgba(255, 255, 255, 0.08);
        color: inherit !important;
        text-decoration: none !important;
        font-size: clamp(0.72rem, 2.85vw, 1rem);
        font-weight: 950;
        line-height: 1.05;
        text-align: center;
        letter-spacing: -0.02em;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        user-select: none;
        -webkit-tap-highlight-color: transparent;
        overflow-wrap: anywhere;
    }

    .top-nav-button.active {
        background: #ef4444;
        border-color: #ef4444;
        color: #ffffff !important;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.28);
    }

    .top-nav-button:active {
        transform: translateY(1px);
    }

    /* Streamlit widget navigation: smoother than link navigation because it avoids full browser reloads.
       Force the four choices into ONE horizontal row on phones instead of Streamlit stacking them. */
    div[data-testid="stRadio"] {
        width: 100% !important;
    }

    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        justify-content: space-between !important;

        gap: 0.25rem !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0.75rem 0 0.55rem 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: 25% !important;
        width: auto !important;

        min-height: 68px !important;
        padding: 0.7rem 0.18rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.38) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        color: inherit !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        user-select: none;
        -webkit-tap-highlight-color: transparent;
        text-align: center !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #ef4444 !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.28);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
        color: #ffffff !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child,
    div[data-testid="stRadio"] div[role="radiogroup"] label input {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label span {

        font-size: clamp(0.76rem, 3.2vw, 1.05rem) !important;
        font-weight: 1000 !important;
        line-height: 1.08 !important;
        letter-spacing: -0.035em !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .ai-status-one-line {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.15rem 0 1rem 0;
        padding: 0.35rem 0.58rem;
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        line-height: 1;
    }

    .ai-status-text {
        font-size: 0.9rem;
        font-weight: 950;
        letter-spacing: 0.02em;
    }

    .ai-status-dot {
        width: 0.75rem;
        height: 0.75rem;
        display: inline-block;
        border-radius: 999px;
        flex: 0 0 auto;
    }

    /* Bigger tap targets for iPhone users. */
    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 48px;
        border-radius: 14px;
        font-weight: 800;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input {
        border-radius: 14px;
    }

    div[data-baseweb="textarea"] textarea,
    div[data-testid="stChatInput"] textarea {
        font-size: 16px !important; /* Prevents iPhone Safari zoom on focus. */
        line-height: 1.4 !important;
    }

    .stAlert {
        border-radius: 16px;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
            padding-top: 0.65rem;
            max-width: 100vw !important;
        }

        .top-nav-row {
            gap: 0.28rem;
            margin-top: 0.55rem;
        }

        .top-nav-button {
            min-height: 48px;
            border-radius: 13px;
            padding: 0.42rem 0.18rem;
            font-size: clamp(0.62rem, 2.8vw, 0.82rem);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;

            gap: 0.18rem !important;
            margin-top: 0.55rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            max-width: 25% !important;

            min-height: 62px !important;
            border-radius: 13px !important;
            padding: 0.52rem 0.08rem !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p,
        div[data-testid="stRadio"] div[role="radiogroup"] label span {

            font-size: clamp(0.68rem, 3.05vw, 0.9rem) !important;
            line-height: 1.05 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"💬 {APP_NAME}")
st.caption("Think before you text.")

# -----------------------------
# Safety Acknowledgment
# -----------------------------

if not st.session_state.safety_acknowledged:
    st.subheader("Before we begin")

    st.info(
        """
        RelationshipPath AI is an AI relationship coach. It is not a licensed therapist,
        not couples therapy, and not a crisis service.

        If you are in immediate danger or there is abuse, violence, threats, or self-harm,
        please seek immediate human support instead of using this app.
        """
    )

    acknowledge = st.checkbox("I understand this is AI relationship coaching, not therapy or emergency support.")

    if acknowledge:
        mark_safety_acknowledged()
        st.rerun()

    st.stop()

# AI status is cached for 5 minutes, but the page no longer auto-refreshes.
# It rechecks on normal app reruns after the cache expires.

# -----------------------------
# Main Navigation — no sidebar on phone
# -----------------------------

st.session_state.current_modality = MODALITY_NAME

NAV_OPTIONS = {
    "coach": "Coach Chat",
    "say": "Say It For Me",
    "check": "Check-In",
    "journal": "Journal",
}

NAV_LABELS = list(NAV_OPTIONS.values())

# Use a real Streamlit widget for navigation instead of HTML links.
# This avoids a full browser page reload on phone and makes page changes feel smoother.
if "active_page" not in st.session_state:
    requested_page = get_query_param_value("page", "coach")
    st.session_state.active_page = NAV_OPTIONS.get(requested_page, "Coach Chat")

if st.session_state.active_page not in NAV_LABELS:
    st.session_state.active_page = "Coach Chat"

page = st.radio(
    "Main navigation",
    NAV_LABELS,
    index=NAV_LABELS.index(st.session_state.active_page),
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)

# Minimal one-line AI status.
render_ai_status_light()


# -----------------------------
# Page 1: Coach Chat
# -----------------------------

if page == "Coach Chat":
    st.header("Coach Chat")

    st.markdown(
        """
        Type what happened like you are texting a coach. The AI will answer in a more natural conversation style and help you decide what to say next.
        """
    )

    situation_type = st.selectbox(
        "What do you need help with?",
        SITUATION_OPTIONS
    )

    # Removed the optional 3-step input fields for a cleaner phone-first flow.
    happened_text = ""
    felt_text = ""
    need_text = ""

    user_message = st.chat_input("Type what happened or paste the text...")

    if user_message:
        append_capped_state_list("chat_history", {
            "role": "user",
            "content": user_message,
            "time": datetime.now().isoformat(),
            "situation_type": situation_type
        }, MAX_CHAT_MESSAGES)

        assistant_response = generate_relationship_coaching_response(
            user_message,
            situation_type,
            happened_text,
            felt_text,
            need_text,
        )

        append_capped_state_list("chat_history", {
            "role": "assistant",
            "content": assistant_response,
            "time": datetime.now().isoformat()
        }, MAX_CHAT_MESSAGES)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# -----------------------------
# Page 2: Say It For Me
# -----------------------------

elif page == "Say It For Me":
    st.header("Say It For Me")

    st.markdown(
        """
        Paste the text you are about to send.
        The app rewrites it into a calmer, clearer version made for real phone conversations.
        """
    )

    if "last_rewrite_output" not in st.session_state:
        st.session_state.last_rewrite_output = ""

    with st.form("say_it_for_me_form", clear_on_submit=False):
        original_message = st.text_area(
            "Original message",
            height=180,
            placeholder="Paste your message here..."
        )

        rewrite_style = st.radio(
            "Rewrite style",
            ["Calm", "Loving", "Boundary", "Apology", "Short Text"],
            horizontal=True,
            key="rewrite_style_choice",
        )

        submitted_rewrite = st.form_submit_button("Say It For Me")

    if submitted_rewrite:
        st.session_state.last_rewrite_output = rewrite_message(original_message, rewrite_style)

    if st.session_state.last_rewrite_output:
        st.subheader("Suggested Message")
        st.markdown(st.session_state.last_rewrite_output)

        st.download_button(
            label="Download suggested message",
            data=st.session_state.last_rewrite_output,
            file_name="relationship_message_rewrite.txt",
            mime="text/plain"
        )


# -----------------------------
# Page 3: Check-In
# -----------------------------

elif page == "Check-In":
    st.header("Check-In")

    st.markdown(
        """
        A quick relationship check-in. It does not diagnose your relationship.
        It helps you notice patterns and prepare one healthy next step.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        connection_score = st.slider("Connection", 1, 10, 5)
        communication_score = st.slider("Communication", 1, 10, 5)

    with col2:
        trust_score = st.slider("Trust", 1, 10, 5)
        conflict_score = st.slider("Conflict handling", 1, 10, 5)

    weekly_notes = st.text_area(
        "What stood out this week?",
        height=140,
        placeholder="Example: We argued twice, but we also had one good conversation..."
    )

    if st.button("Generate Check-In"):
        summary = create_weekly_summary(
            connection_score,
            communication_score,
            trust_score,
            conflict_score,
            weekly_notes
        )

        st.markdown(summary)

        st.download_button(
            label="Download check-in",
            data=summary,
            file_name="relationship_checkin.txt",
            mime="text/plain"
        )


# -----------------------------
# Page 4: Journal
# -----------------------------

elif page == "Journal":
    st.header("Journal")

    st.markdown(
        """
        Save private reflections during this session.
        This MVP stores entries only in the current Streamlit session unless you later add a database.
        """
    )

    journal_prompt = st.selectbox(
        "Journal prompt",
        [
            "What happened today?",
            "What did I feel?",
            "What did I need but not say?",
            "What boundary do I need?",
            "What do I appreciate about my partner?",
            "What pattern am I noticing?",
            "What do I want to say calmly?"
        ]
    )

    journal_text = st.text_area(
        "Your journal entry",
        height=180,
        placeholder="Write your reflection here..."
    )

    if st.button("Save Journal Entry"):
        if journal_text.strip():
            entry = {
                "time": datetime.now().isoformat(),
                "prompt": journal_prompt,
                "entry": journal_text.strip()
            }
            append_capped_state_list("journal_entries", entry, MAX_JOURNAL_ENTRIES)
            st.success("Journal entry saved for this session.")
        else:
            st.error("Write something before saving.")

    st.subheader("Saved Entries")

    if not st.session_state.journal_entries:
        st.info("No journal entries yet.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.journal_entries), start=1):
            with st.expander(f"Entry {idx}: {entry['prompt']}"):
                st.caption(entry["time"])
                st.write(entry["entry"])

        st.download_button(
            label="Download journal as JSON",
            data=export_journal_as_json(),
            file_name="relationship_journal.json",
            mime="application/json"
        )


# About/Safety page removed from the public navigation.
