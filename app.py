import streamlit as st
from datetime import datetime
import json
import re

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

# ============================================================
# RelationshipPath AI - MVP
# First modality: Relationships
# Positioning: AI relationship coach, not licensed therapy.
# ============================================================

st.set_page_config(
    page_title="RelationshipPath AI",
    page_icon="💬",
    layout="wide"
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


def append_capped_state_list(state_key, item, max_items):
    st.session_state[state_key].append(item)
    if len(st.session_state[state_key]) > max_items:
        st.session_state[state_key] = st.session_state[state_key][-max_items:]


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

    if detect_safety_issue(user_input):
        step_4 = (
            "Safety override triggered. The formula detected potential risk language, so coaching is paused and immediate human support is recommended."
        )
        step_5 = "Next step: pause messaging and contact trusted or professional support now."
        return step_4, step_5

    if detect_manipulation_request(user_input):
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
    Rule-based relationship coaching response.
    This MVP does not use an external LLM yet.
    Later you can replace this function with an OpenAI API call.
    """

    if detect_safety_issue(user_input):
        return """
I’m concerned there may be a safety issue here.

This app is not designed for emergencies, abuse, violence, threats, self-harm, or situations where someone may be in danger. Please pause relationship coaching for now and contact emergency services, a trusted person nearby, or a qualified professional support resource.

Your immediate safety matters more than resolving the relationship conflict in this chat.
"""

    if detect_manipulation_request(user_input):
        return """
I can’t help with manipulation, revenge, spying, emotional control, or trying to force someone to stay.

What I can help with is a healthier version of this: clearly expressing your feelings, setting a boundary, asking for honesty, or deciding what you need to do next.

A better starting point could be:

“I don’t want to control you or pressure you. I want to understand where we stand and what we both honestly want.”
"""

    if detect_diagnosis_request(user_input):
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

    ai_step_4, ai_step_5 = build_ai_step_outputs(user_input, happened_text, felt_text, need_text)

    response = f"""
It sounds like you may be feeling **{emotion}**.

### Focus for this session
{situation_focus}

Let’s slow it down and separate the situation into four parts:

### 1. What happened
{happened_line}

### 2. What you felt
{felt_line}

### 3. What you need
{need_line}

### 4. AI-calculated message formula
Based on your inputs, here is a calmer draft:

"{scripted_message}"

{ai_step_4}

### 5. AI-calculated next step
{ai_step_5}
"""

    return response


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

    template = REWRITE_TEMPLATES.get(style)
    if template:
        return template.format(base=base)

    return base


def create_weekly_summary(connection_score, communication_score, trust_score, conflict_score, notes):
    avg_score = round((connection_score + communication_score + trust_score + conflict_score) / 4, 1)

    if avg_score >= 8:
        tone = "Your relationship check-in looks strong this week."
    elif avg_score >= 5:
        tone = "Your relationship check-in looks mixed. There may be some strengths and some areas needing attention."
    else:
        tone = "Your relationship check-in suggests this may be a difficult week emotionally."

    summary = f"""
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
    return summary


def export_journal_as_json():
    return json.dumps(st.session_state.journal_entries, indent=2)


# -----------------------------
# UI
# -----------------------------

initialize_session_state()

st.title(f"💬 {APP_NAME}")
st.caption("First modality: Relationships — AI relationship coaching for communication, reflection, and conflict de-escalation.")

with st.sidebar:
    st.header("Modality")
    st.session_state.current_modality = st.selectbox(
        "Choose modality",
        [MODALITY_NAME],
        index=0
    )

    st.divider()

    st.header("Important")
    st.warning(
        "This app is not therapy, couples therapy, medical care, legal advice, or emergency support."
    )

    st.markdown(
        """
        Use this app for:
        - communication help
        - relationship reflection
        - conflict de-escalation
        - message rewriting
        - weekly check-ins

        Do not use this app for:
        - emergencies
        - abuse or violence
        - self-harm
        - threats
        - stalking
        - coercion
        - diagnosis
        """
    )

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

    if st.button("Clear Journal"):
        st.session_state.journal_entries = []
        st.success("Journal cleared.")


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
        st.session_state.safety_acknowledged = True
        st.rerun()

    st.stop()


# -----------------------------
# Main Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Coach Chat",
    "Rewrite Message",
    "Weekly Check-In",
    "Journal",
    "About / Safety"
])


# -----------------------------
# Tab 1: Coach Chat
# -----------------------------

with tab1:
    st.header("Relationship Coach Chat")

    st.markdown(
        """
        Use this when you had an argument, feel hurt, need help understanding the situation,
        or want to prepare for a calmer conversation.
        """
    )

    situation_type = st.selectbox(
        "What are you here for today?",
        SITUATION_OPTIONS
    )

    st.markdown("### Optional 3-step input")
    happened_text = st.text_area(
        "1. What happened",
        height=90,
        placeholder="Facts only: what happened, when, and what was said or done."
    )
    felt_text = st.text_area(
        "2. What you felt",
        height=90,
        placeholder="Use I feel language, for example: I felt hurt and disconnected."
    )
    need_text = st.text_area(
        "3. What you need",
        height=90,
        placeholder="State one clear relationship need or request."
    )

    user_message = st.chat_input("Tell me what happened...")

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
# Tab 2: Rewrite Message
# -----------------------------

with tab2:
    st.header("Help Me Say This Better")

    st.markdown(
        """
        Paste an angry, emotional, confusing, or difficult message.
        The app will rewrite it into a calmer relationship-focused version.
        """
    )

    original_message = st.text_area(
        "Original message",
        height=160,
        placeholder="Paste your message here..."
    )

    rewrite_style = st.radio(
        "Rewrite style",
        ["Calm", "Loving", "Boundary", "Apology", "Short Text"],
        horizontal=True
    )

    if st.button("Rewrite Message"):
        rewritten = rewrite_message(original_message, rewrite_style)
        st.subheader("Rewritten Version")
        st.markdown(rewritten)

        st.download_button(
            label="Download rewritten message",
            data=rewritten,
            file_name="relationship_message_rewrite.txt",
            mime="text/plain"
        )


# -----------------------------
# Tab 3: Weekly Check-In
# -----------------------------

with tab3:
    st.header("Weekly Relationship Check-In")

    st.markdown(
        """
        This is a simple weekly reflection tool. It does not diagnose your relationship.
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

    if st.button("Generate Weekly Summary"):
        summary = create_weekly_summary(
            connection_score,
            communication_score,
            trust_score,
            conflict_score,
            weekly_notes
        )

        st.markdown(summary)

        st.download_button(
            label="Download weekly summary",
            data=summary,
            file_name="weekly_relationship_checkin.txt",
            mime="text/plain"
        )


# -----------------------------
# Tab 4: Journal
# -----------------------------

with tab4:
    st.header("Relationship Journal")

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


# -----------------------------
# Tab 5: About / Safety
# -----------------------------

with tab5:
    st.header("About RelationshipPath AI")

    st.markdown(
        """
        **RelationshipPath AI** is an early MVP for an AI relationship-coaching app.

        It helps users:
        - slow down emotional conflict
        - rewrite difficult messages
        - prepare for conversations
        - reflect on relationship patterns
        - set healthier boundaries
        - do weekly relationship check-ins

        It does **not** provide:
        - licensed therapy
        - couples therapy
        - medical diagnosis
        - psychiatric care
        - legal advice
        - emergency support
        """
    )

    st.subheader("Safety Policy")

    st.markdown(
        """
        The app should redirect instead of coaching when the user describes:

        - immediate danger
        - domestic violence
        - threats
        - coercion
        - stalking
        - self-harm
        - harm to others
        - weapon involvement
        - manipulation requests
        - revenge requests
        - attempts to control a partner

        The app should also avoid diagnosing partners.
        Instead of saying:

        > “Your partner is a narcissist.”

        It should say:

        > “That behavior sounds painful and possibly unhealthy. Let’s focus on what happened, how it affected you, and what boundary you may need.”
        """
    )

    st.subheader("Future Upgrade Path")

    st.markdown(
        """
        Version 2 can add:
        - real AI API responses
        - user login
        - saved journal database
        - couples mode
        - two-person shared check-ins
        - relationship pattern dashboard
        - therapist referral directory
        - human coach escalation
        - privacy controls
        - delete-my-data button
        """
    )