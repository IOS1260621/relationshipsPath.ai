APP_NAME = "RelationshipPath AI"
MODALITY_NAME = "Relationships"

MAX_CHAT_MESSAGES = 100
MAX_JOURNAL_ENTRIES = 200

SAFETY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "hurt myself", "hurt her", "hurt him", "hurt them", "kill her", "kill him",
    "violence", "hit me", "hit her", "hit him", "choked", "strangled",
    "threatened", "threat", "weapon", "gun", "knife",
    "abuse", "abusive", "domestic violence", "scared of him", "scared of her",
    "afraid of him", "afraid of her", "stalking", "stalk",
    "forced me", "coerced", "blackmail"
]

MANIPULATION_KEYWORDS = [
    "make her stay", "make him stay", "control her", "control him",
    "manipulate", "gaslight", "revenge", "punish her", "punish him",
    "spy on", "track her", "track him", "hack", "hide cheating",
    "make jealous", "test her loyalty", "test his loyalty"
]

DIAGNOSIS_KEYWORDS = [
    "narcissist", "borderline", "bipolar", "psychopath", "sociopath",
    "crazy", "insane", "mental illness", "diagnose"
]

SITUATION_OPTIONS = [
    "I had an argument",
    "I feel hurt",
    "I want to understand my partner better",
    "I need help saying something",
    "I want to rebuild trust",
    "I am thinking about ending the relationship",
    "We want a weekly check-in",
    "I just need to vent"
]

SITUATION_FOCUS = {
    "I had an argument": "Focus on one specific moment from the argument and one sentence you wish you had said more calmly.",
    "I feel hurt": "Focus on naming the exact action that hurt you and one need underneath that feeling.",
    "I want to understand my partner better": "Focus on one open-ended question you can ask instead of making assumptions.",
    "I need help saying something": "Focus on a short message with facts, feelings, and one clear request.",
    "I want to rebuild trust": "Focus on one repeated behavior that builds safety over time.",
    "I am thinking about ending the relationship": "Focus on your non-negotiables, your values, and what a respectful next conversation looks like.",
    "We want a weekly check-in": "Focus on one topic, one appreciation, and one improvement for this week.",
    "I just need to vent": "Focus on emotional clarity first, then decide what outcome you want before sending any message."
}

REWRITE_TEMPLATES = {
    "Calm": """
I want to talk about this calmly.

What happened affected me, and I do not want this to turn into a fight. I would like us to slow down, understand each other, and talk about what we both need.

Here is what I am trying to say:

\"{base}\"

Can we talk about this when we both have a little space to listen?
""",
    "Loving": """
I care about you and I care about us.

I do not want to attack you or make this worse. I want to share what I have been feeling and understand your side too.

What I am trying to say is:

\"{base}\"

Can we talk in a way where we both feel heard?
""",
    "Boundary": """
I want to be respectful, but I also need to be honest about my boundary.

What happened does not feel okay to me, and I do not want to keep repeating the same pattern.

What I am trying to communicate is:

\"{base}\"

I am open to talking, but I need the conversation to stay respectful and honest.
""",
    "Apology": """
I want to apologize for my part in this.

I may not have expressed myself in the best way, and I do not want my words to make things worse. I care about how this affected you.

What I was trying to say was:

\"{base}\"

I am sorry for the way I handled it. I would like to talk and do better.
""",
    "Short Text": """
I do not want us to fight. I care about us, and I want to talk calmly. What I was trying to say is: \"{base}\" Can we talk when we are both ready to listen?
"""
}
