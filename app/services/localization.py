"""Localization service — Hebrew labels, explanations, actions, and tips.

A pure mapping layer that translates English diagnosis types, intervention
types, and score names into Hebrew strings for the frontend UI.

Requirements implemented: 11.2, 11.3, 11.4.
"""

from __future__ import annotations

from app.models.schemas import LocalizedDiagnosis, LocalizedIntervention


# ---------------------------------------------------------------------------
# Diagnosis type → Hebrew localization
# ---------------------------------------------------------------------------

DIAGNOSIS_MAP: dict[str, dict] = {
    "low_cohesion": {
        "label_he": "קוהרנטיות נמוכה",
        "explanation_he": "הטקסט חסר קשרים ברורים בין המשפטים, מה שמקשה על הקורא לעקוב אחרי הרצף הלוגי.",
        "actions_he": [
            "הוסיפו מילות קישור בין משפטים (לכן, בנוסף, עם זאת)",
            "חזרו על מילות מפתח ממשפט למשפט כדי ליצור רצף",
            "ודאו שכל משפט נובע מהמשפט הקודם",
        ],
        "tip_he": "הוסיפו מילות קישור כמו 'לכן', 'בנוסף'",
    },
    "sentence_over_complexity": {
        "label_he": "משפטים מורכבים מדי",
        "explanation_he": "הטקסט מכיל משפטים ארוכים ומסובכים עם מבנה תחבירי עמוק מדי.",
        "actions_he": [
            "פצלו משפטים ארוכים לשניים או שלושה משפטים קצרים",
            "הפחיתו את מספר פסוקיות המשנה בכל משפט",
            "פשטו מבנים תחביריים מקוננים",
        ],
        "tip_he": "פצלו משפטים ארוכים לשניים-שלושה קצרים",
    },
    "low_lexical_diversity": {
        "label_he": "אוצר מילים מצומצם",
        "explanation_he": "הטקסט משתמש במילים חוזרות ואינו מגוון מספיק מבחינה לקסיקלית.",
        "actions_he": [
            "החליפו מילים חוזרות במילים נרדפות",
            "הרחיבו את אוצר המילים באמצעות קריאה",
            "השתמשו בצורות שונות של אותו שורש",
        ],
        "tip_he": "החליפו מילים חוזרות במילים נרדפות",
    },
    "pronoun_overuse": {
        "label_he": "שימוש יתר בכינויי גוף",
        "explanation_he": "הטקסט מכיל ריכוז גבוה של כינויי גוף, מה שעלול ליצור עמימות.",
        "actions_he": [
            "החליפו כינויים עמומים בשמות עצם מפורשים",
            "הפחיתו את צפיפות הכינויים במשפטים עוקבים",
            "ודאו שכל כינוי מתייחס בבירור לשם עצם קודם",
        ],
        "tip_he": "החליפו כינויים בשמות מפורשים",
    },
    "structural_inconsistency": {
        "label_he": "חוסר עקביות מבנית",
        "explanation_he": "המשפטים בטקסט שונים מאוד באורכם ובמבנם, מה שפוגע בקריאות.",
        "actions_he": [
            "שמרו על אורך אחיד יחסית של משפטים",
            "הפחיתו שונות בדפוסים תחביריים",
            "התאימו את מבני המשפטים לקריאות עקבית",
        ],
        "tip_he": "שמרו על אורך ומבנה דומים",
    },
    "low_morphological_richness": {
        "label_he": "מגוון מורפולוגי נמוך",
        "explanation_he": "הטקסט אינו מנצל את העושר המורפולוגי של העברית ומשתמש בדפוסים חוזרים.",
        "actions_he": [
            "השתמשו בבניינים שונים (פיעל, הפעיל, התפעל)",
            "גוונו את צורות הפועל והשם",
            "נסו להשתמש בסמיכות ובצורות מורפולוגיות מגוונות",
        ],
        "tip_he": "השתמשו בבניינים שונים",
    },
    "fragmented_writing": {
        "label_he": "כתיבה מקוטעת",
        "explanation_he": "הטקסט מכיל משפטים קצרים ומנותקים שאינם יוצרים זרימה רציפה.",
        "actions_he": [
            "חברו משפטים קצרים למשפטים שלמים ומורכבים יותר",
            "הוסיפו ביטויי חיבור בין קטעים מנותקים",
            "הרחיבו משפטים קצרצרים בפסוקיות נוספות",
        ],
        "tip_he": "חברו משפטים קצרים למשפטים שלמים",
    },
    "punctuation_deficiency": {
        "label_he": "בעיות פיסוק",
        "explanation_he": "הטקסט חסר סימני פיסוק חיוניים או משתמש בהם באופן שגוי.",
        "actions_he": [
            "בדקו שכל משפט מסתיים בנקודה",
            "הוסיפו פסיקים להפרדת פסוקיות",
            "השתמשו בסימני פיסוק לשיפור זרימת הטקסט",
        ],
        "tip_he": "בדקו נקודות בסוף משפטים ופסיקים",
    },
}


# ---------------------------------------------------------------------------
# Score name → Hebrew label
# ---------------------------------------------------------------------------

SCORE_NAME_MAP: dict[str, str] = {
    "difficulty": "קושי",
    "style": "סגנון",
    "fluency": "שטף",
    "cohesion": "קוהרנטיות",
    "complexity": "מורכבות",
}


# ---------------------------------------------------------------------------
# Intervention type → Hebrew localization
# ---------------------------------------------------------------------------

INTERVENTION_MAP: dict[str, dict] = {
    "vocabulary_expansion": {
        "label_he": "הרחבת אוצר מילים",
        "actions_he": [
            "תרגלו החלפת מילים חוזרות במילים נרדפות",
            "עודדו שימוש בצורות מילים מגוונות",
            "תרגלו ניסוח מחדש של משפטים עם אוצר מילים חדש",
        ],
        "exercises_he": [
            "שכתבו פסקה תוך החלפת מילים חוזרות במילים נרדפות",
            "בנו רשימת אוצר מילים אישית מטקסטים שקראתם",
            "השלימו משפטים חסרים תוך שימוש במילים מגוונות",
        ],
    },
    "pronoun_clarification": {
        "label_he": "הבהרת כינויי גוף",
        "actions_he": [
            "החליפו כינויים עמומים בשמות עצם מפורשים",
            "הפחיתו צפיפות כינויים במשפטים עוקבים",
            "הבהירו שרשראות הפניה בין פסקאות",
        ],
        "exercises_he": [
            "זהו את כל הכינויים בפסקה והחליפו את העמומים",
            "שכתבו קטעים תוך הפחתת יחס כינויים לשמות עצם",
            "התאימו כינויים לשמות העצם שאליהם הם מתייחסים בטקסטים לדוגמה",
        ],
    },
    "sentence_simplification": {
        "label_he": "פישוט משפטים",
        "actions_he": [
            "פרקו משפטים ארוכים ליחידות קצרות יותר",
            "הפחיתו קינון של פסוקיות משנה",
            "פשטו מבנים תחביריים מורכבים",
        ],
        "exercises_he": [
            "שכתבו משפטים מעל 30 מילים לשניים או יותר",
            "זהו וחלצו פסוקיות משובצות",
            "הפחיתו עומק עץ תחבירי על ידי ארגון מחדש של משפטים מורכבים",
        ],
    },
    "cohesion_improvement": {
        "label_he": "שיפור קוהרנטיות",
        "actions_he": [
            "הוסיפו מילות קישור בין משפטים",
            "הגבירו חפיפה לקסיקלית בין משפטים סמוכים",
            "השתמשו בביטויי מעבר לקישור רעיונות",
        ],
        "exercises_he": [
            "הכניסו מילות קישור מתאימות לקטע טקסט",
            "שכתבו פסקאות לשיפור החפיפה בין משפט למשפט",
            "זהו מעברים חסרים והוסיפו אותם",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def localize_diagnosis(type: str, severity: float) -> LocalizedDiagnosis:
    """Localize a diagnosis type to Hebrew.

    Parameters
    ----------
    type : str
        One of the 8 recognized diagnosis types.
    severity : float
        Severity score in [0.0, 1.0].

    Returns
    -------
    LocalizedDiagnosis
        Pydantic model with Hebrew label, explanation, actions, and tip.

    Raises
    ------
    KeyError
        If *type* is not a recognized diagnosis type.
    """
    entry = DIAGNOSIS_MAP[type]
    return LocalizedDiagnosis(
        type=type,
        severity=severity,
        label_he=entry["label_he"],
        explanation_he=entry["explanation_he"],
        actions_he=list(entry["actions_he"]),
        tip_he=entry["tip_he"],
    )


def localize_intervention(intervention_dict: dict) -> LocalizedIntervention:
    """Localize an intervention to Hebrew.

    Parameters
    ----------
    intervention_dict : dict
        Dictionary with keys ``type``, ``priority``, ``target_diagnosis``.
        The ``type`` must be one of the 4 recognized intervention types.

    Returns
    -------
    LocalizedIntervention
        Pydantic model with Hebrew actions and exercises.

    Raises
    ------
    KeyError
        If the intervention type is not recognized.
    """
    itype = intervention_dict["type"]
    entry = INTERVENTION_MAP[itype]
    return LocalizedIntervention(
        type=itype,
        priority=intervention_dict.get("priority", 0.0),
        target_diagnosis=intervention_dict.get("target_diagnosis", ""),
        actions_he=list(entry["actions_he"]),
        exercises_he=list(entry["exercises_he"]),
    )


def localize_score_name(key: str) -> str:
    """Return the Hebrew label for a score name.

    Parameters
    ----------
    key : str
        One of: difficulty, style, fluency, cohesion, complexity.

    Returns
    -------
    str
        Hebrew label.

    Raises
    ------
    KeyError
        If *key* is not a recognized score name.
    """
    return SCORE_NAME_MAP[key]


def get_diagnosis_types() -> list[str]:
    """Return the list of all 8 recognized diagnosis type strings."""
    return list(DIAGNOSIS_MAP.keys())
