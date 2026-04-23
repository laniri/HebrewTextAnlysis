"""POST /api/exercise endpoint — multiple-choice rewrite exercises.

Uses Amazon Bedrock to generate 3 rewrite options for a diagnosed issue:
one correct fix, two with different problems. Returns the options with
explanations so the frontend can present a quiz.

Requirements: pedagogical practice feature.
"""

from __future__ import annotations

import json
import logging
import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.rewrite import bedrock_service
from app.services.bedrock_service import BedrockUnavailableError
from app.services.localization import DIAGNOSIS_MAP, get_diagnosis_types

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["exercise"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ExerciseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Problematic sentence(s)")
    diagnosis_type: str = Field(..., description="One of 8 diagnosis types")


class ExerciseOption(BaseModel):
    text: str
    is_correct: bool
    explanation_he: str


class ExerciseResponse(BaseModel):
    original_text: str
    diagnosis_label_he: str
    diagnosis_explanation_he: str
    tip_he: str
    options: list[ExerciseOption]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXERCISE_PROMPT = """\
You are a Hebrew writing coach creating a multiple-choice exercise.

The user's text has been diagnosed with "{diagnosis_type}".
Diagnosis: {diagnosis_label_he}
Explanation: {explanation_he}

Original sentence(s):
{original_text}

Generate exactly 3 rewrite options in Hebrew:
- Option 1: The CORRECT fix that resolves the diagnosed issue
- Option 2: A rewrite that has a DIFFERENT problem (explain what's wrong)
- Option 3: A rewrite that partially fixes the issue but introduces another problem (explain what's wrong)

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
  "options": [
    {{"text": "...", "is_correct": true, "explanation_he": "..."}},
    {{"text": "...", "is_correct": false, "explanation_he": "..."}},
    {{"text": "...", "is_correct": false, "explanation_he": "..."}}
  ]
}}

The explanations should be in Hebrew. Shuffle the order so the correct answer isn't always first."""


# ---------------------------------------------------------------------------
# Robust JSON parser for LLM output
# ---------------------------------------------------------------------------

import re

def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling common issues like invalid
    Unicode escapes, markdown code blocks, and unescaped quotes in
    Hebrew text (e.g. הרמטכ"ל)."""
    # Strip markdown code blocks
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Fix invalid \uXXXX escapes
    text = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', '', text)

    # Try strict parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try with strict=False
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Fix unescaped quotes inside JSON string values.
    # Strategy: find "text": "..." patterns and escape internal quotes.
    def _escape_value_quotes(m: re.Match) -> str:
        key = m.group(1)
        val = m.group(2)
        # Escape any unescaped quotes inside the value
        escaped = val.replace('\\"', '\x00').replace('"', '\\"').replace('\x00', '\\"')
        return f'"{key}": "{escaped}"'

    fixed = re.sub(
        r'"(text|explanation_he)":\s*"((?:[^"\\]|\\.)*(?:"(?![\s,}\]])(?:[^"\\]|\\.)*)*)"',
        _escape_value_quotes,
        text,
    )
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError:
        pass

    # Nuclear option: extract options via regex
    options = []
    # Find each option block
    for block in re.finditer(
        r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.|"(?![\s,}]))*?)"\s*,\s*'
        r'"is_correct"\s*:\s*(true|false)\s*,\s*'
        r'"explanation_he"\s*:\s*"((?:[^"\\]|\\.|"(?![\s,}]))*?)"\s*\}',
        text,
    ):
        opt_text = block.group(1).replace('\\"', '"').replace('\\n', '\n')
        is_correct = block.group(2) == 'true'
        explanation = block.group(3).replace('\\"', '"').replace('\\n', '\n')
        options.append({
            "text": opt_text,
            "is_correct": is_correct,
            "explanation_he": explanation,
        })

    if options:
        return {"options": options}

    raise json.JSONDecodeError("Could not parse LLM JSON output", text, 0)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/exercise", response_model=ExerciseResponse)
async def generate_exercise(request: ExerciseRequest) -> ExerciseResponse:
    """Generate a multiple-choice rewrite exercise."""
    valid_types = get_diagnosis_types()
    if request.diagnosis_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"סוג האבחנה אינו מוכר: {request.diagnosis_type}",
        )

    entry = DIAGNOSIS_MAP[request.diagnosis_type]

    prompt = _EXERCISE_PROMPT.format(
        diagnosis_type=request.diagnosis_type,
        diagnosis_label_he=entry["label_he"],
        explanation_he=entry["explanation_he"],
        original_text=request.text,
    )

    # Build request body based on model provider
    model_id = bedrock_service._model_id

    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        })
    elif "nova" in model_id:
        body = json.dumps({
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 2048, "temperature": 0.8},
        })
    else:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        })

    try:
        response = bedrock_service._runtime_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response["body"].read())

        # Extract text based on model
        if "anthropic" in model_id:
            raw_text = response_body["content"][0]["text"].strip()
        elif "nova" in model_id:
            raw_text = response_body["output"]["message"]["content"][0]["text"].strip()
        else:
            raw_text = response_body.get("content", [{}])[0].get("text", "").strip()

        # Parse JSON from the LLM response
        # Strip markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        parsed = _parse_llm_json(raw_text)
        options = [
            ExerciseOption(
                text=opt["text"],
                is_correct=opt["is_correct"],
                explanation_he=opt["explanation_he"],
            )
            for opt in parsed["options"]
        ]

        # Shuffle so the correct answer isn't always first
        random.shuffle(options)

        return ExerciseResponse(
            original_text=request.text,
            diagnosis_label_he=entry["label_he"],
            diagnosis_explanation_he=entry["explanation_he"],
            tip_he=entry["tip_he"],
            options=options,
        )

    except (BedrockUnavailableError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Exercise generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="לא הצלחנו ליצור תרגיל כרגע, נסו שוב",
        )
    except Exception as exc:
        logger.error("Exercise generation unexpected error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="שגיאה ביצירת התרגיל — בדקו הגדרות AWS",
        )
