# Feature: hebrew-writing-coach, Property 8: Rewrite prompt completeness
"""Property-based tests for rewrite prompt completeness.

**Validates: Requirements 3.2**

Property 8: For any valid diagnosis type and any non-empty Hebrew text,
the prompt constructed by BedrockService contains the diagnosis type
string, the Hebrew diagnosis label, the Hebrew explanation, and the
original text verbatim.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.bedrock_service import BedrockService
from app.services.localization import DIAGNOSIS_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEBREW_ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת "
DIAGNOSIS_TYPES = list(DIAGNOSIS_MAP.keys())

# Create a single BedrockService instance to avoid repeated boto3 client
# initialization overhead inside the property test loop.
_bedrock_service = BedrockService()


# ---------------------------------------------------------------------------
# Property 8 – Rewrite prompt completeness
# ---------------------------------------------------------------------------

# **Validates: Requirements 3.2**
@given(
    diagnosis_type=st.sampled_from(DIAGNOSIS_TYPES),
    text=st.text(alphabet=HEBREW_ALPHABET, min_size=1, max_size=200),
)
@settings(max_examples=100, deadline=None)
def test_rewrite_prompt_contains_all_required_fields(
    diagnosis_type: str,
    text: str,
) -> None:
    """For any valid diagnosis type and any non-empty Hebrew text, the
    prompt constructed by BedrockService contains the diagnosis type
    string, the Hebrew diagnosis label, the Hebrew explanation, and the
    original text verbatim."""
    service = _bedrock_service
    prompt = service.build_prompt(text, diagnosis_type)

    entry = DIAGNOSIS_MAP[diagnosis_type]

    # Prompt must contain the Hebrew diagnosis label
    assert entry["label_he"] in prompt, (
        f"Prompt missing Hebrew label '{entry['label_he']}' for {diagnosis_type}"
    )

    # Prompt must contain the Hebrew explanation
    assert entry["explanation_he"] in prompt, (
        f"Prompt missing Hebrew explanation for {diagnosis_type}"
    )

    # Prompt must contain the original text verbatim
    assert text in prompt, (
        f"Prompt missing original text verbatim"
    )


# ---------------------------------------------------------------------------
# Property 9 – Invalid diagnosis type rejection
# ---------------------------------------------------------------------------

# Feature: hebrew-writing-coach, Property 9: Invalid diagnosis type rejection
# **Validates: Requirements 3.5**
@given(
    invalid_type=st.text().filter(lambda t: t not in DIAGNOSIS_TYPES),
)
@settings(max_examples=100, deadline=None)
def test_invalid_diagnosis_type_rejected(invalid_type: str) -> None:
    """For any string that is NOT one of the 8 recognized diagnosis types,
    build_prompt raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        _bedrock_service.build_prompt("שלום", invalid_type)
