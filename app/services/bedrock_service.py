"""Bedrock service — Amazon Bedrock integration for AI rewrite suggestions.

Manages a boto3 Bedrock Runtime client for invoking models and a Bedrock
management client for listing available foundation models.  Exposes methods
for generating Hebrew rewrite suggestions, listing models, and updating the
active model.

Requirements implemented: 3.1, 3.2, 3.3, 3.4, 12.3, 12.5.
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from app.config import settings
from app.models.schemas import ModelInfo
from app.services.localization import DIAGNOSIS_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class BedrockUnavailableError(Exception):
    """Raised when the Bedrock service is unreachable or returns an error."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an expert Hebrew writing coach. The user's text has a specific linguistic issue that needs fixing.

Issue diagnosed: {diagnosis_label_he}
Why it's a problem: {explanation_he}

Specific actions to take:
{actions_list}

Quick tip: {tip_he}

Original text:
{original_text}

Rewrite the text applying the specific actions listed above. Fix ONLY the diagnosed issue — keep the meaning, tone, and content the same. The result should be natural Hebrew prose.
Respond only with the rewritten Hebrew text, no explanations or commentary."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class BedrockService:
    """Amazon Bedrock integration for the Hebrew Writing Coach.

    Attributes
    ----------
    _runtime_client : boto3 client
        Bedrock Runtime client used for ``invoke_model`` calls.
    _mgmt_client : boto3 client
        Bedrock management client used for ``list_foundation_models``.
    _model_id : str
        Currently active Bedrock model identifier.
    """

    def __init__(self) -> None:
        session_kwargs: dict = {}
        if settings.AWS_PROFILE:
            session_kwargs["profile_name"] = settings.AWS_PROFILE
        session = boto3.Session(**session_kwargs)

        self._runtime_client = session.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
        )
        self._mgmt_client = session.client(
            "bedrock",
            region_name=settings.AWS_REGION,
        )
        self._model_id: str = settings.BEDROCK_MODEL_ID

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def build_prompt(self, text: str, diagnosis_type: str) -> str:
        """Build the Hebrew rewrite prompt for a given diagnosis type.

        This is exposed as a public method so that property tests can
        verify prompt content without calling Bedrock (Property 8).

        Parameters
        ----------
        text:
            Original Hebrew sentence(s) to rewrite.
        diagnosis_type:
            One of the 8 recognized diagnosis types.

        Returns
        -------
        str
            The fully formatted prompt string.

        Raises
        ------
        KeyError
            If *diagnosis_type* is not a recognized diagnosis type.
        """
        entry = DIAGNOSIS_MAP[diagnosis_type]
        actions_list = "\n".join(f"- {a}" for a in entry["actions_he"])
        return _PROMPT_TEMPLATE.format(
            diagnosis_label_he=entry["label_he"],
            explanation_he=entry["explanation_he"],
            actions_list=actions_list,
            tip_he=entry["tip_he"],
            original_text=text,
        )

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    def rewrite(self, text: str, diagnosis_type: str, context: str = "") -> dict:
        """Generate a rewrite suggestion via Amazon Bedrock.

        Parameters
        ----------
        text:
            Original Hebrew sentence(s) to rewrite.
        diagnosis_type:
            One of the 8 recognized diagnosis types.
        context:
            Optional surrounding text for additional context.

        Returns
        -------
        dict with keys ``suggestion`` (str) and ``model_used`` (str).

        Raises
        ------
        BedrockUnavailableError
            If the Bedrock service is unreachable or returns an error.
        KeyError
            If *diagnosis_type* is not a recognized diagnosis type.
        """
        prompt = self.build_prompt(text, diagnosis_type)

        # Build request body based on model provider
        if "anthropic" in self._model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            })
        elif "nova" in self._model_id:
            body = json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]},
                ],
                "inferenceConfig": {"maxTokens": 1024, "temperature": 0.7},
            })
        elif "titan" in self._model_id:
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1024,
                    "temperature": 0.7,
                },
            })
        else:
            # Default: Anthropic Messages format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            })

        try:
            response = self._runtime_client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            response_body = json.loads(response["body"].read())

            # Extract text based on model provider response format
            if "anthropic" in self._model_id:
                suggestion = response_body["content"][0]["text"].strip()
            elif "nova" in self._model_id:
                suggestion = response_body["output"]["message"]["content"][0]["text"].strip()
            elif "titan" in self._model_id:
                suggestion = response_body["results"][0]["outputText"].strip()
            else:
                suggestion = response_body.get("content", [{}])[0].get("text", "").strip()

            return {
                "suggestion": suggestion,
                "model_used": self._model_id,
            }

        except (ClientError, EndpointConnectionError, ReadTimeoutError) as exc:
            logger.error("Bedrock call failed: %s", exc)
            raise BedrockUnavailableError(
                "שירות השכתוב אינו זמין כרגע, נסו שוב מאוחר יותר"
            ) from exc

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """List available Bedrock foundation models for text generation.

        Returns a curated list of Anthropic and Amazon models, supplemented
        by any additional models discovered via the Bedrock API.
        """
        # Curated list of Anthropic and Amazon text models
        # Uses cross-region inference profile IDs — prefix matches AWS_REGION
        region = settings.AWS_REGION
        if region.startswith("eu"):
            prefix = "eu"
        elif region.startswith("ap"):
            prefix = "ap"
        else:
            prefix = "us"

        curated: list[ModelInfo] = [
            # Anthropic — latest models
            ModelInfo(model_id=f"{prefix}.anthropic.claude-sonnet-4-5-20250929-v1:0", model_name="Claude Sonnet 4.5", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-sonnet-4-6", model_name="Claude Sonnet 4.6", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-sonnet-4-20250514-v1:0", model_name="Claude Sonnet 4", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-opus-4-7", model_name="Claude Opus 4.7", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-opus-4-6-v1", model_name="Claude Opus 4.6", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-opus-4-5-20251101-v1:0", model_name="Claude Opus 4.5", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-opus-4-20250514-v1:0", model_name="Claude Opus 4", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-opus-4-1-20250805-v1:0", model_name="Claude Opus 4.1", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-haiku-4-5-20251001-v1:0", model_name="Claude Haiku 4.5", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-3-7-sonnet-20250219-v1:0", model_name="Claude 3.7 Sonnet", provider="Anthropic"),
            ModelInfo(model_id=f"{prefix}.anthropic.claude-3-5-haiku-20241022-v1:0", model_name="Claude 3.5 Haiku", provider="Anthropic"),
            # Amazon Nova
            ModelInfo(model_id=f"{prefix}.amazon.nova-pro-v1:0", model_name="Nova Pro", provider="Amazon"),
            ModelInfo(model_id=f"{prefix}.amazon.nova-lite-v1:0", model_name="Nova Lite", provider="Amazon"),
            ModelInfo(model_id=f"{prefix}.amazon.nova-micro-v1:0", model_name="Nova Micro", provider="Amazon"),
            ModelInfo(model_id="us.amazon.nova-premier-v1:0", model_name="Nova Premier", provider="Amazon"),
            ModelInfo(model_id="us.amazon.nova-2-lite-v1:0", model_name="Nova 2 Lite", provider="Amazon"),
        ]

        return curated

    def update_model(self, model_id: str) -> None:
        """Update the active Bedrock model identifier.

        Parameters
        ----------
        model_id:
            The new model identifier to use for subsequent rewrite calls.
        """
        self._model_id = model_id
        logger.info("BedrockService model updated to %s", model_id)
