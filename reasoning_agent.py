# autonomous/reasoning_agent.py

import os
import json
import logging
from typing import Any, Dict, List
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# OLLAMA CONFIGURATION
# -----------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gemma3:1b")

DEFAULT_SYSTEM_PROMPT = (
    "You are Veri-ADAM Reasoner. Respond ONLY with JSON following this schema: "
    '{"decision":"escalate|monitor|ignore", '
    '"reason":["..."], '
    '"recommended_action":"...", '
    '"confidence":float}'
)

# -----------------------------------------------------
# RESPONSE EXTRACTION
# -----------------------------------------------------

def _extract_text_from_ollama_response(resp: Any) -> str:
    """
    Extract text from an Ollama response object in a robust way.
    Handles different response shapes.
    """

    try:
        if isinstance(resp, dict):
            # Standard Ollama response
            if "response" in resp:
                return resp["response"]

            # Some wrappers return 'message'
            if "message" in resp:
                msg = resp["message"]
                if isinstance(msg, dict):
                    return msg.get("content", "")

            # fallback fields
            for key in ("text", "result", "output"):
                if resp.get(key):
                    return resp.get(key)

            return json.dumps(resp)

    except Exception:
        pass

    return str(resp)

# -----------------------------------------------------
# OLLAMA CALL
# -----------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """
    Send prompt to local Ollama model.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",   # enforce JSON output
                "options": {
                    "temperature": 0.0,
                    "num_predict": 600
                }
            },
            timeout=300
        )

        response.raise_for_status()

    except Exception as e:
        logger.exception("Ollama call failed")
        raise RuntimeError(f"Ollama call error: {e}") from e

    try:
        resp_json = response.json()
    except Exception:
        raise RuntimeError("Invalid JSON returned from Ollama")

    return _extract_text_from_ollama_response(resp_json)

# -----------------------------------------------------
# MAIN REASONING FUNCTION
# -----------------------------------------------------

def call_gemma_reasoner(
    anomalies: List[Dict],
    context_readings: List[Dict] | None = None
) -> Dict:
    """
    Use a locally deployed Gemma model (via Ollama)
    to reason about anomalies.

    Parameters
    ----------
    anomalies : list
        flagged sensor readings

    context_readings : list
        recent readings for temporal context

    Returns
    -------
    dict
        {
          "decision": "...",
          "reason": [...],
          "recommended_action": "...",
          "confidence": float
        }
    """

    if context_readings is None:
        context_readings = []

    # -------------------------------------------------
    # BUILD USER PAYLOAD
    # -------------------------------------------------

    user_payload = {
        "anomalies": anomalies,
        "context_readings": context_readings
    }

    user_text = json.dumps(user_payload, default=str)

    logger.info("Using local reasoning model: %s", MODEL_NAME)

    prompt = f"""
SYSTEM:
{DEFAULT_SYSTEM_PROMPT}

INPUT:
{user_text}
"""

    # -------------------------------------------------
    # CALL LLM
    # -------------------------------------------------

    raw_text = _call_ollama(prompt)

    logger.info("Raw LLM output (truncated): %s", raw_text[:2000])

    # -------------------------------------------------
    # PARSE JSON OUTPUT
    # -------------------------------------------------

    try:

        text = raw_text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            json_blob = text[start:end + 1]
        else:
            json_blob = text

        result = json.loads(json_blob)

    except Exception as e:
        logger.exception("Failed to parse model JSON output")
        raise ValueError("Unable to parse model output as JSON") from e

    # -------------------------------------------------
    # VALIDATE OUTPUT
    # -------------------------------------------------

    if "decision" not in result or "confidence" not in result:
        raise ValueError("Model output missing required fields")

    return result