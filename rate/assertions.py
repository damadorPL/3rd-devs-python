# rate/assertions.py
import json

def _validate_base_structure(output_str: str) -> dict:
    """Helper to parse the output and validate basic JSON structure and types."""
    try:
        parsed = json.loads(output_str)
    except json.JSONDecodeError:
        raise ValueError("Output is not valid JSON")

    if not isinstance(parsed.get("reason"), str) or not isinstance(parsed.get("score"), (int, float)):
        raise ValueError("Output should have a string 'reason' and a numeric 'score'")
    return parsed

def assert_hardware_score_low_mid(output: str, context: dict) -> bool:
    """Asserts score is 0-0.5 and reason mentions 'hardware'."""
    parsed = _validate_base_structure(output)
    score = parsed["score"]
    reason_lower = parsed["reason"].lower()

    if not (0 <= score <= 0.5):
        raise ValueError(f"Expected score between 0 and 0.5, got {score}")
    if "hardware" not in reason_lower:
        raise ValueError("Reason should mention hardware")
    return True

def assert_apps_software_score_low(output: str, context: dict) -> bool:
    """Asserts score is 0.1-0.4 and reason mentions 'apps' or 'software'."""
    parsed = _validate_base_structure(output)
    score = parsed["score"]
    reason_lower = parsed["reason"].lower()

    if not (0.1 <= score <= 0.4):
        raise ValueError(f"Expected low score (0.1-0.4), got {score}")
    if "apps" not in reason_lower and "software" not in reason_lower:
        raise ValueError("Reason should mention apps or software, not hardware")
    return True

def assert_wolfram_ted_score_high(output: str, context: dict) -> bool:
    """Asserts score is 0.9-1.0 and reason mentions 'wolfram' and 'ted'."""
    parsed = _validate_base_structure(output)
    score = parsed["score"]
    reason_lower = parsed["reason"].lower()

    if not (0.9 <= score <= 1.0):
        raise ValueError(f"Expected very high score (0.9-1.0), got {score}")
    if "wolfram" not in reason_lower or "ted" not in reason_lower:
        raise ValueError("Reason should mention Stephen Wolfram and TED talk")
    return True

def assert_specific_hardware_score_low_mid(output: str, context: dict) -> bool:
    """Asserts score is 0.1-0.5 and reason mentions 'hardware'."""
    parsed = _validate_base_structure(output)
    score = parsed["score"]
    reason_lower = parsed["reason"].lower()

    if not (0.1 <= score <= 0.5):
        raise ValueError(f"Expected medium score (0.1-0.5), got {score}")
    if "hardware" not in reason_lower:
        raise ValueError("Reason should mention hardware")
    return True

def assert_wolfram_not_ted_score_medium_low(output: str, context: dict) -> bool:
    """Asserts score is 0.1-0.5 and reason mentions 'wolfram' and 'ted' (contextually that it's not a TED talk)."""
    parsed = _validate_base_structure(output)
    score = parsed["score"]
    reason_lower = parsed["reason"].lower()

    if not (0.1 <= score <= 0.5):
        raise ValueError(f"Expected medium-low score (0.1-0.5), got {score}")
    if "wolfram" not in reason_lower:
        raise ValueError("Reason should mention Stephen Wolfram")
    if "ted" not in reason_lower: # Checks if 'ted' is mentioned in the reason.
        raise ValueError("Reason should mention that this is not a TED talk (implying 'ted' should be in the string)")
    return True
