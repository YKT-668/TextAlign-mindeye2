"""Validation for counterfactual candidate records."""

ALLOWED_TYPES = {"object", "attribute", "relation", "random", "clip_nearest"}
REQUIRED_FIELDS = {"image_id", "positive", "negative", "negative_type"}


def validate_candidate_negative(record: dict) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(record["image_id"], (int, str)):
        raise TypeError("image_id must be an integer or string")
    for field in ("positive", "negative"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if record["negative_type"] not in ALLOWED_TYPES:
        raise ValueError(f"unsupported negative_type: {record['negative_type']}")
