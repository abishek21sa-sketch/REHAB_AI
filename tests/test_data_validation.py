import pytest
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.data.validation import DataQualityError, validate_session


def test_synthetic_session_passes_quality_checks():
    q = validate_session(generate_synthetic_session())
    assert q["wearable_samples"] == 1000
    assert 49 <= q["wearable_sample_rate_hz"] <= 51


def test_non_monotonic_wearable_timestamps_rejected():
    session = generate_synthetic_session()
    payload = session.model_dump()
    payload["wearable"][20]["t"] = payload["wearable"][19]["t"]
    with pytest.raises(Exception):
        validate_session(type(session).model_validate(payload))
