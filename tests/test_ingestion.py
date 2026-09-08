import pytest
from rehab_ai.data.ingestion import SensorStreamBuffer, normalize_fhir_observation
from rehab_ai.domain import PoseFrame, WearableSample


def test_fhir_observation_adapter_is_explicit_and_typed():
    o = normalize_fhir_observation({
        'resourceType':'Observation',
        'subject':{'reference':'Patient/P-10'},
        'code':{'coding':[{'code':'6MWD'}]},
        'valueQuantity':{'value':355,'unit':'m'},
        'effectiveDateTime':'2026-08-11T12:00:00Z',
    })
    assert o.patient_id == 'P-10' and o.code == '6MWD' and o.value == 355
    with pytest.raises(ValueError):
        normalize_fhir_observation({'resourceType':'Patient'})


def test_stream_buffer_enforces_bounds_and_readiness():
    b = SensorStreamBuffer(max_wearable=20, max_pose=10)
    for i in range(22):
        b.add_wearable(WearableSample(t=i*.02, ax=0, ay=0, az=9.81))
    for i in range(11):
        b.add_pose(PoseFrame(t=i*.04,left_hip_y=1,right_hip_y=1,left_knee_angle=120,right_knee_angle=120,trunk_lean_deg=0))
    assert len(b.wearable) == 20 and len(b.pose) == 10 and b.ready()
