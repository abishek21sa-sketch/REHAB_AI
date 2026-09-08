from pathlib import Path
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.pipeline import RehabDecisionPipeline


def test_twin_persists_and_computes_deltas(tmp_path: Path):
    p = RehabDecisionPipeline(database_path=str(tmp_path / "test.db"))
    a = p.run(generate_synthetic_session(patient_id="P1", impairment=0.6, seed=1))
    b = p.run(generate_synthetic_session(patient_id="P1", impairment=0.45, seed=2))
    assert a.twin.previous is None
    assert b.twin.previous is not None
    assert "functional_capacity" in b.twin.deltas
    assert len(p.repository.history("P1")) == 2
