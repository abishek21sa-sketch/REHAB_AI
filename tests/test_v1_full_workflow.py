from pathlib import Path

from rehab_ai.clinical_workflow import build_clinical_episode, episode_markdown_report
from rehab_ai.biomechanics.assessment_protocols import assessment_suite
from rehab_ai.vision.motion_session import analyze_landmark_csv


def test_clinical_episode_is_closed_loop():
    d=build_clinical_episode(impairment=.45,phenotype='balanced',weeks=5,seed=42).to_dict()
    assert len(d['weeks'])==5
    assert d['learning']['replans']==5
    assert d['learning']['final_uncertainty'] < d['learning']['initial_uncertainty']
    assert d['learning']['envelope_violations']==0
    assert set(d['assessments'])=={'gait','balance','mobility','endurance'}
    assert d['next_control']['action'] != 'NOT AVAILABLE'
    kinds={x['kind'] for x in d['evidence_ledger']}
    assert {'CALCULATED','ESTIMATED','PREDICTED','SIMULATED','OPTIMIZED'} <= kinds


def test_episode_report_preserves_evidence_language():
    s=build_clinical_episode(impairment=.45,phenotype='balanced',weeks=4,seed=42)
    text=episode_markdown_report(s)
    assert 'EXTERNAL VALIDATION PENDING' in text
    assert 'OPTIMIZED' in text
    assert 'Weekly Decision Ledger' in text


def test_assessment_suite_is_multi_protocol():
    suite=assessment_suite(gait_speed_mps=.82,cadence_spm=108,knee_rom_deg=34,symmetry_index=.31,margin_of_stability_m=.018,fatigue=.3,pain=.2,motor_capacity=.58)
    assert len(suite)==4
    assert all(0<=x.score<=1 for x in suite.values())
    assert suite['gait'].primary_limitation=='dynamic stability'


def test_motion_ingestion_returns_trace_and_readiness():
    p=Path(__file__).parents[1]/'sample_data'/'observed_landmark_session.csv'
    d=analyze_landmark_csv(p.read_text())
    assert len(d['trace']['time_s'])>10
    assert d['readiness']['state_estimation']=='READY_FOR_KINEMATIC_FEATURES'
    assert d['readiness']['full_gait_assessment']=='INSUFFICIENT_SIGNALS'
