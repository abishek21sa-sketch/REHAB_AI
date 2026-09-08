from pathlib import Path
import asyncio
import httpx
from rehab_ai.web.app import app
from rehab_ai.vision.motion_session import analyze_landmark_csv

def test_bundled_motion_fixture_computes_features():
    path=Path(__file__).resolve().parents[1]/'sample_data'/'observed_landmark_session.csv'
    d=analyze_landmark_csv(path.read_text())
    assert d['frames'] > 100
    assert 3.9 <= d['duration_s'] <= 4.1
    assert d['knee_rom_deg'] > 20
    assert d['estimated_cadence_spm'] > 60

def test_missing_required_motion_column_rejected():
    bad='time_s,left_heel_y,right_heel_y,left_knee_deg\n0,0,0,120\n1,1,1,130\n'
    try:
        analyze_landmark_csv(bad)
    except ValueError as e:
        assert 'right_knee_deg' in str(e)
    else:
        raise AssertionError('missing field must be rejected')

def test_sample_analysis_api():
    async def go():
        transport=httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,base_url='http://test') as c:
            r=await c.get('/api/motion/sample-analysis')
            assert r.status_code==200
            assert r.json()['frames']>100
    asyncio.run(go())

def test_asymmetric_fixture_changes_biomechanical_features():
    root=Path(__file__).resolve().parents[1]/'sample_data'
    a=analyze_landmark_csv((root/'observed_landmark_session.csv').read_text())
    b=analyze_landmark_csv((root/'observed_landmark_session_asymmetric.csv').read_text())
    assert abs(a['knee_rom_deg']-b['knee_rom_deg']) > 5
    assert b['bilateral_symmetry_index'] > a['bilateral_symmetry_index']
