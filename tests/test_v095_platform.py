from rehab_ai.biomechanics.assessment_protocols import gait_assessment
from rehab_ai.vision.motion_session import analyze_landmark_csv
from rehab_ai.data.longitudinal_store import LongitudinalEvidenceStore,EvidenceRecord
from rehab_ai.apace.observatory import algorithm_observatory

def test_gait_protocol_identifies_limitation():
    x=gait_assessment(.75,100,30,.38,.012); assert x.primary_limitation in {'dynamic stability','bilateral symmetry','gait velocity','knee excursion'}; assert x.score<.8

def test_motion_csv_real_input_contract():
    txt='time_s,left_heel_y,right_heel_y,left_knee_deg,right_knee_deg\n0,0,1,140,145\n0.5,1,0,165,160\n1,0,1,142,146\n'
    x=analyze_landmark_csv(txt); assert x['frames']==3 and x['knee_rom_deg']>=20 and x['status'].startswith('OBSERVED')

def test_evidence_store_provenance_and_timeline(tmp_path):
    s=LongitudinalEvidenceStore(tmp_path/'e.db'); r=EvidenceRecord('P1','E1',0,'gait_speed','imu',.9,'m/s'); assert s.append_evidence([r,r])==1
    s.append_decision('P1','E1',1,'Balance',.4,.3,.04,.03,.12); assert s.timeline('P1')[0]['action']=='Balance'

def test_observatory_runs_all_phenotypes():
    x=algorithm_observatory(1,2,7); assert x['status']=='PASS' and x['metrics']['phenotypes']==5 and x['metrics']['envelope_violation_rate']==0
