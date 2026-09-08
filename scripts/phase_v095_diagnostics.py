from rehab_ai.biomechanics.assessment_protocols import gait_assessment
from rehab_ai.vision.motion_session import analyze_landmark_csv
from rehab_ai.apace.observatory import algorithm_observatory
import json
csv='time_s,left_heel_y,right_heel_y,left_knee_deg,right_knee_deg\n0,0,1,140,145\n0.5,1,0,165,160\n1,0,1,142,146\n'
a=gait_assessment(.82,108,34,.31,.018); m=analyze_landmark_csv(csv); o=algorithm_observatory(1,2,42)
checks={'assessment_rationale':bool(a.primary_limitation),'observed_motion_ingestion':m['frames']==3,'observatory_all_phenotypes':o['metrics']['phenotypes']==5,'observatory_safe':o['metrics']['envelope_violation_rate']==0}
print(json.dumps({'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'assessment':a.to_dict(),'motion':m,'observatory':o['metrics']},indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
