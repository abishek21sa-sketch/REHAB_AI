from pathlib import Path
import csv,statistics,math
ROOT=Path(__file__).resolve().parents[1]
def _session(name):
 rows=list(csv.DictReader((ROOT/'sample_data'/name).open())); asym=[abs(float(r['left_knee_deg'])-float(r['right_knee_deg'])) for r in rows]; heel=[abs(float(r['left_heel_y'])-float(r['right_heel_y'])) for r in rows]; return rows,asym,heel
def domain_diagnostics():
 rows,a,h=_session('observed_landmark_session.csv'); rows2,a2,h2=_session('observed_landmark_session_asymmetric.csv'); sa=sorted(a); sa2=sorted(a2)
 return {'analysis':'bilateral movement asymmetry and progression-risk evidence','metrics':{'frames':len(rows),'baseline_mean_knee_asym_deg':round(statistics.fmean(a),3),'baseline_p95_knee_asym_deg':round(sa[int(.95*(len(sa)-1))],3),'asymmetric_mean_knee_asym_deg':round(statistics.fmean(a2),3),'asymmetry_delta_deg':round(statistics.fmean(a2)-statistics.fmean(a),3),'baseline_mean_heel_y_gap':round(statistics.fmean(h),5)},'decision_signal':'HMM state estimation and MOTION-GUARD should treat sustained bilateral asymmetry as evidence against automatic progression before APACE/twin review.','evidence_boundary':'Observed landmark demo sessions are engineering validation fixtures, not clinical outcome evidence.'}
