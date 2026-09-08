from __future__ import annotations
import csv, io


def _downsample(values, target=80):
    if len(values) <= target:
        return values
    step = max(1, len(values)//target)
    return values[::step][:target]


def analyze_landmark_csv(text:str)->dict:
    rows=list(csv.DictReader(io.StringIO(text)))
    required={'time_s','left_heel_y','right_heel_y','left_knee_deg','right_knee_deg'}
    if not rows or not required.issubset(rows[0]): raise ValueError('CSV requires '+', '.join(sorted(required)))
    vals=lambda k:[float(r[k]) for r in rows]
    t=vals('time_s'); lk,rk=vals('left_knee_deg'),vals('right_knee_deg'); lh,rh=vals('left_heel_y'),vals('right_heel_y')
    rom=max(max(lk)-min(lk),max(rk)-min(rk)); asym=abs((sum(lk)/len(lk))-(sum(rk)/len(rk)))/max(1.0,(abs(sum(lk)/len(lk))+abs(sum(rk)/len(rk)))/2)
    crossings=sum(1 for i in range(1,len(rows)) if (lh[i]-rh[i])*(lh[i-1]-rh[i-1])<0)
    duration=max(.001,float(rows[-1]['time_s'])-float(rows[0]['time_s'])); cadence=60*crossings/duration
    return {
        'frames':len(rows),'duration_s':duration,'knee_rom_deg':rom,'bilateral_symmetry_index':asym,'estimated_cadence_spm':cadence,
        'source':'uploaded landmark CSV','status':'OBSERVED INPUT / COMPUTED FEATURES',
        'trace': {
            'time_s': _downsample(t),
            'left_heel_y': _downsample(lh),
            'right_heel_y': _downsample(rh),
            'left_knee_deg': _downsample(lk),
            'right_knee_deg': _downsample(rk),
        },
        'readiness': {
            'state_estimation': 'READY_FOR_KINEMATIC_FEATURES',
            'full_gait_assessment': 'INSUFFICIENT_SIGNALS' if not {'gait_speed_mps','margin_of_stability_m'}.issubset(rows[0]) else 'READY',
            'missing_for_full_assessment': ['gait_speed_mps','margin_of_stability_m'] if not {'gait_speed_mps','margin_of_stability_m'}.issubset(rows[0]) else [],
        }
    }
