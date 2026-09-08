from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class AssessmentFinding:
    protocol:str; score:float; severity:str; primary_limitation:str; evidence:list[str]; rationale:str
    def to_dict(self): return asdict(self)


def _severity(score: float) -> str:
    return 'high' if score < .45 else 'moderate' if score < .70 else 'low'


def gait_assessment(gait_speed_mps:float,cadence_spm:float,knee_rom_deg:float,symmetry_index:float,margin_of_stability_m:float)->AssessmentFinding:
    deficits={
      'dynamic stability': max(0.0,(0.035-margin_of_stability_m)/0.035),
      'bilateral symmetry': min(1.0,max(0.0,symmetry_index)),
      'gait velocity': max(0.0,(1.1-gait_speed_mps)/1.1),
      'knee excursion': max(0.0,(45-knee_rom_deg)/45),
    }
    primary=max(deficits,key=deficits.get); score=max(0.0,1-sum(deficits.values())/len(deficits))
    ev=[f'gait speed {gait_speed_mps:.2f} m/s',f'cadence {cadence_spm:.0f} spm',f'knee ROM {knee_rom_deg:.1f}°',f'symmetry index {symmetry_index:.2f}',f'MoS {margin_of_stability_m:.3f} m']
    return AssessmentFinding('gait',score,_severity(score),primary,ev,f'{primary} contributes the largest normalized deficit in the gait evidence vector.')


def balance_assessment(margin_of_stability_m: float, symmetry_index: float, fatigue: float) -> AssessmentFinding:
    deficits = {
        'dynamic stability': max(0.0, (0.04-margin_of_stability_m)/0.04),
        'asymmetric support': min(1.0, max(0.0, symmetry_index*1.5)),
        'fatigue-mediated balance reserve': min(1.0, max(0.0, fatigue)),
    }
    primary=max(deficits,key=deficits.get); score=max(0.0,1-sum(deficits.values())/len(deficits))
    ev=[f'MoS {margin_of_stability_m:.3f} m',f'symmetry index {symmetry_index:.2f}',f'fatigue burden {fatigue:.2f}']
    return AssessmentFinding('balance',score,_severity(score),primary,ev,f'{primary} is the dominant normalized balance deficit.')


def mobility_assessment(knee_rom_deg: float, motor_capacity: float, pain: float) -> AssessmentFinding:
    deficits = {
        'knee excursion': max(0.0,(50-knee_rom_deg)/50),
        'functional motor capacity': max(0.0,1-motor_capacity),
        'pain-limited motion': min(1.0,max(0.0,pain)),
    }
    primary=max(deficits,key=deficits.get); score=max(0.0,1-sum(deficits.values())/len(deficits))
    ev=[f'knee ROM {knee_rom_deg:.1f}°',f'motor capacity {motor_capacity:.2f}',f'pain burden {pain:.2f}']
    return AssessmentFinding('mobility',score,_severity(score),primary,ev,f'{primary} contributes the largest normalized mobility deficit.')


def endurance_assessment(cadence_spm: float, fatigue: float, motor_capacity: float) -> AssessmentFinding:
    deficits = {
        'fatigue burden': min(1.0,max(0.0,fatigue)),
        'cadence reserve': max(0.0,(110-cadence_spm)/110),
        'functional reserve': max(0.0,1-motor_capacity),
    }
    primary=max(deficits,key=deficits.get); score=max(0.0,1-sum(deficits.values())/len(deficits))
    ev=[f'cadence {cadence_spm:.0f} spm',f'fatigue burden {fatigue:.2f}',f'motor capacity {motor_capacity:.2f}']
    return AssessmentFinding('endurance',score,_severity(score),primary,ev,f'{primary} is the largest normalized endurance deficit.')


def assessment_suite(*, gait_speed_mps:float,cadence_spm:float,knee_rom_deg:float,symmetry_index:float,margin_of_stability_m:float,fatigue:float,pain:float,motor_capacity:float)->dict[str,AssessmentFinding]:
    return {
        'gait': gait_assessment(gait_speed_mps,cadence_spm,knee_rom_deg,symmetry_index,margin_of_stability_m),
        'balance': balance_assessment(margin_of_stability_m,symmetry_index,fatigue),
        'mobility': mobility_assessment(knee_rom_deg,motor_capacity,pain),
        'endurance': endurance_assessment(cadence_spm,fatigue,motor_capacity),
    }
