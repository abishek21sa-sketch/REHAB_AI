from __future__ import annotations
from tenx.engine import run_decision
from campaign.engine import run_campaign
from empirical.backbone import run_empirical_reference

def lifecycle_report():
    d=run_decision(37); v=d['model_validation']; metric=float(v.get('value', v.get('accuracy',0)) or 0)
    state='STATE_MODEL_REVIEW' if ('lower' in str(v.get('direction','')).lower() and metric>.30) else 'MONITORING_ACTIVE'
    public_data=d.get('public_data_backbone',{})
    return {'public_data_state':public_data.get('dataset_state'),'public_evidence_gate':d.get('public_evidence_gate'),'model_family':d['ml_family'],'target':d['prediction_target'],'validation':v,'state_model_status':state,'retrain_trigger':'re-estimate HMM emissions/transitions if state likelihood degrades, sensor modality changes, or longitudinal phenotype drift is detected','monitoring':['sequence log-likelihood','state occupancy drift','transition stability','sensor missingness','prediction-to-progression decision delta'],'registry_state':'HMM_ACTIVE' if state=='MONITORING_ACTIVE' else 'HMM_SHADOW','source_mode':run_empirical_reference().get('data_mode')}

def run_agent():
    d=run_decision(37); life=lifecycle_report(); c=run_campaign(); steps=['fuse wearable/motion observations','infer latent recovery state with HMM','estimate biomechanics and asymmetry']
    state='CLINICIAN_REVIEW'
    public_gate=d.get('public_evidence_gate')
    if public_gate=='REFERENCE_MODE_HOLD_FOR_REAL_DATA_CLAIM':
        steps.append('flag external public-data acquisition gap; prohibit real-data performance claim')
        state='REFERENCE_MODE_HOLD'
    if life['state_model_status']!='MONITORING_ACTIVE': steps += ['freeze progression recommendation','request repeat assessment/sensor review']; state='STATE_MODEL_HOLD'
    else: steps += ['generate APACE candidate progression','filter through MOTION-GUARD safety envelope','forecast digital-twin response','assemble clinician review packet']
    return {'agent':'Rehabilitation Planning Agent','objective':'propose progression only when latent-state evidence and modeled safety envelope agree','prediction':d.get('prediction') or d.get('predictions'),'decision':d['decision'],'decision_state':state,'chosen_tool_sequence':steps,'why_this_sequence':'latent-state confidence determines whether the system may explore progression or must repeat assessment','challenge':d['counterfactual'],'ml_lifecycle':life,'campaign_state':c.get('state'),'operator_actions':['inspect state timeline','review left/right asymmetry','compare APACE alternatives','approve/HOLD progression'],'human_authority':d['human_authority'],'autonomous_execution':False,'clinical_claim':'NONE'}
