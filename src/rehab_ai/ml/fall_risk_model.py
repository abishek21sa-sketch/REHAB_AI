from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.sensing.wearable import extract_wearable_features
from rehab_ai.vision.movement import extract_movement_features
from rehab_ai.biomechanics.gait import reconstruct_patient_state
from rehab_ai.domain import PatientState, Prediction

FEATURES = ["gait_speed_mps","gait_variability","sway_index","knee_asymmetry_deg","trunk_instability","fatigue","functional_capacity"]
MODEL_VERSION = "logistic-fall-risk-synthetic-v2-json"
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "artifacts/models/fall_risk_logistic_v2.json"
METRICS_PATH = ROOT / "artifacts/models/fall_risk_metrics.json"

def vectorize(s: PatientState) -> np.ndarray:
    return np.array([getattr(s, k) for k in FEATURES], dtype=float)

def _latent_probability(x: np.ndarray) -> float:
    z = -3.0 + 1.5*max(0,1.15-x[0]) + 1.4*x[1] + 2.4*x[2] + .045*x[3] + 1.2*x[4] + .9*x[5] + 1.3*(1-x[6])
    return 1/(1+math.exp(-z))

def build_synthetic_cohort(n: int=900, seed: int=2026):
    rng=np.random.default_rng(seed); X=[]; y=[]
    for i in range(n):
        impairment=float(rng.uniform(.05,.95))
        session=generate_synthetic_session(patient_id=f"ML-{i}", impairment=impairment, seed=seed+i)
        state=reconstruct_patient_state(session, extract_wearable_features(session.wearable), extract_movement_features(session.pose))
        x=vectorize(state); p=_latent_probability(x)
        X.append(x); y.append(int(rng.random()<p))
    return np.vstack(X), np.asarray(y,dtype=int)

def _save_model(model: LogisticRegression, path: Path) -> None:
    payload={"model_version":MODEL_VERSION,"features":FEATURES,"coef":model.coef_[0].tolist(),"intercept":float(model.intercept_[0]),"classes":model.classes_.tolist()}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2), encoding="utf-8")

def train_and_validate(model_path: Path=MODEL_PATH, metrics_path: Path=METRICS_PATH):
    X,y=build_synthetic_cohort(n=1800); split=int(.70*len(y)); val=int(.85*len(y))
    Xtr,Xv,Xt=X[:split],X[split:val],X[val:]; ytr,yv,yt=y[:split],y[split:val],y[val:]
    model=LogisticRegression(max_iter=3000, random_state=2026).fit(Xtr,ytr)
    pv=model.predict_proba(Xv)[:,1]; pt=model.predict_proba(Xt)[:,1]
    prevalence=float(ytr.mean()); naive=np.full(len(yt),prevalence)
    metrics={"validation_auc":roc_auc_score(yv,pv),"test_auc":roc_auc_score(yt,pt),"test_brier":brier_score_loss(yt,pt),"baseline":"training-prevalence constant probability","baseline_test_brier":brier_score_loss(yt,naive),"n_train":len(ytr),"n_validation":len(yv),"n_test":len(yt),"data_status":"SYNTHETIC VALIDATION","model_version":MODEL_VERSION,"features":FEATURES,"persistence_format":"portable JSON coefficients; no pickle/joblib runtime dependency"}
    _save_model(model, model_path); metrics_path.parent.mkdir(parents=True,exist_ok=True); metrics_path.write_text(json.dumps(metrics,indent=2), encoding="utf-8")
    return metrics

def _predict_json(x: np.ndarray, model_path: Path) -> float:
    payload=json.loads(model_path.read_text(encoding="utf-8")); coef=np.asarray(payload["coef"],float); intercept=float(payload["intercept"])
    z=float(intercept + np.dot(coef,x)); return 1.0/(1.0+math.exp(-z))

def predict_ml_fall_risk(state: PatientState, model_path: Path=MODEL_PATH) -> Prediction:
    p=_predict_json(vectorize(state), model_path)
    return Prediction(name="six_week_fall_risk_probability_synthetic_model",value=p,lower=max(0,p-.12),upper=min(1,p+.12),confidence=.72,model_version=MODEL_VERSION)
