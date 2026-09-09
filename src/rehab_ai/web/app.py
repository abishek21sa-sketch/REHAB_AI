from __future__ import annotations

from pathlib import Path
import json
import os
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from rehab_ai import __version__
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.apace_studio import run_apace_studio
from rehab_ai.control.stochastic_mpc import default_actions
from rehab_ai.control.signature_algorithm import choose_safe_load as signature_choose, ablation as signature_ablation, sensitivity as signature_sensitivity
from rehab_ai.apace.benchmark import run_policy_benchmark
from rehab_ai.ml.safe_envelope import PatientSafeEnvelopeLearner
from rehab_ai.ml.temporal_recovery import TemporalRecoveryModel
from rehab_ai.twin.adaptive_episode import run_adaptive_episode
from rehab_ai.apace.population_stress import run_population_stress
from rehab_ai.apace.observatory import algorithm_observatory
from rehab_ai.biomechanics.assessment_protocols import gait_assessment
from rehab_ai.vision.motion_session import analyze_landmark_csv
from rehab_ai.clinical_workflow import build_clinical_episode, episode_markdown_report
from rehab_ai.biomechanics.assessment_protocols import assessment_suite
from rehab_ai.governance import build_apace_safety_certificate, verify_apace_safety_certificate
from rehab_ai.ai.copilot import build_response as build_copilot_response, status as copilot_status

FRONTEND = Path(__file__).resolve().parent / "frontend"
SAMPLE_DATA = Path(__file__).resolve().parent / "sample_data"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "rehab-ai-policy-studio", "version": __version__})

def copilot_status_endpoint(_: Request) -> JSONResponse:
    return JSONResponse(copilot_status())

async def copilot_chat(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        message = str(payload.get("message", "")).strip()
        if not message or len(message) > 2000:
            return JSONResponse({"error": "message must be between 1 and 2000 characters"}, status_code=422)
        return JSONResponse(build_copilot_response(message, {"evidence": "synthetic motion, biomechanics, patient-safe-envelope, APACE, and longitudinal evidence", "mode": "clinician-review-only"}))
    except Exception as exc:
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)}, status_code=422)


def algorithm(_: Request) -> JSONResponse:
    return JSONResponse({
        "name": "APACE",
        "expanded_name": "Adaptive Patient-Aware Control and Exploration",
        "role": "risk-sensitive adaptive dual-control search for rehabilitation policy selection",
        "objective_components": ["expected functional gain", "information gain", "CVaR tail risk", "patient burden"],
        "safety": ["patient-specific load envelope", "fatigue limit", "pain limit", "minimum margin of stability"],
        "status": "research algorithm; synthetic validation; external clinical validation pending",
    })


def signature_contract(_: Request) -> JSONResponse:
    """Expose the executable SAFE-MPC reference policy and safety counterfactuals."""
    args = (0.5, 0.8, [0.0, 0.2, 0.4])
    return JSONResponse({
        "status": "HUMAN_GATED_REFERENCE",
        "signature_algorithm": "SAFE-MPC",
        "decision": signature_choose(*args, lower_state=.2, upper_state=.9, max_step=.4),
        "baseline": signature_ablation(*args, max_step=.4),
        "sensitivity": signature_sensitivity(*args, .05, lower_state=.2, upper_state=.9, max_step=.4),
        "objective": "minimize target error plus burden while remaining in the state envelope",
        "counterfactual": "safety-envelope ablation",
        "evidence_artifact": "artifacts/fortune50_capability_benchmark.json",
        "autonomous_execution": False,
    })


def actions(_: Request) -> JSONResponse:
    return JSONResponse({"actions": [
        {"name": a.name, "modality": a.modality, "intensity": a.intensity, "duration_min": a.duration_min, "instability_demand": a.instability_demand}
        for a in default_actions()
    ]})


def demo(request: Request) -> JSONResponse:
    try:
        impairment = float(request.query_params.get("impairment", "0.45"))
        horizon = int(request.query_params.get("horizon", "5"))
        seed = int(request.query_params.get("seed", "42"))
        impairment = min(0.90, max(0.10, impairment))
        horizon = min(6, max(2, horizon))
        session = generate_synthetic_session(patient_id=f"DEMO-{int(impairment*100):02d}", seed=seed, impairment=impairment)
        result = run_apace_studio(session, horizon=horizon, seed=seed)
        return JSONResponse(result.to_dict())
    except Exception as exc:
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)}, status_code=422)



def benchmark(request: Request) -> JSONResponse:
    try:
        patients = min(24, max(4, int(request.query_params.get("patients", "8"))))
        horizon = min(3, max(2, int(request.query_params.get("horizon", "2"))))
        seed = int(request.query_params.get("seed", "42"))
        return JSONResponse(run_policy_benchmark(patients=patients, horizon=horizon, seed=seed).to_dict())
    except Exception as exc:
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)}, status_code=422)


_DEPTH_CACHE = None

def computational_depth(_: Request) -> JSONResponse:
    global _DEPTH_CACHE
    if _DEPTH_CACHE is None:
        envelope = PatientSafeEnvelopeLearner(); em = envelope.fit_validate(n=900, seed=42)
        temporal = TemporalRecoveryModel(random_state=42); tm = temporal.fit_validate(seed=42)
        _DEPTH_CACHE = {
            "safe_envelope": em.__dict__,
            "temporal_recovery": tm.__dict__,
            "evidence_language": {
                "safe_envelope": "VALIDATED ON SYNTHETIC DATA",
                "temporal_recovery": "VALIDATED ON SYNTHETIC LONGITUDINAL DATA",
                "clinical_accuracy": "EXTERNAL VALIDATION PENDING",
            },
        }
    return JSONResponse(_DEPTH_CACHE)

def adaptive_episode(request: Request) -> JSONResponse:
    try:
        phenotype=request.query_params.get("phenotype","balanced")
        weeks=min(8,max(2,int(request.query_params.get("weeks","5"))))
        seed=int(request.query_params.get("seed","42"))
        return JSONResponse(run_adaptive_episode(phenotype=phenotype,weeks=weeks,seed=seed).to_dict())
    except Exception as exc:
        return JSONResponse({"error":type(exc).__name__,"detail":str(exc)},status_code=422)

def population_stress(request: Request) -> JSONResponse:
    try:
        n=min(8,max(1,int(request.query_params.get("patients_per_phenotype","2"))))
        weeks=min(6,max(2,int(request.query_params.get("weeks","3"))))
        seed=int(request.query_params.get("seed","42"))
        return JSONResponse(run_population_stress(n,weeks,seed).to_dict())
    except Exception as exc:
        return JSONResponse({"error":type(exc).__name__,"detail":str(exc)},status_code=422)


def assessment_demo(request: Request) -> JSONResponse:
    impairment=float(request.query_params.get("impairment","0.45")); session=generate_synthetic_session(patient_id="ASSESS",seed=42,impairment=impairment); r=run_apace_studio(session,horizon=3,seed=42).to_dict(); e=r["evidence"]; b=r["biomechanics"]
    return JSONResponse(gait_assessment(e["gait_speed_mps"],e["cadence_spm"],e["knee_rom_deg"],b["bilateral_symmetry_index"],b["margin_of_stability_m"]).to_dict())

def observatory(request: Request) -> JSONResponse:
    return JSONResponse(algorithm_observatory(min(4,max(1,int(request.query_params.get("patients_per_phenotype","2")))),min(5,max(2,int(request.query_params.get("weeks","3")))),42))

async def motion_upload(request: Request) -> JSONResponse:
    try:
        body=(await request.body()).decode("utf-8"); return JSONResponse(analyze_landmark_csv(body))
    except Exception as exc: return JSONResponse({"error":type(exc).__name__,"detail":str(exc)},status_code=422)



def motion_sample(_: Request):
    path = SAMPLE_DATA / "observed_landmark_session.csv"
    return FileResponse(path, media_type="text/csv", filename=path.name)



def motion_sample_asymmetric(_: Request):
    path = SAMPLE_DATA / "observed_landmark_session_asymmetric.csv"
    return FileResponse(path, media_type="text/csv", filename=path.name)

def motion_sample_analysis(_: Request) -> JSONResponse:
    path = SAMPLE_DATA / "observed_landmark_session.csv"
    return JSONResponse(analyze_landmark_csv(path.read_text(encoding="utf-8")))


def clinical_episode(request: Request) -> JSONResponse:
    try:
        impairment=min(.90,max(.10,float(request.query_params.get("impairment","0.45"))))
        phenotype=request.query_params.get("phenotype","balanced")
        weeks=min(8,max(3,int(request.query_params.get("weeks","5"))))
        seed=int(request.query_params.get("seed","42"))
        return JSONResponse(build_clinical_episode(impairment=impairment,phenotype=phenotype,weeks=weeks,seed=seed).to_dict())
    except Exception as exc:
        return JSONResponse({"error":type(exc).__name__,"detail":str(exc)},status_code=422)

def clinical_report(request: Request) -> Response:
    impairment=min(.90,max(.10,float(request.query_params.get("impairment","0.45"))))
    phenotype=request.query_params.get("phenotype","balanced")
    weeks=min(8,max(3,int(request.query_params.get("weeks","5"))))
    summary=build_clinical_episode(impairment=impairment,phenotype=phenotype,weeks=weeks,seed=42)
    text=episode_markdown_report(summary)
    return Response(text,media_type="text/markdown",headers={"Content-Disposition":f'attachment; filename="{summary.patient_id}_rehab_report.md"'})

def assessment_suite_demo(request: Request) -> JSONResponse:
    impairment=float(request.query_params.get("impairment","0.45"))
    session=generate_synthetic_session(patient_id="ASSESS-SUITE",seed=42,impairment=impairment)
    r=run_apace_studio(session,horizon=3,seed=42).to_dict(); e=r["evidence"]; b=r["biomechanics"]; l=r["latent_state"]
    suite=assessment_suite(gait_speed_mps=e["gait_speed_mps"],cadence_spm=e["cadence_spm"],knee_rom_deg=e["knee_rom_deg"],symmetry_index=b["bilateral_symmetry_index"],margin_of_stability_m=b["margin_of_stability_m"],fatigue=l["fatigue_burden"],pain=l["pain_burden"],motor_capacity=l["motor_capacity"])
    return JSONResponse({k:v.to_dict() for k,v in suite.items()})

def platform_manifest(_: Request) -> JSONResponse:
    return JSONResponse({"version":__version__,"workspaces":["Digital Patient Bay","Motion Capture Studio","Assessment Laboratory","Longitudinal Record","APACE Decision Chamber","Population Lab","Algorithm Observatory","Twin Theater"],"data_plane":{"operational":"SQLite","analytics":"DuckDB + Parquet when optional dependency installed","provenance":"evidence fingerprints + model version"},"clinical_status":"research decision support; external clinical validation pending"})


def safety_certificate(_: Request) -> JSONResponse:
    path=PROJECT_ROOT/'artifacts'/'portfolio_validation.json'
    if not path.exists():
        return JSONResponse({'error':'validation evidence unavailable','required_action':'run scripts/portfolio_validation.py'},status_code=503)
    validation=json.loads(path.read_text(encoding='utf-8'))
    cert=build_apace_safety_certificate(validation)
    return JSONResponse({'certificate':cert,'verification':verify_apace_safety_certificate(cert)})

def index(_: Request) -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


routes = [
    Route("/", index),
    Route("/health", health),
    Route("/api/copilot/status", copilot_status_endpoint),
    Route("/api/copilot/chat", copilot_chat, methods=["POST"]),
    Route("/api/algorithm", algorithm),
    Route("/api/governance/signature", signature_contract),
    Route("/api/actions", actions),
    Route("/api/apace/demo", demo),
    Route("/api/apace/benchmark", benchmark),
    Route("/api/research/depth", computational_depth),
    Route("/api/adaptive/episode", adaptive_episode),
    Route("/api/adaptive/population-stress", population_stress),
    Route("/api/assessment/gait", assessment_demo),
    Route("/api/assessment/suite", assessment_suite_demo),
    Route("/api/clinical/episode", clinical_episode),
    Route("/api/clinical/report", clinical_report),
    Route("/api/research/observatory", observatory),
    Route("/api/motion/landmarks", motion_upload, methods=["POST"]),
    Route("/api/motion/sample", motion_sample),
    Route("/api/motion/sample-analysis", motion_sample_analysis),
    Route("/api/motion/sample-asymmetric", motion_sample_asymmetric),
    Route("/api/platform/manifest", platform_manifest),
    Route("/api/governance/apace-safety-certificate", safety_certificate),
    Mount("/assets", StaticFiles(directory=FRONTEND), name="assets"),
]

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id=request.headers.get("X-Request-ID") or f"rehab-{uuid.uuid4().hex[:16]}"
        started=time.perf_counter()
        response=await call_next(request)
        response.headers["X-Request-ID"]=request_id
        response.headers["X-Response-Time-Ms"]=f"{(time.perf_counter()-started)*1000:.3f}"
        response.headers["X-Therapy-Execution"]="CLINICIAN_REVIEW_ONLY"
        return response

app = Starlette(debug=False, routes=routes)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("REHAB_CORS_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
