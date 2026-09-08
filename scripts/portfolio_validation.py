from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rehab_ai.apace.validation import build_apace_validation

payload=build_apace_validation(patients=100,horizon=2,seed=2026)
root=ROOT
(root/'artifacts').mkdir(exist_ok=True)
(root/'artifacts'/'portfolio_validation.json').write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps({'release':payload['release'],'status':payload['status'],'checks':payload['checks'],'apace':payload['policies']['apace']},indent=2))
if payload['status']!='PASS': raise SystemExit('portfolio validation requires review')
print('REHAB_AI_PORTFOLIO_VALIDATION=PASS')
