import asyncio, httpx
from rehab_ai.web.app import app

async def _get(path):
    transport=httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,base_url='http://test') as c:
        return await c.get(path)

def test_safety_certificate_endpoint_is_verified_and_clinician_gated():
    r=asyncio.run(_get('/api/governance/apace-safety-certificate'))
    assert r.status_code==200
    assert r.headers['X-Request-ID'].startswith('rehab-')
    assert float(r.headers['X-Response-Time-Ms']) >= 0
    b=r.json()
    assert b['verification']['valid'] is True
    assert b['certificate']['decision_state']=='CLINICAL_REVIEW_REQUIRED'
    assert b['certificate']['therapy_change_execution_allowed'] is False
