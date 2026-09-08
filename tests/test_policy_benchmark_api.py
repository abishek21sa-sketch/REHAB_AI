import asyncio
import httpx
from rehab_ai.web.app import app

async def _get(path: str):
    transport=httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,base_url='http://testserver') as client:
        return await client.get(path)

def test_benchmark_endpoint_returns_common_objective_comparison():
    r=asyncio.run(_get('/api/apace/benchmark?patients=4&horizon=2&seed=5'))
    assert r.status_code==200
    data=r.json()
    assert data['validation_scope']=='SYNTHETIC POLICY BENCHMARK'
    names={x['policy'] for x in data['results']}
    assert {'apace','greedy','risk_only','standard_mpc','oracle'} <= names
