import asyncio, httpx
from rehab_ai.web.app import app

async def _get(path):
    transport=httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,base_url='http://test') as c: return await c.get(path)

def test_health_and_algorithm_contract():
    h=asyncio.run(_get('/health')); assert h.status_code==200 and h.json()['service']=='rehab-ai-policy-studio'
    a=asyncio.run(_get('/api/algorithm')).json(); assert a['name']=='APACE' and 'CVaR tail risk' in a['objective_components']

def test_vue_frontend_shell_served():
    r=asyncio.run(_get('/')); assert r.status_code==200 and 'Recovery Policy Studio' in r.text and 'vue.global.prod.js' in r.text
