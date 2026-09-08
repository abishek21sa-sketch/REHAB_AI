from __future__ import annotations
import math
import numpy as np
class GaussianHMM:
    def fit_supervised(self,seqs,states,k=3):
        self.k=k; D=seqs[0].shape[1]; self.means=[]; self.vars=[]
        allx=np.concatenate(seqs); alls=np.concatenate(states)
        for s in range(k):
            x=allx[alls==s]; self.means.append(x.mean(0)); self.vars.append(x.var(0)+.03)
        self.trans=np.ones((k,k))*.2
        for ss in states:
            for a,b in zip(ss[:-1],ss[1:]): self.trans[a,b]+=1
        self.trans/=self.trans.sum(1,keepdims=True); self.pi=np.array([.55,.3,.15]); return self
    def _emit(self,x,s):
        v=self.vars[s]; d=x-self.means[s]; return math.exp(float(-.5*np.sum(d*d/v)))/math.sqrt(float(np.prod(2*math.pi*v)))
    def filter(self,seq):
        p=self.pi.copy()
        for x in seq:
            e=np.array([self._emit(x,s) for s in range(self.k)]); p=(p@self.trans)*e; p/=p.sum()+1e-15
        return p
class MotionGuard:
    def choose(self,posterior,actions,pain,fatigue,stability):
        state_risk=float(posterior[2]+.45*posterior[1]); rows=[]
        for a in actions:
            safety=1-(.55*a['intensity']+.35*a['instability']+.25*pain+.25*fatigue+.35*state_risk-.25*stability); gain=a['gain']*(posterior[0]+.65*posterior[1]+.25*posterior[2]); score=gain+1.4*max(safety,0)-2.5*max(-safety,0); rows.append({**a,'safety_margin':safety,'guard_score':score})
        safe=[x for x in rows if x['safety_margin']>=.12]; return max(safe,key=lambda x:x['guard_score']) if safe else {'name':'HOLD_PROGRESSION','safety_margin':max(x['safety_margin'] for x in rows),'guard_score':-999}
def _run_decision_core(seed=9):
    r=np.random.default_rng(seed); seqs=[]; states=[]
    means=[np.array([1.05,.18,.18]),np.array([.78,.42,.38]),np.array([.48,.72,.68])]
    for _ in range(45):
        ss=[]; xs=[]; s=int(r.choice(3,p=[.5,.35,.15]))
        for t in range(12):
            if r.random()<.25:
                s=min(2,max(0,s+int(r.choice([-1,0,1],p=[.25,.5,.25]))))
            ss.append(s)
            xs.append(r.normal(means[s],[.08,.08,.08]))
        seqs.append(np.array(xs)); states.append(np.array(ss))
    hmm=GaussianHMM().fit_supervised(seqs[:36],states[:36]); acc=float(np.mean([int(np.argmax(hmm.filter(x))==int(ss[-1])) for x,ss in zip(seqs[36:],states[36:]) ])); obs=np.array([[.72,.45,.42],[.70,.48,.46],[.68,.51,.49],[.69,.50,.52]]); post=hmm.filter(obs)
    actions=[{'name':'progress_resistance','intensity':.82,'instability':.55,'gain':.95},{'name':'maintain_mixed','intensity':.56,'instability':.32,'gain':.72},{'name':'deload_balance','intensity':.35,'instability':.18,'gain':.48}]
    dec=MotionGuard().choose(post,actions,pain=.38,fatigue=.46,stability=.52); naive=max(actions,key=lambda x:x['gain'])['name']
    return {'project':'REHAB AI','ml_family':'Gaussian hidden-state sequence learning (HMM)','prediction_target':'latent rehabilitation state distribution / progression risk','model_validation':{'metric':'final latent-state holdout accuracy','value':acc,'direction':'higher_is_better','split':'held synthetic patient sequences'},'prediction':{'ready':float(post[0]),'caution':float(post[1]),'high_risk':float(post[2])},'original_algorithm':'MOTION-GUARD-v1','decision':dec,'counterfactual':{'naive_policy':'maximum expected functional gain','choice':naive,'disagrees':dec['name']!=naive},'uncertainty':'Posterior state probability is propagated into progression safety margin.','or_escalation':'MOTION-GUARD-approved candidate enters APACE risk-sensitive dual control and digital-twin forecast.','tool_trace':['estimate latent state with HMM','score progression risk','screen actions with MOTION-GUARD','challenge max-gain action','escalate to APACE','forecast in patient twin'],'limitations':['bundled evidence is synthetic/research only','not clinically validated'],'abstention_conditions':['sensor quality inadequate','posterior high-risk state elevated','pain/fatigue envelope breach'],'user_aid':['inspect latent-state evidence','compare guarded vs max-gain action','review twin forecast','approve/hold therapy progression'],'human_authority':'CLINICIAN','autonomous_execution':False,'clinical_validation':'EXTERNAL_VALIDATION_PENDING'}


def run_decision(seed=None):
    from empirical.backbone import run_empirical_reference
    import inspect
    sig=inspect.signature(_run_decision_core)
    if seed is None:
        out=_run_decision_core()
    else:
        out=_run_decision_core(seed)
    emp=run_empirical_reference()
    out["empirical_backbone"]=emp
    from empirical.public_data_backbone import integrate_decision
    out=integrate_decision(out)
    out.setdefault("tool_trace",[]).insert(0,"resolve empirical data provenance and source mode")
    out.setdefault("user_aid",[]).append("open empirical case study and entity/history drilldowns before approval")
    return out
