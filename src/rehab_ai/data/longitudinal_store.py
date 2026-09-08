from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, sqlite3, hashlib
from typing import Iterable

@dataclass(frozen=True)
class EvidenceRecord:
    patient_id: str; episode_id: str; week: int; evidence_type: str; source: str; value: float; unit: str; model_version: str="0.99.0"
    def fingerprint(self)->str:
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]

class LongitudinalEvidenceStore:
    """Portable operational store; DuckDB/Parquet export is enabled when duckdb is installed."""
    def __init__(self,path: str|Path=":memory:"):
        self.path=str(path); self.db=sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS evidence(patient_id TEXT,episode_id TEXT,week INTEGER,evidence_type TEXT,source TEXT,value REAL,unit TEXT,model_version TEXT,fingerprint TEXT PRIMARY KEY)")
        self.db.execute("CREATE TABLE IF NOT EXISTS decisions(patient_id TEXT,episode_id TEXT,week INTEGER,action TEXT,safe_limit REAL,applied_load REAL,predicted_gain REAL,observed_gain REAL,parameter_std REAL)")
    def append_evidence(self, records: Iterable[EvidenceRecord])->int:
        n=0
        for r in records:
            cur=self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?,?,?)",(*asdict(r).values(),r.fingerprint())); n+=cur.rowcount
        self.db.commit(); return n
    def append_decision(self, patient_id, episode_id, week, action, safe_limit, applied_load, predicted_gain, observed_gain, parameter_std):
        self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?)",(patient_id,episode_id,week,action,safe_limit,applied_load,predicted_gain,observed_gain,parameter_std)); self.db.commit()
    def timeline(self,patient_id:str)->list[dict]:
        cols=['episode_id','week','action','safe_limit','applied_load','predicted_gain','observed_gain','parameter_std']
        rows=self.db.execute("SELECT episode_id,week,action,safe_limit,applied_load,predicted_gain,observed_gain,parameter_std FROM decisions WHERE patient_id=? ORDER BY week",(patient_id,)).fetchall()
        return [dict(zip(cols,r)) for r in rows]
    def export_analytics(self,out_dir:str|Path)->dict:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
        try:
            import duckdb
        except ImportError:
            manifest=out/'PARQUET_EXPORT_REQUIRES_DUCKDB.json'; manifest.write_text(json.dumps({'status':'dependency_required','install':'pip install duckdb','operational_store':'sqlite','schema':['evidence','decisions']},indent=2)); return {'status':'dependency_required','manifest':str(manifest)}
        con=duckdb.connect();
        dbpath=self.path.replace("'","''")
        con.execute(f"ATTACH '{dbpath}' AS src (TYPE SQLITE)")
        for table in ('evidence','decisions'):
            target=(out/f'{table}.parquet').as_posix(); con.execute(f"COPY (SELECT * FROM src.{table}) TO '{target}' (FORMAT PARQUET)")
        return {'status':'exported','files':['evidence.parquet','decisions.parquet']}
