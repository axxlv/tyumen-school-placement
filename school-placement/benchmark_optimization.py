"""Limited testing, never records user A/B choices or starts a full search."""
import importlib.util
import json
import sqlite3
import statistics
import tempfile
import time
from unittest.mock import patch
from pathlib import Path
import numpy as np
from engine import Engine, ROOT

def stats(times):
    return dict(count=len(times),mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),p90_seconds=float(np.percentile(times,90)),min_seconds=min(times),max_seconds=max(times))

if __name__=='__main__':
    start=time.perf_counter();e=Engine();startup=time.perf_counter()-start
    validated=json.loads((ROOT/'state'/'optimization_equivalence.json').read_text(encoding='utf-8'))
    assert validated['info']['fingerprint']==e.fingerprint,'Re-run equivalence checks for this engine revision first'
    # Uniform reproducible controls, supplemented by structural extremes.
    rng=np.random.default_rng(20261116)
    selected=list(map(int,rng.choice(e.ids,72,replace=False)))
    order=np.argsort([len(s) for s in e._sources])
    extremes=[e.ids[int(order[i])] for i in np.linspace(0,len(order)-1,12,dtype=int)]
    schools=list(map(int,e.input[e.input.capacity>0].index[:8]))
    maxpop=list(map(int,e.input.population.nlargest(4).index))
    controls=list(dict.fromkeys(selected+extremes+schools+maxpop))
    spec=importlib.util.spec_from_file_location('legacy_engine',ROOT/'engine_before.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    old=object.__new__(legacy.Engine)
    old.input=e.input;old.ids=list(e.ids);old.matrix=e._matrix;old.norm=e.norm;old.access=e.access
    old.reachable=e._matrix.to_numpy()<=e.access
    original_times=[];old_times=[];comparison=[]
    # Warm up executable/filesystem; no profiler, no cross-scenario caching.
    for block in controls[:2]:e.calculate_deficit([dict(id=block,capacity=600)])
    for block in controls[:12]:
        plan=[dict(id=block,capacity=600)]
        start=time.perf_counter();reference=e.calculate(plan,reference=True);original_times.append(time.perf_counter()-start)
        start=time.perf_counter();previous=old.calculate(plan);old_times.append(time.perf_counter()-start)
        fast=e.calculate_deficit(plan)
        assert fast==float(reference.demand_left.sum())==float(previous.demand_left.sum())
        comparison.append(dict(id=block,reference_seconds=original_times[-1],previous_seconds=old_times[-1]))
        print('TIMING comparison',block,flush=True)
    del old.reachable
    elapsed=[];items=[]
    import server
    class TestEngine:
        ids=controls
        deficit=e.deficit
        fingerprint=e.fingerprint
        def calculate_deficit(self,plan):
            result=e.calculate_deficit(plan)
            if (len(items)+1)%16==0:print('TIMING optimized',len(items)+1,'/',len(controls),flush=True)
            items.append(dict(id=plan[0]['id'],deficit=result))
            return result
    with tempfile.TemporaryDirectory(dir=ROOT/'state') as tmp:
        temporary_db=Path(tmp)/'benchmark.sqlite'
        source=sqlite3.connect(server.DB);dest=sqlite3.connect(temporary_db)
        try:source.backup(dest)
        finally:source.close();dest.close()
        with patch.object(server,'DB',temporary_db),patch.object(server,'ENGINE',TestEngine()):
            with server.db() as db:
                db.execute('DELETE FROM candidate_cache')
                db.execute('INSERT INTO jobs(id,kind,status,plan,capacity,total,fingerprint) VALUES(?,?,?,?,?,?,?)',('TEST_ONLY_BENCHMARK','search','queued','[]',600,len(controls),e.fingerprint))
            start=time.perf_counter();server.run_job('TEST_ONLY_BENCHMARK');worker_wall=time.perf_counter()-start
            with server.db() as db:job=db.execute("SELECT * FROM jobs WHERE id='TEST_ONLY_BENCHMARK'").fetchone()
            assert job['status']=='complete' and job['cached']==0 and job['processed']==len(controls)
            measured=server.ranking('TEST_ONLY_BENCHMARK')
            elapsed=[r['seconds'] for r in measured]
            # Rank timings end before the final write transaction. Add the
            # observed connection/commit overhead to estimate per-item wall time.
            overhead=(worker_wall-sum(elapsed))/len(elapsed)
            elapsed=[value+overhead for value in elapsed]
            by_id={r['id']:r for r in measured}
            for item in items:item['seconds']=by_id[item['id']]['seconds']+overhead
    result=dict(kind='LIMITED_TESTING_NOT_FULL_SEARCH_NOT_USER_CHOICES',fingerprint=e.fingerprint,capacity=600,startup_seconds=startup,reference=stats(original_times),previous_application=stats(old_times),optimized=stats(elapsed),worker_wall_seconds=worker_wall,estimated_minutes=len(e.ids)*statistics.mean(elapsed)/60,conservative_minutes=len(e.ids)*max(statistics.mean(elapsed)*1.25,float(np.percentile(elapsed,90)))/60,controls=items,comparison=comparison,notes='Real single sequential worker on a limited control list; every CBC solve uses all 12805 blocks. Isolated SQLite, no profiler, no cache hits. Wall time includes DB connections/reads/durable commits. Controls do not establish a global best. CPU load and antivirus can change wall time. Reserve is an engineering allowance, not a statistical confidence bound.')
    (ROOT/'state'/'optimization_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['controls','comparison']},ensure_ascii=False,indent=2),flush=True)
