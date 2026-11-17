"""TEST ONLY: controls are not user A/B choices or an exhaustive search."""
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from engine import Engine, TOLERANCE, best_rows, ROOT

def mps_digest(prob,path):
    prob.writeMPS(str(path),rename=1)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(e):
    population=e.input.population;capacity=e.input.capacity
    maxpop=int(population.idxmax());school=int(capacity[capacity>0].index[0])
    zero=int(population[population==0].index[0]);last=e.ids[-1]
    cases=[('zero',[]),('zero-population',[dict(id=zero,capacity=600)]),('max-population',[dict(id=maxpop,capacity=600)]),('existing-school',[dict(id=school,capacity=600)]),('boundary-last',[dict(id=last,capacity=600)]),('one-seat',[dict(id=2,capacity=1)]),('300-seats',[dict(id=school,capacity=300)]),('1200-seats',[dict(id=maxpop,capacity=1200)]),('large-self-supply',[dict(id=maxpop,capacity=2000)]),('duplicate-additions',[dict(id=2,capacity=300),dict(id=2,capacity=300)]),('two-different',[dict(id=school,capacity=300),dict(id=maxpop,capacity=300)])]
    rng=np.random.default_rng(20261116)
    for block in rng.choice(e.ids,8,replace=False):cases.append(('random-'+str(block),[dict(id=int(block),capacity=600)]))
    before=e.input;report=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,plan in cases:
            fast=e.calculate(plan);reference=e.calculate(plan,reference=True)
            np.testing.assert_allclose(fast,reference,atol=TOLERANCE,rtol=0,equal_nan=True)
            pd.testing.assert_index_equal(fast.index,reference.index)
            assert e.calculate_deficit(plan)==float(reference.demand_left.sum())
            optimized_problem,_=e.problem(e.prepare(plan))
            a=mps_digest(optimized_problem,Path(tmp)/'fast.mps');b=mps_digest(e.reference_problem(plan),Path(tmp)/'reference.mps')
            assert a==b, f'MPS differs: {name}'
            pd.testing.assert_frame_equal(before,e.input)
            report.append(dict(name=name,plan=plan,deficit=float(reference.demand_left.sum()),max_delta=float(np.nanmax(np.abs(fast.to_numpy()-reference.to_numpy()))),identical_mps=True,mps_sha256=a))
            print('PASS',name,report[-1]['deficit'],'MPS identical',flush=True)
    # Independence: reverse scenario order must not change any outcome.
    for case in reversed(report):assert e.calculate_deficit(case['plan'])==case['deficit']
    assert e.calculate_deficit([dict(id=2,capacity=600)])==e.calculate_deficit([dict(id=2,capacity=300),dict(id=2,capacity=300)])
    for array in [e._population,e._capacity,e._demand,e.matrix,e._sources[0],e._times[0]]:
        assert not array.flags.writeable
        if array.size:
            try:array.flat[0]=0
            except ValueError:pass
            else:raise AssertionError('Mutable source array')
    detached=e.input;detached.iloc[0,0]=999999;pd.testing.assert_frame_equal(before,e.input)
    comparisons=[dict(id=i,reduction=e.deficit-case['deficit'],error=None) for i,case in enumerate(report)]
    ref_best=best_rows(comparisons)
    fast_comparisons=[dict(id=i,reduction=e.deficit-e.calculate_deficit(case['plan']),error=None) for i,case in enumerate(report)]
    assert best_rows(fast_comparisons)==ref_best
    assert len(best_rows([dict(id=1,reduction=600,error=None),dict(id=2,reduction=600+1e-9,error=None),dict(id=3,reduction=599,error=None)]))==2
    out=dict(kind='TEST_ONLY_NOT_FULL_SEARCH_NOT_USER_CHOICES',info=e.info(),cases=report,independence=True,immutable=True,ties=True,per_block_all_columns=True)
    (ROOT/'state').mkdir(exist_ok=True)
    (ROOT/'state'/'optimization_equivalence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

if __name__=='__main__':validate(Engine())
