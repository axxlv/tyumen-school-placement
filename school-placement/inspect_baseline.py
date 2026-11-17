import json, time, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from blocksnet.config import service_types_config
from blocksnet.analysis.provision import competitive_provision

SOURCE = Path(r'D:\Рабочий стол\ITMO AI\python-package-boilerplate\examples\data_tyumen\refined')
def props(name):
    return pd.DataFrame([f['properties'] for f in json.loads((SOURCE/name).read_text(encoding='utf-8'))['features']]).set_index('id')
if __name__ == '__main__':
    print('version', importlib.metadata.version('blocksnet'), flush=True)
    cfg=service_types_config['school']; print('config',cfg,flush=True)
    blocks=props('buildings_blocks.geojson')[['population']].join(props('schools_blocks.geojson')[['capacity']]).fillna(0)
    t=time.perf_counter(); matrix=pd.read_parquet(SOURCE/'accessibility_matrix.parquet')
    print('matrix',matrix.shape, matrix.index.dtype,matrix.columns.dtype,'seconds',time.perf_counter()-t,flush=True)
    matrix.columns=matrix.columns.astype(int); matrix.index=matrix.index.astype(int)
    assert blocks.index.equals(matrix.index) and blocks.index.equals(matrix.columns)
    t=time.perf_counter(); result,links=competitive_provision(blocks,matrix,int(cfg['accessibility']),demand=int(cfg['demand']))
    saved=pd.read_csv(SOURCE/'provision_schools.csv',index_col=0)
    delta=(result[saved.columns]-saved).abs().max().to_dict()
    report=dict(version=importlib.metadata.version('blocksnet'),demand=int(cfg['demand']),accessibility=int(cfg['accessibility']),count=len(blocks),population=float(blocks.population.sum()),capacity=float(blocks.capacity.sum()),deficit=float(result.demand_left.sum()),total_demand=float(result.demand.sum()),seconds=time.perf_counter()-t,max_deltas=delta)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    Path('baseline_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
