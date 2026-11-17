"""BlocksNet 1.0.0a9 model; cached immutable inputs, identical CBC problems."""
import hashlib
import importlib.metadata
import json
import math
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from blocksnet.analysis.provision import competitive_provision
from blocksnet.analysis.provision.competitive import core
from blocksnet.config import service_types_config
from loguru import logger
from pulp import LpInteger, LpMaximize, LpProblem, LpVariable, LpStatus, PULP_CBC_CMD, LpAffineExpression, LpConstraint, LpConstraintLE

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('SCHOOL_DATA', r'D:\Рабочий стол\ITMO AI\python-package-boilerplate-updated\examples\data_tyumen\refined'))
TOLERANCE = 1e-7
METHOD_VERSION = 'blocksnet-1.0.0a9-array-cbc-v2'
REFERENCE_BUILDER = core._set_lp_problem

def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_properties(name):
    data=json.loads((SOURCE/name).read_text(encoding='utf-8'))
    return pd.DataFrame([f['properties'] for f in data['features']]).set_index('id')

def validate_plan(plan, ids):
    if not isinstance(plan,list):raise ValueError('План должен быть списком школ')
    result={}
    for item in plan:
        if not isinstance(item,dict):raise ValueError('Школа должна содержать ID и вместимость')
        block,capacity=item.get('id'),item.get('capacity')
        if type(block) is not int or block not in ids:raise ValueError('Неизвестный идентификатор квартала')
        if type(capacity) is not int or capacity<=0:raise ValueError('Вместимость — положительное целое число мест')
        result[block]=result.get(block,0)+capacity
    return result

def best_rows(rows):
    good=[r for r in rows if r.get('error') is None]
    if not good:return []
    best=max(r['reduction'] for r in good)
    return [r for r in good if abs(r['reduction']-best)<=TOLERANCE]

@dataclass
class Prepared:
    capacity: np.ndarray
    own: np.ndarray
    demand_left: np.ndarray
    capacity_left: np.ndarray

class Engine:
    def __init__(self):
        logger.disable('blocksnet')
        self._lock=threading.RLock()
        self._source_names=('buildings_blocks.geojson','schools_blocks.geojson','blocks_refined.geojson','accessibility_matrix.parquet','provision_schools.csv','schools.geojson','tyumen_schools_82.csv','population_repair_report.json')
        self.school_count=len(json.loads((SOURCE/'schools.geojson').read_text(encoding='utf-8'))['features'])
        self._source_stamps={name:((SOURCE/name).stat().st_size,(SOURCE/name).stat().st_mtime_ns) for name in self._source_names}
        self.version=importlib.metadata.version('blocksnet')
        if self.version!='1.0.0a9':raise RuntimeError(f'Нужен BlocksNet 1.0.0a9, установлен {self.version}')
        cfg=service_types_config['school'];self.norm,self.access=int(cfg['demand']),int(cfg['accessibility'])
        self.population_metadata=read_properties('buildings_blocks.geojson')[['population_status','population_unresolved','residential_buildings','inferred_residential_buildings','unresolved_buildings']]
        self.population_report=json.loads((SOURCE/'population_repair_report.json').read_text(encoding='utf-8'))
        self._input=read_properties('buildings_blocks.geojson')[['population']].join(read_properties('schools_blocks.geojson')[['capacity']])
        self._input['capacity']=self._input.capacity.fillna(0).astype(int)
        self.ids=tuple(map(int,self._input.index));self.id_set=frozenset(self.ids)
        self.positions={block:i for i,block in enumerate(self.ids)}
        self._matrix=pd.read_parquet(SOURCE/'accessibility_matrix.parquet')
        self._matrix.index=self._matrix.index.astype(int);self._matrix.columns=self._matrix.columns.astype(int)
        geo_ids=read_properties('blocks_refined.geojson').index
        if not self._input.index.is_unique or not self._matrix.index.is_unique or not self._matrix.columns.is_unique:raise ValueError('Дублирующиеся идентификаторы')
        if not (self._input.index.equals(self._matrix.index) and self._input.index.equals(self._matrix.columns) and self._input.index.equals(geo_ids)):raise ValueError('Идентификаторы и порядок кварталов/матрицы не совпадают')
        if not np.isfinite(self._input.to_numpy()).all() or (self._input.to_numpy()<0).any():raise ValueError('Некорректное население или вместимость')
        # The original function validates the complete matrix here, once.
        start=time.perf_counter()
        self.baseline=self.calculate([],reference=True)
        self.reference_seconds=time.perf_counter()-start
        saved=pd.read_csv(SOURCE/'provision_schools.csv',index_col=0)
        if not saved.index.equals(self.baseline.index):raise ValueError('Сохранённая обеспеченность: другие идентификаторы')
        np.testing.assert_allclose(self.baseline[saved.columns],saved,atol=TOLERANCE,rtol=0,equal_nan=True)
        self.deltas=(self.baseline[saved.columns]-saved).abs().max().to_dict()
        self.deficit=float(self.baseline.demand_left.sum())
        self._population=self._input.population.to_numpy(copy=True)
        self._capacity=self._input.capacity.to_numpy(copy=True)
        # Use the literal scalar expression from the installed library once.
        self._demand=np.array([math.ceil(p/1000*self.norm) for p in self._population],dtype=np.int64)
        np.testing.assert_array_equal(self._demand,self.baseline.demand)
        values=self._matrix.to_numpy(copy=False)
        if (values<0).any():raise ValueError('Отрицательные времена в пути')
        self._sources=[];self._times=[]
        for col in range(len(self.ids)):
            source=np.flatnonzero(values[:,col]<=self.access)
            travel=values[source,col].copy()
            source.flags.writeable=False;travel.flags.writeable=False
            self._sources.append(source);self._times.append(travel)
        self._sources=tuple(self._sources);self._times=tuple(self._times)
        for array in [self._population,self._capacity,self._demand]:array.flags.writeable=False
        for block in self._matrix._mgr.blocks:block.values.flags.writeable=False
        # Content hashes, not mtime/size. New method keys cannot reuse legacy rows.
        self.source_hashes={name:sha256_file(SOURCE/name) for name in self._source_names}
        self.assert_sources_unchanged()
        solver=PULP_CBC_CMD(msg=False)
        self.identity=dict(sources=self.source_hashes,blocksnet=self.version,blocksnet_core=sha256_file(core.__file__),pulp=importlib.metadata.version('pulp'),numpy=np.__version__,pandas=pd.__version__,cbc=sha256_file(solver.path),cbc_options=dict(msg=False),norm=self.norm,accessibility=self.access,self_supply=True,max_depth=1,method=METHOD_VERSION,engine_source=sha256_file(Path(__file__)))
        self.fingerprint=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()
        start=time.perf_counter();optimized=self.calculate([])
        self.seconds=time.perf_counter()-start
        np.testing.assert_allclose(optimized,self.baseline,atol=TOLERANCE,rtol=0,equal_nan=True)
        self.search_seconds=self.seconds

    @property
    def input(self):
        # Callers receive a detached copy; the calculation uses private arrays.
        return self._input.copy(deep=True)

    def assert_sources_unchanged(self):
        for name,stamp in self._source_stamps.items():
            stat=(SOURCE/name).stat()
            if (stat.st_size,stat.st_mtime_ns)!=stamp:
                raise RuntimeError('Исходный файл изменился после загрузки; перезапустите приложение: '+name)

    @property
    def matrix(self):
        # Read-only ndarray view, without the DataFrame mutation interface.
        view=self._matrix.to_numpy(copy=False).view();view.flags.writeable=False
        return view

    def prepare(self,plan):
        additions=validate_plan(plan,self.id_set)
        capacity=self._capacity.copy()
        for block,value in additions.items():
            position=self.positions[block]
            if value>np.iinfo(np.int64).max-int(capacity[position]):raise ValueError('Вместимость превышает диапазон целочисленной модели')
            capacity[position]+=value
        own=np.minimum(self._demand,capacity)
        return Prepared(capacity,own,self._demand-own,capacity-own)

    def problem(self,state):
        di=np.flatnonzero(state.demand_left>0);ci=np.flatnonzero(state.capacity_left>0)
        rs=[];cs=[];ts=[]
        for c in ci:
            sources=self._sources[c];mask=state.demand_left[sources]>0
            sources=sources[mask]
            if len(sources):
                rs.append(sources);cs.append(np.full(len(sources),c,dtype=np.int64));ts.append(self._times[c][mask])
        if rs:
            rr=np.concatenate(rs);cc=np.concatenate(cs);tt=np.concatenate(ts)
            order=np.lexsort((cc,rr));rr,cc,tt=rr[order],cc[order],tt[order]
        else:
            rr=np.array([],dtype=np.int64);cc=rr.copy();tt=np.array([],dtype=self._matrix.dtypes.iloc[0])
        # Original demand-major, capacity-minor product order, names and scalar
        # dtypes. Do not vectorize weights: NumPy array promotion can differ.
        products=[(self.ids[r],self.ids[c]) for r,c in zip(rr,cc)]
        prob=LpProblem('Provision',LpMaximize)
        variables=LpVariable.dicts('link',products,0,None,cat=LpInteger)
        prob+=LpAffineExpression([(variables[pair],(state.demand_left[r]*state.capacity_left[c])/(distance+1)) for r,c,distance,pair in zip(rr,cc,tt,products)])
        dc={r:[] for r in di};sc={c:[] for c in ci};lookup={}
        for r,c,pair in zip(rr,cc,products):
            var=variables[pair];dc[r].append(var);sc[c].append(var);lookup[var.name]=(r,c)
        # Keep even empty constraints, preserving the original MPS row order.
        for r,vs in dc.items():prob+=LpConstraint(e=LpAffineExpression(dict.fromkeys(vs,1)),sense=LpConstraintLE,rhs=state.demand_left[r])
        for c,vs in sc.items():prob+=LpConstraint(e=LpAffineExpression(dict.fromkeys(vs,1)),sense=LpConstraintLE,rhs=state.capacity_left[c])
        return prob,lookup

    def _solve(self,plan,details):
        state=self.prepare(plan);prob,lookup=self.problem(state)
        prob.solve(PULP_CBC_CMD(msg=False))
        if LpStatus[prob.status]!='Optimal':raise RuntimeError('CBC не подтвердил оптимальность: '+LpStatus[prob.status])
        if details:
            within=state.own.copy();capacity_within=np.zeros(len(self.ids),dtype=np.int64)
        # Same lexicographic variable order as original postprocessing.
        for var in prob.variables():
            if var.name not in lookup:continue
            value=var.value()
            if value is None:raise RuntimeError('CBC не вернул значение переменной')
            if value>0:
                integer=round(value)
                if abs(value-integer)>TOLERANCE:raise RuntimeError('Нецелочисленное решение CBC')
                r,c=lookup[var.name];state.demand_left[r]-=integer
                if details:
                    within[r]+=integer;capacity_within[c]+=integer;state.capacity_left[c]-=integer
        if not details:return float(state.demand_left.sum())
        zeros=np.zeros(len(self.ids),dtype=np.int64)
        with np.errstate(divide='ignore',invalid='ignore'):
            provision=within/self._demand
        arrays=dict(demand=self._demand,capacity=state.capacity,demand_left=state.demand_left,demand_within=within,demand_without=zeros,capacity_left=state.capacity_left,capacity_within=capacity_within,capacity_without=zeros,provision_strong=provision,provision_weak=provision.copy())
        return pd.DataFrame(arrays,index=self.baseline.index.copy(),columns=self.baseline.columns)

    def calculate(self,plan,reference=False):
        with self._lock:
            if reference:
                additions=validate_plan(plan,self.id_set)
                blocks=self._input.copy(deep=True)
                for block,value in additions.items():blocks.loc[block,'capacity']+=value
                result,_=competitive_provision(blocks,self._matrix,self.access,demand=self.norm,self_supply=True,max_depth=1)
                return result
            return self._solve(plan,details=True)

    def calculate_deficit(self,plan):
        """Identical whole-territory solve; no DataFrame or UI rows allocated."""
        with self._lock:return self._solve(plan,details=False)

    def reference_problem(self,plan):
        additions=validate_plan(plan,self.id_set);blocks=self._input.copy(deep=True)
        for block,value in additions.items():blocks.loc[block,'capacity']+=value
        blocks=core._initialize_provision_df(blocks,self.norm);core._supply_self(blocks)
        return REFERENCE_BUILDER(blocks,self._matrix,self.access)

    def rows(self,result):
        b=self.baseline
        frame=pd.DataFrame(dict(id=self.ids,population=self._population,base_capacity=b.capacity.to_numpy(),capacity=result.capacity.to_numpy(),demand=self._demand,base_deficit=b.demand_left.to_numpy(),deficit=result.demand_left.to_numpy(),reduction=b.demand_left.to_numpy()-result.demand_left.to_numpy(),base_provision=b.provision_strong.to_numpy(),provision=result.provision_strong.to_numpy()))
        frame=frame.join(self.population_metadata.reindex(self.ids).reset_index(drop=True))
        return frame.astype(object).where(pd.notna(frame),None).to_dict('records')

    def summary(self,result):
        deficit=float(result.demand_left.sum())
        return dict(capacity=float(result.capacity.sum()),added=float(result.capacity.sum()-self._capacity.sum()),deficit=deficit,reduction=self.deficit-deficit,demand=float(self._demand.sum()))

    def info(self):
        return dict(count=len(self.ids),population=float(self._population.sum()),version=self.version,norm=self.norm,accessibility=self.access,seconds=self.search_seconds,reference_seconds=self.reference_seconds,max_deltas=self.deltas,fingerprint=self.fingerprint,source=str(SOURCE),tolerance=TOLERANCE,method=METHOD_VERSION,source_hashes=self.source_hashes,population_inference=self.population_report['classification_counts'],population_unresolved_blocks=self.population_report['unresolved_zero_blocks'],school_count=self.school_count,school_source='tyumen_schools_82.csv',capacity_note='Оценочная вместимость; разные методы из CSV, не единый реестр проектных мощностей',benchmark=getattr(self,'benchmark',None))

    def geometry(self):
        import geopandas as gpd
        gdf=gpd.read_file(SOURCE/'blocks_refined.geojson').set_index('id').to_crs(4326)
        return json.loads(gdf.reset_index()[['id','geometry']].to_json())

    def school_geometry(self):
        import geopandas as gpd
        schools=gpd.read_file(SOURCE/'schools.geojson')
        if not (schools.geometry.geom_type=='Point').all():
            schools=schools.copy();schools.geometry=schools.geometry.representative_point()
        return json.loads(schools.to_crs(4326).to_json(drop_id=True))
