"""Local, read-only refined data; BlocksNet 1.0.0a9 calculation."""
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from blocksnet.analysis.provision import competitive_provision
from blocksnet.analysis.provision.competitive import core
from blocksnet.config import service_types_config
from loguru import logger
from pulp import LpInteger, LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('SCHOOL_DATA', r'D:\Рабочий стол\ITMO AI\python-package-boilerplate\examples\data_tyumen\refined'))
TOLERANCE = 1e-7
REFERENCE_BUILDER = core._set_lp_problem

def read_properties(name):
    data = json.loads((SOURCE / name).read_text(encoding='utf-8'))
    return pd.DataFrame([f['properties'] for f in data['features']]).set_index('id')

def validate_plan(plan, ids):
    if not isinstance(plan, list):
        raise ValueError('План должен быть списком школ')
    result = {}
    for item in plan:
        block, capacity = item.get('id'), item.get('capacity')
        if type(block) is not int or block not in ids:
            raise ValueError('Неизвестный идентификатор квартала')
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Вместимость — положительное целое число мест')
        result[block] = result.get(block, 0) + capacity
    return result

def best_rows(rows):
    good = [r for r in rows if r.get('error') is None]
    if not good:
        return []
    best = max(r['reduction'] for r in good)
    return [r for r in good if abs(r['reduction'] - best) <= TOLERANCE]

class Engine:
    def __init__(self):
        logger.disable('blocksnet')
        self.version = importlib.metadata.version('blocksnet')
        if self.version != '1.0.0a9':
            raise RuntimeError(f'Нужен BlocksNet 1.0.0a9, установлен {self.version}')
        cfg = service_types_config['school']
        self.norm, self.access = int(cfg['demand']), int(cfg['accessibility'])
        self.input = read_properties('buildings_blocks.geojson')[['population']].join(read_properties('schools_blocks.geojson')[['capacity']])
        self.input['capacity'] = self.input.capacity.fillna(0).astype(int)
        self.ids = list(map(int, self.input.index))
        self.matrix = pd.read_parquet(SOURCE / 'accessibility_matrix.parquet')
        self.matrix.index = self.matrix.index.astype(int)
        self.matrix.columns = self.matrix.columns.astype(int)
        geometry_ids = read_properties('blocks_refined.geojson').index
        if not self.input.index.is_unique or not self.matrix.index.is_unique or not self.matrix.columns.is_unique:
            raise ValueError('Дублирующиеся идентификаторы')
        if not (self.input.index.equals(self.matrix.index) and self.input.index.equals(self.matrix.columns) and self.input.index.equals(geometry_ids)):
            raise ValueError('Порядок идентификаторов кварталов, строк и столбцов матрицы не совпадает')
        if not np.isfinite(self.input.to_numpy()).all() or (self.input.to_numpy() < 0).any():
            raise ValueError('Некорректные население или вместимость')
        # Match original <=15 selection, in original row-major product order.
        # Matrix remains unchanged. Do not symmetrize or round travel times.
        self.reachable = self.matrix.to_numpy() <= self.access
        stat = (SOURCE/'accessibility_matrix.parquet').stat()
        self.fingerprint = hashlib.sha256(self.input.to_csv().encode() + f'{stat.st_size}:{stat.st_mtime_ns}:{self.version}:{self.norm}:{self.access}'.encode()).hexdigest()
        start = time.perf_counter()
        self.baseline = self.calculate([], reference=True)
        self.reference_seconds = time.perf_counter() - start
        saved = pd.read_csv(SOURCE / 'provision_schools.csv', index_col=0)
        if not saved.index.equals(self.baseline.index):
            raise ValueError('Идентификаторы сохранённой обеспеченности не совпадают')
        np.testing.assert_allclose(self.baseline[saved.columns], saved, atol=TOLERANCE, rtol=0, equal_nan=True)
        self.deltas = (self.baseline[saved.columns] - saved).abs().max().to_dict()
        start = time.perf_counter()
        optimized = self.calculate([])
        np.testing.assert_allclose(optimized[self.baseline.columns], self.baseline, atol=TOLERANCE, rtol=0, equal_nan=True)
        self.seconds = time.perf_counter() - start
        self.deficit = float(self.baseline.demand_left.sum())

    def fast_builder(self, blocks, matrix, selection_range):
        if selection_range != self.access:
            return REFERENCE_BUILDER(blocks, matrix, selection_range)
        di = np.flatnonzero(blocks.demand_left.to_numpy() > 0)
        ci = np.flatnonzero(blocks.capacity_left.to_numpy() > 0)
        rr, cc = np.nonzero(self.reachable[np.ix_(di, ci)])
        products = [(int(blocks.index[di[r]]), int(blocks.index[ci[c]])) for r, c in zip(rr, cc)]
        prob = LpProblem('Provision', LpMaximize)
        x = LpVariable.dicts('link', products, 0, None, cat=LpInteger)
        prob += lpSum((blocks.loc[a, 'demand_left'] * blocks.loc[b, 'capacity_left']) / (matrix.loc[a, b] + 1) * x[a, b] for a, b in products)
        demands, capacities = {int(blocks.index[i]): [] for i in di}, {int(blocks.index[i]): [] for i in ci}
        for a, b in products:
            demands[a].append(x[a, b]); capacities[b].append(x[a, b])
        for a, variables in demands.items():
            prob += lpSum(variables) <= blocks.loc[a, 'demand_left']
        for b, variables in capacities.items():
            prob += lpSum(variables) <= blocks.loc[b, 'capacity_left']
        return prob

    def calculate(self, plan, reference=False):
        additions = validate_plan(plan, self.ids)
        blocks = self.input.copy(deep=True)
        for block, capacity in additions.items():
            blocks.loc[block, 'capacity'] += capacity
        # Calculations are serialized by the server worker. The public library
        # function still performs initialization, self supply, CBC, postprocessing.
        previous = core._set_lp_problem
        core._set_lp_problem = REFERENCE_BUILDER if reference else self.fast_builder
        original_post = core._postprocess_lp_problem
        def checked_post(prob, *args):
            if LpStatus[prob.status] != 'Optimal':
                raise RuntimeError('CBC не подтвердил оптимальность: ' + LpStatus[prob.status])
            return original_post(prob, *args)
        core._postprocess_lp_problem = checked_post
        try:
            result, _ = competitive_provision(blocks, self.matrix, self.access, demand=self.norm, self_supply=True, max_depth=1)
            return result
        finally:
            core._set_lp_problem = previous
            core._postprocess_lp_problem = original_post

    def rows(self, result):
        rows = []
        for block in self.ids:
            b, s = self.baseline.loc[block], result.loc[block]
            rows.append(dict(id=block, population=float(self.input.loc[block,'population']), base_capacity=int(b.capacity), capacity=int(s.capacity), demand=int(s.demand), base_deficit=float(b.demand_left), deficit=float(s.demand_left), reduction=float(b.demand_left-s.demand_left), base_provision=None if pd.isna(b.provision_strong) else float(b.provision_strong), provision=None if pd.isna(s.provision_strong) else float(s.provision_strong)))
        return rows

    def summary(self, result):
        deficit = float(result.demand_left.sum())
        return dict(capacity=float(result.capacity.sum()), added=float(result.capacity.sum()-self.input.capacity.sum()), deficit=deficit, reduction=self.deficit-deficit, demand=float(result.demand.sum()))

    def info(self):
        return dict(count=len(self.ids), population=float(self.input.population.sum()), version=self.version, norm=self.norm, accessibility=self.access, seconds=self.seconds, reference_seconds=self.reference_seconds, max_deltas=self.deltas, fingerprint=self.fingerprint, source=str(SOURCE), tolerance=TOLERANCE)

    def geometry(self):
        # GeoJSON is in projected metres; transform once for offline canvas map.
        import geopandas as gpd
        gdf = gpd.read_file(SOURCE/'blocks_refined.geojson').set_index('id').to_crs(4326)
        return json.loads(gdf.reset_index()[['id','geometry']].to_json())
