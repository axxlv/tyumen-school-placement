"""Run with the same Python as server.py. Tests do not record user's A/B choices."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from engine import Engine, validate_plan, best_rows, TOLERANCE

class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=Engine()

    def test_01_baseline_and_zero(self):
        np.testing.assert_allclose(self.e.calculate([]),self.e.baseline,atol=TOLERANCE,rtol=0,equal_nan=True)
        self.assertEqual(len(self.e.ids),12805)
        self.assertEqual(self.e.deficit,self.e.population_report['total_deficit'])

    def test_02_capacity_validation(self):
        for capacity in [0,-1,2.5,'600',True,None]:
            with self.assertRaises(ValueError):validate_plan([{'id':2,'capacity':capacity}],self.e.ids)
        with self.assertRaises(ValueError):validate_plan([{'id':999999,'capacity':600}],self.e.ids)
        self.assertEqual(validate_plan([{'id':2,'capacity':300},{'id':2,'capacity':300}],self.e.ids),{2:600})

    def test_03_reference_equivalence_scenarios_and_immutability(self):
        before=self.e.input.copy(deep=True)
        plans=[[{'id':2,'capacity':600}],[{'id':0,'capacity':600}],[{'id':2,'capacity':300},{'id':12804,'capacity':300}]]
        timings=[]
        for plan in plans:
            start=time.perf_counter();fast=self.e.calculate(plan);seconds=time.perf_counter()-start
            reference=self.e.calculate(plan,reference=True)
            np.testing.assert_allclose(fast,reference,atol=TOLERANCE,rtol=0,equal_nan=True)
            self.assertEqual(fast.capacity.sum()-before.capacity.sum(),600)
            for item in plan:self.assertEqual(fast.loc[item['id'],'capacity'],before.loc[item['id'],'capacity']+item['capacity'])
            pd.testing.assert_frame_equal(before,self.e.input)
            rows=self.e.rows(fast)
            self.assertEqual(sum(r['deficit'] for r in rows),self.e.summary(fast)['deficit'])
            timings.append(dict(plan=plan,seconds=seconds,deficit=float(fast.demand_left.sum())))
        # Order independence: a new call starts from original, not last scenario.
        first=self.e.calculate(plans[0]); repeated=self.e.calculate(plans[0])
        pd.testing.assert_frame_equal(first,repeated)
        combined=self.e.calculate([{'id':2,'capacity':300},{'id':2,'capacity':300}])
        pd.testing.assert_frame_equal(first,combined)
        pd.testing.assert_frame_equal(self.e.baseline,self.e.calculate([]))
        (Path(__file__).parent/'state'/'scenario_validation.json').write_text(json.dumps(timings,ensure_ascii=False,indent=2),encoding='utf-8')

    def test_04_ties(self):
        self.assertEqual([r['id'] for r in best_rows([dict(id=1,reduction=600,error=None),dict(id=2,reduction=600+1e-9,error=None),dict(id=3,reduction=599,error=None),dict(id=4,reduction=None,error='error')])],[1,2])

class SearchWorkerTests(unittest.TestCase):
    def test_schema_upgrade_preserves_legacy_history(self):
        import server,sqlite3
        from contextlib import contextmanager
        memory=sqlite3.connect(':memory:');memory.row_factory=sqlite3.Row
        memory.executescript('''
        CREATE TABLE jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,plan TEXT,capacity INTEGER,processed INTEGER DEFAULT 0,total INTEGER DEFAULT 1,error TEXT,result TEXT,created REAL,fingerprint TEXT);
        CREATE TABLE scenarios(id TEXT PRIMARY KEY,name TEXT,plan TEXT,result TEXT,created REAL,fingerprint TEXT);
        CREATE TABLE experiments(label TEXT PRIMARY KEY,plan TEXT,reason TEXT,created REAL,fingerprint TEXT);
        INSERT INTO jobs VALUES('legacy','scenario','complete','[]',NULL,1,1,NULL,'{}',123,'old');
        INSERT INTO scenarios VALUES('legacy','Сохранённый сценарий','[]','{}',123,'old');
        INSERT INTO experiments VALUES('А','[]','Моё обоснование',123,'old');
        ''')
        before={table:[tuple(r) for r in memory.execute('SELECT * FROM '+table)] for table in ['jobs','scenarios','experiments']}
        @contextmanager
        def isolated_db():
            with memory:yield memory
        try:
            with patch.object(server,'db',isolated_db):server.initialize_schema()
            for table in ['scenarios','experiments']:
                self.assertEqual([tuple(r) for r in memory.execute('SELECT * FROM '+table)],before[table])
            row=memory.execute('SELECT * FROM jobs').fetchone()
            self.assertEqual(tuple(row)[:11],before['jobs'][0]);self.assertEqual(row['cached'],0)
        finally:memory.close()

    def test_all_12805_candidates_and_cancel_resume(self):
        import server,sqlite3
        from contextlib import contextmanager
        from types import SimpleNamespace
        memory=sqlite3.connect(':memory:');memory.row_factory=sqlite3.Row
        with server.db() as source:source.backup(memory)
        @contextmanager
        def isolated_db():
            with memory:yield memory
        calls=[]
        class Fake:
            deficit=1000
            fingerprint='test'
            ids=list(range(12805))
            def calculate_deficit(self,plan):
                self_test.assertEqual(len(plan),1)
                self_test.assertEqual(plan[0]['capacity'],600)
                calls.append(plan[0]['id'])
                if len(calls)==3:server.CANCEL.add('full-test')
                return 400.0
        self_test=self
        try:
            with patch.object(server,'db',isolated_db),patch.object(server,'ENGINE',Fake()):
                with server.db() as c:c.execute("INSERT INTO jobs(id,kind,status,plan,capacity,total,fingerprint) VALUES('full-test','search','queued','[]',600,12805,'test')")
                server.run_job('full-test')
                self.assertEqual(len(calls),3)
                server.run_job('full-test')
                self.assertEqual(calls,list(range(12805)))
                result=server.ranking('full-test')
                self.assertEqual(len(result),12805)
                self.assertEqual(len(best_rows(result)),12805)
                with server.db() as c:row=c.execute("SELECT * FROM jobs WHERE id='full-test'").fetchone()
                self.assertEqual(row['processed'],12805)
                self.assertEqual(row['status'],'complete')
        finally:memory.close()

    def test_full_loop_independence_errors_and_resume(self):
        import server
        # Exercise real worker and SQLite persistence using an isolated test DB.
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'test.sqlite'
            import sqlite3
            source=sqlite3.connect(server.DB); dest=sqlite3.connect(target)
            try:source.backup(dest)
            finally:source.close();dest.close()
            ids=[0,2,101,12804]; calls=[]
            class Fake:
                deficit=1000
                fingerprint='test'
                def __init__(self):self.ids=ids
                def calculate_deficit(self,plan):
                    calls.append(plan)
                    if plan[0]['id']==101:raise RuntimeError('Контрольная ошибка')
                    return 400.0
            with patch.object(server,'DB',target),patch.object(server,'ENGINE',Fake()):
                with server.db() as c:
                    c.execute("INSERT INTO jobs(id,kind,status,plan,capacity,total,fingerprint) VALUES('test','search','queued','[]',600,4,'test')")
                server.run_job('test')
                self.assertEqual(calls,[[dict(id=i,capacity=600)] for i in ids])
                result=server.ranking('test');self.assertEqual(len(result),4)
                self.assertEqual(len(best_rows(result)),3)
                self.assertEqual(sum(r['error'] is not None for r in result),1)
                server.run_job('test');self.assertEqual(len(calls),4)
                with server.db() as c:row=c.execute("SELECT * FROM jobs WHERE id='test'").fetchone()
                self.assertEqual(row['status'],'complete_with_errors');self.assertEqual(row['processed'],4)

    def test_cache_identity_capacity_and_method(self):
        import server,sqlite3
        from contextlib import contextmanager
        memory=sqlite3.connect(':memory:');memory.row_factory=sqlite3.Row
        with server.db() as source:source.backup(memory)
        @contextmanager
        def isolated_db():
            with memory:yield memory
        class Fake:
            ids=[0,2,101];deficit=1000;fingerprint='data-and-method-a'
            def __init__(self):self.calls=[]
            def calculate_deficit(self,plan):
                self.calls.append(plan)
                return float(1000-plan[0]['capacity'])
        fake=Fake()
        def run(job,capacity,fp=None):
            with server.db() as c:c.execute('INSERT INTO jobs(id,kind,status,plan,capacity,total,fingerprint) VALUES(?,?,?,?,?,?,?)',(job,'search','queued','[]',capacity,len(fake.ids),fp or fake.fingerprint))
            server.run_job(job)
        try:
            with patch.object(server,'db',isolated_db),patch.object(server,'ENGINE',fake):
                run('first',600);self.assertEqual(len(fake.calls),3)
                run('repeat',600);self.assertEqual(len(fake.calls),3)
                with server.db() as c:row=c.execute("SELECT * FROM jobs WHERE id='repeat'").fetchone()
                self.assertEqual(row['cached'],3)
                run('other-capacity',300);self.assertEqual(len(fake.calls),6)
                fake.fingerprint='data-and-method-b'
                run('other-data',600);self.assertEqual(len(fake.calls),9)
                with patch.object(server,'METHOD_VERSION','other-method'):
                    run('other-method',600);self.assertEqual(len(fake.calls),12)
                run('stale-job',600,fp='data-and-method-a');self.assertEqual(len(fake.calls),12)
                with server.db() as c:row=c.execute("SELECT * FROM jobs WHERE id='stale-job'").fetchone()
                self.assertEqual(row['status'],'failed')
        finally:memory.close()

if __name__=='__main__':unittest.main(verbosity=2)
