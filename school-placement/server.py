"""Offline browser UI and serialized background calculation worker."""
import argparse
import csv
import io
import json
import sqlite3
import threading
import time
import traceback
import uuid
import os
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from engine import Engine, ROOT, TOLERANCE, METHOD_VERSION, best_rows, validate_plan

DATA = Path(os.environ.get('SCHOOL_STATE',str(ROOT/'state')))
DATA.mkdir(exist_ok=True)
DB = DATA/'application.sqlite'
POOL = ThreadPoolExecutor(max_workers=1)
ENGINE = None
INIT_ERROR = None
LOCK = threading.RLock()
CANCEL = set()

@contextmanager
def db():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()

def initialize_schema():
  with db() as c:
    c.executescript('''
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,plan TEXT,capacity INTEGER,processed INTEGER DEFAULT 0,total INTEGER DEFAULT 1,error TEXT,result TEXT,created REAL,fingerprint TEXT);
    CREATE TABLE IF NOT EXISTS ranking(job TEXT,block INTEGER,deficit REAL,reduction REAL,error TEXT,seconds REAL,PRIMARY KEY(job,block));
    CREATE TABLE IF NOT EXISTS scenarios(id TEXT PRIMARY KEY,name TEXT,plan TEXT,result TEXT,created REAL,fingerprint TEXT);
    CREATE TABLE IF NOT EXISTS experiments(label TEXT PRIMARY KEY,plan TEXT,reason TEXT,created REAL,fingerprint TEXT);
    CREATE TABLE IF NOT EXISTS candidate_cache(fingerprint TEXT,capacity INTEGER,block INTEGER,method TEXT,deficit REAL,PRIMARY KEY(fingerprint,capacity,block,method));
    ''')
    if 'cached' not in {r[1] for r in c.execute('PRAGMA table_info(jobs)')}:
        c.execute('ALTER TABLE jobs ADD COLUMN cached INTEGER DEFAULT 0')

initialize_schema()

def update_job(job, **values):
    with db() as c:
        c.execute('UPDATE jobs SET '+','.join(f'{key}=?' for key in values)+' WHERE id=?', [*values.values(),job])

def ranking(job):
    with db() as c:
        return [dict(r) for r in c.execute('SELECT block AS id,deficit,reduction,error,seconds FROM ranking WHERE job=? ORDER BY error IS NOT NULL,reduction DESC,block', (job,))]

def run_job(job):
    with db() as c:
        row = dict(c.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone())
    update_job(job, status='running',error=None)
    try:
        if row['fingerprint']!=ENGINE.fingerprint:
            raise ValueError('Исходные данные или версия метода изменились; старый поиск нельзя продолжить')
        if row['kind'] == 'scenario':
            result = ENGINE.calculate(json.loads(row['plan']))
            payload = dict(rows=ENGINE.rows(result),summary=ENGINE.summary(result),plan=json.loads(row['plan']))
            update_job(job,status='complete',processed=1,result=json.dumps(payload,allow_nan=False))
        else:
            done = {r['id'] for r in ranking(job)}
            for block in ENGINE.ids:
                if block in done:
                    continue
                if job in CANCEL:
                    CANCEL.discard(job)
                    update_job(job,status='cancelled')
                    return
                start=time.perf_counter()
                error=None; deficit=None; reduction=None
                reused=False
                try:
                    with db() as c:
                        cached=c.execute('SELECT deficit FROM candidate_cache WHERE fingerprint=? AND capacity=? AND block=? AND method=?',(ENGINE.fingerprint,row['capacity'],block,METHOD_VERSION)).fetchone()
                    if cached is None:
                        deficit=ENGINE.calculate_deficit([dict(id=block,capacity=row['capacity'])])
                    else:
                        deficit=float(cached['deficit']);reused=True
                    reduction=ENGINE.deficit-deficit
                except Exception as exc:
                    error=str(exc)
                with db() as c:
                    c.execute('INSERT OR REPLACE INTO ranking VALUES(?,?,?,?,?,?)',(job,block,deficit,reduction,error,time.perf_counter()-start))
                    if error is None and not reused:
                        c.execute('INSERT OR REPLACE INTO candidate_cache VALUES(?,?,?,?,?)',(ENGINE.fingerprint,row['capacity'],block,METHOD_VERSION,deficit))
                    c.execute('UPDATE jobs SET processed=processed+1,cached=COALESCE(cached,0)+? WHERE id=?',(int(reused),job))
            rows=ranking(job)
            update_job(job,status='complete' if all(r['error'] is None for r in rows) else 'complete_with_errors')
    except Exception as exc:
        traceback.print_exc()
        update_job(job,status='failed',error=str(exc))

def init():
    global ENGINE, INIT_ERROR
    try:
        ENGINE=Engine()
        benchmark=DATA/'optimization_benchmark.json'
        if benchmark.exists():
            measured=json.loads(benchmark.read_text(encoding='utf-8'))
            if measured.get('fingerprint')==ENGINE.fingerprint:
                ENGINE.search_seconds=measured['optimized']['mean_seconds']
                ENGINE.benchmark=dict(count=measured['optimized']['count'],estimated_minutes=measured['estimated_minutes'],conservative_minutes=measured['conservative_minutes'],kind=measured['kind'])
        (DATA/'baseline_validation.json').write_text(json.dumps(ENGINE.info(),ensure_ascii=False,indent=2),encoding='utf-8')
        print('READY: исходный расчёт подтверждён',flush=True)
    except Exception as exc:
        INIT_ERROR=str(exc); traceback.print_exc()

def create_job(kind,plan=None,capacity=None):
    ENGINE.assert_sources_unchanged()
    if kind=='search':
        verification=DATA/'optimization_equivalence.json'
        if not verification.exists():raise ValueError('Нет отчёта проверки оптимизированного метода для полного поиска')
        verified=json.loads(verification.read_text(encoding='utf-8'))
        if verified.get('info',{}).get('fingerprint')!=ENGINE.fingerprint or not verified.get('per_block_all_columns'):
            raise ValueError('Данные или метод изменились после проверок. Сначала повторите validate_optimization.py')
    job=uuid.uuid4().hex
    with db() as c:
        busy=c.execute("SELECT id FROM jobs WHERE status IN ('running','queued')").fetchone()
        if busy:
            raise ValueError('Другой расчёт уже выполняется. Дождитесь завершения или остановите поиск.')
        c.execute('INSERT INTO jobs(id,kind,status,plan,capacity,total,created,fingerprint) VALUES(?,?,?,?,?,?,?,?)',(job,kind,'queued',json.dumps(plan or []),capacity,len(ENGINE.ids) if kind=='search' else 1,time.time(),ENGINE.fingerprint))
    POOL.submit(run_job,job)
    return dict(job=job)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def send(self,value,status=200,ctype='application/json; charset=utf-8',filename=None):
        if isinstance(value,(dict,list)):
            value=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        if isinstance(value,str): value=value.encode('utf-8')
        self.send_response(status); self.send_header('Content-Type',ctype)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(value)))
        if filename: self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers(); self.wfile.write(value)

    def do_GET(self):
        try:
            self.get()
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as exc:
            self.send(dict(error=str(exc)),400)

    def get(self):
        parsed=urlparse(self.path); path=parsed.path; q=parse_qs(parsed.query)
        if path in ('/','/app.js','/style.css'):
            name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
            return self.send((ROOT/'web'/name).read_bytes(),ctype={'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}[path])
        if path=='/api/status':
            return self.send(dict(ready=ENGINE is not None,error=INIT_ERROR))
        if ENGINE is None:
            return self.send(dict(error=INIT_ERROR or 'Загрузка матрицы и проверка исходного расчёта'),503)
        if path=='/api/base': return self.send(dict(info=ENGINE.info(),rows=ENGINE.rows(ENGINE.baseline),summary=ENGINE.summary(ENGINE.baseline)))
        if path=='/api/geometry': return self.send(ENGINE.geometry())
        if path=='/api/schools': return self.send(ENGINE.school_geometry())
        if path=='/practice2':
            from engine import SOURCE
            return self.send((SOURCE.parent/'dashboard_refined.html').read_bytes(),ctype='text/html; charset=utf-8')
        if path=='/api/scenarios':
            with db() as c:
                rows=[dict(r) for r in c.execute('SELECT * FROM scenarios ORDER BY created')]
            for r in rows:
                r['plan']=json.loads(r['plan']); r['result']=json.loads(r['result']);r['compatible']=r['fingerprint']==ENGINE.fingerprint
            return self.send(rows)
        if path=='/api/experiments':
            with db() as c: rows=[dict(r) for r in c.execute('SELECT * FROM experiments')]
            for r in rows:r['plan']=json.loads(r['plan']);r['compatible']=r['fingerprint']==ENGINE.fingerprint
            return self.send(rows)
        if path=='/api/jobs':
            with db() as c: rows=[dict(r) for r in c.execute('SELECT id,kind,status,capacity,processed,total,error,created,fingerprint,cached FROM jobs ORDER BY created DESC')]
            for r in rows:r['compatible']=r['fingerprint']==ENGINE.fingerprint
            return self.send(rows)
        if path=='/api/job':
            job=q['id'][0]
            with db() as c: raw=c.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
            if raw is None: raise ValueError('Расчёт не найден для текущих исходных данных')
            row=dict(raw); row['plan']=json.loads(row['plan']);row['compatible']=row['fingerprint']==ENGINE.fingerprint
            row['result']=json.loads(row['result']) if row['result'] else None
            if row['kind']=='search':
                row['ranking']=ranking(job); row['best']=best_rows(row['ranking'])
                row['exhaustive']=row['compatible'] and row['processed']==len(ENGINE.ids) and {r['id'] for r in row['ranking']}==set(ENGINE.ids) and all(r['error'] is None for r in row['ranking'])
            if 'csv' in q:
                rows=row['ranking'] if row['kind']=='search' else row['result']['rows']
                return self.csv(rows,'ranking.csv' if row['kind']=='search' else 'scenario.csv')
            return self.send(row)
        if path=='/api/base.csv':return self.csv(ENGINE.rows(ENGINE.baseline),'baseline.csv')
        return self.send(dict(error='Не найдено'),404)

    def csv(self,rows,name):
        out=io.StringIO(); out.write('\ufeff')
        if rows:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),delimiter=';');writer.writeheader();writer.writerows(rows)
        self.send(out.getvalue(),ctype='text/csv; charset=utf-8',filename=name)

    def do_POST(self):
        try:
            if self.headers.get('Origin') and self.headers['Origin'] != 'http://'+self.headers.get('Host',''):
                return self.send(dict(error='Недопустимый источник запроса'),403)
            if ENGINE is None:raise ValueError('Исходные данные ещё загружаются')
            size=int(self.headers.get('Content-Length',0))
            if size > 1_000_000:raise ValueError('Слишком большой запрос')
            body=json.loads(self.rfile.read(size) or b'{}')
            with LOCK:self.post(body)
        except Exception as exc:self.send(dict(error=str(exc)),400)

    def post(self,body):
        path=urlparse(self.path).path
        if path=='/api/calculate':
            validate_plan(body['plan'],ENGINE.ids)
            return self.send(create_job('scenario',plan=body['plan']))
        if path=='/api/search':
            validate_plan([dict(id=ENGINE.ids[0],capacity=body['capacity'])],ENGINE.ids)
            with db() as c: labels={r[0] for r in c.execute('SELECT label FROM experiments WHERE fingerprint=?',(ENGINE.fingerprint,))}
            if not {'А','Б'} <= labels:raise ValueError('Сначала зафиксируйте свой выбор и обоснования А и Б')
            return self.send(create_job('search',capacity=body['capacity']))
        if path=='/api/cancel':
            CANCEL.add(body['job']);return self.send(dict(ok=True))
        if path=='/api/resume':
            ENGINE.assert_sources_unchanged()
            with db() as c:
                row=c.execute('SELECT * FROM jobs WHERE id=? AND fingerprint=?',(body['job'],ENGINE.fingerprint)).fetchone()
                if row is None or row['kind']!='search' or row['status'] not in ('interrupted','cancelled','failed'):raise ValueError('Этот поиск нельзя продолжить')
                if c.execute("SELECT id FROM jobs WHERE status IN ('running','queued')").fetchone():raise ValueError('Другой расчёт уже запущен')
                c.execute("UPDATE jobs SET status='queued' WHERE id=?",(body['job'],))
            POOL.submit(run_job,body['job']);return self.send(dict(job=body['job']))
        if path=='/api/save':
            with db() as c:
                row=c.execute("SELECT * FROM jobs WHERE id=? AND status='complete' AND kind='scenario' AND fingerprint=?",(body['job'],ENGINE.fingerprint)).fetchone()
                if row is None:raise ValueError('Сначала рассчитайте сценарий')
                name=str(body.get('name','')).strip()
                if not name:raise ValueError('Введите название сценария')
                c.execute('INSERT INTO scenarios VALUES(?,?,?,?,?,?)',(uuid.uuid4().hex,name[:150],row['plan'],row['result'],time.time(),ENGINE.fingerprint))
            return self.send(dict(ok=True))
        if path=='/api/experiment':
            label=body['label'];plan=body['plan']; validate_plan(plan,ENGINE.ids)
            if label=='А':valid=len(plan)==1 and plan[0]['capacity']==600
            elif label=='Б':valid=len(plan)==2 and len({p['id'] for p in plan})==2 and all(p['capacity']==300 for p in plan)
            else:valid=False
            if not valid:raise ValueError('А: одна школа на 600 мест; Б: две школы по 300 в разных кварталах')
            reason=str(body.get('reason','')).strip()
            if not reason:raise ValueError('Запишите своё обоснование выбора')
            with db() as c:
                if c.execute("SELECT id FROM jobs WHERE kind='search' AND fingerprint=?",(ENGINE.fingerprint,)).fetchone():raise ValueError('После начала поиска первоначальный выбор А и Б зафиксирован')
                c.execute('INSERT OR REPLACE INTO experiments VALUES(?,?,?,?,?)',(label,json.dumps(plan),reason,time.time(),ENGINE.fingerprint))
            return self.send(dict(ok=True))
        return self.send(dict(error='Не найдено'),404)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    # A second process must not mark a live worker as interrupted.
    singleton=open(DATA/'server.lock','a+b')
    if os.name=='nt':
        import msvcrt
        singleton.seek(0);singleton.write(b'0');singleton.flush();singleton.seek(0)
        try:msvcrt.locking(singleton.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Приложение с этой базой уже запущено. Активные расчёты не изменены.')
    else:
        import fcntl
        try:fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:raise SystemExit('Приложение с этой базой уже запущено.')
    httpd=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    with db() as c:
        c.execute("UPDATE jobs SET status='interrupted' WHERE status IN ('running','queued')")
    threading.Thread(target=init,daemon=True).start()
    print(f'Откройте http://127.0.0.1:{args.port} — первый запуск проверяет исходный расчёт',flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        with db() as c:
            CANCEL.update(r[0] for r in c.execute("SELECT id FROM jobs WHERE kind='search' AND status IN ('running','queued')"))
        print('Остановка после текущего расчёта; поиск можно продолжить при следующем запуске.',flush=True)
    finally:
        httpd.server_close()
        POOL.shutdown(wait=True)
