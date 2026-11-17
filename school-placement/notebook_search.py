"""Notebook client for the existing school-placement server; no solver code."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXPECTED_PLANS = {
    'А': [{'id': 9345, 'capacity': 600}],
    'Б': [{'id': 9736, 'capacity': 300}, {'id': 477, 'capacity': 300}],
}


class NotebookSearch:
    def __init__(self, project, port=8767, *, expected_total=12805):
        self.project = Path(project).resolve()
        self.state = self.project / 'state'
        self.source = self.project.parent / 'python-package-boilerplate-updated/examples/data_tyumen/refined'
        self.base_url = f'http://127.0.0.1:{int(port)}'
        self.port = int(port)
        self.total = expected_total

    def api(self, path, data=None):
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(
            self.base_url + path, data=body,
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode('utf-8'))['error']
            except (ValueError, KeyError):
                message = str(exc)
            raise RuntimeError(message) from exc

    def ensure_server(self, timeout=120):
        """Connect or launch server.py, never a search. The server owns the lock."""
        try:
            status = self.api('/api/status')
        except urllib.error.URLError:
            if not (self.project / 'server.py').is_file():
                raise RuntimeError(f'Не найден server.py: {self.project}')
            self.state.mkdir(exist_ok=True)
            log_path = self.state / 'notebook-server.log'
            environment = os.environ.copy()
            environment['SCHOOL_STATE'] = str(self.state)
            environment['SCHOOL_DATA'] = str(self.source)
            options = {}
            if os.name == 'nt':
                options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            else:
                options['start_new_session'] = True
            with log_path.open('ab') as log:
                child = subprocess.Popen(
                    [sys.executable, '-X', 'utf8', str(self.project / 'server.py'), '--port', str(self.port)],
                    cwd=self.project, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, **options,
                )
            print(f'Запущен локальный сервер. Загрузка данных; журнал: {log_path}', flush=True)
            status = None
        else:
            child = None
        deadline = time.monotonic() + timeout
        while True:
            if status is not None:
                if status.get('error'):
                    raise RuntimeError('Сервер сообщил ошибку: ' + str(status['error']))
                if status.get('ready'):
                    print(f'Сервер готов: {self.base_url}. Поиск этой ячейкой не запускается.')
                    return status
            if time.monotonic() >= deadline:
                raise RuntimeError('Сервер не готов за отведённое время. См. state/notebook-server.log. '
                                   'Если сервер открыт на другом порту, укажите его в PORT; второй сервер не запускайте.')
            if child is not None and child.poll() is not None:
                raise RuntimeError('Сервер завершился при запуске. См. state/notebook-server.log. '
                                   'Возможно, эта база уже открыта сервером на другом порту.')
            time.sleep(1)
            try:
                status = self.api('/api/status')
            except urllib.error.URLError:
                status = None

    def _snapshot(self):
        database = self.state / 'application.sqlite'
        if not database.is_file():
            raise RuntimeError(f'Не найдена база приложения: {database}')
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=30) as connection:
            connection.row_factory = sqlite3.Row
            jobs = [dict(row) for row in connection.execute('SELECT * FROM jobs')]
            choices = [dict(row) for row in connection.execute('SELECT * FROM experiments')]
        return jobs, choices

    def preflight(self, *, quiet=False):
        base = self.api('/api/base')
        info = base['info']
        if (info['count'], info['norm'], info['accessibility'], info['version']) != (self.total, 120, 15, '1.0.0a9'):
            raise RuntimeError('Состав территории, нормативы или версия BlocksNet отличаются от проверенных.')
        if Path(info['source']).resolve() != self.source.resolve():
            raise RuntimeError('Сервер использует другой каталог исходных данных.')
        fingerprint = info['fingerprint']
        choices = self.api('/api/experiments')
        current = {row['label']: row for row in choices if row['fingerprint'] == fingerprint and row['compatible']}
        for label, plan in EXPECTED_PLANS.items():
            if label not in current or current[label]['plan'] != plan or not current[label]['reason'].strip():
                raise RuntimeError(f'Не найден подтверждённый выбор {label} с прежним планом и обоснованием. '
                                   'Ноутбук не меняет А/Б; проверьте их в приложении.')
        jobs = self.api('/api/jobs')
        local_jobs, local_choices = self._snapshot()
        if {j['id'] for j in jobs} != {j['id'] for j in local_jobs}:
            raise RuntimeError('Сервер и ноутбук видят разные базы или список заданий изменился. Повторите проверку.')
        for label in EXPECTED_PLANS:
            local = next((x for x in local_choices if x['label'] == label), None)
            remote = current[label]
            if local is None or any(local[k] != remote[k] for k in ['fingerprint', 'reason', 'created']):
                raise RuntimeError('Сервер и ноутбук видят разные записи А/Б.')
            if json.loads(local['plan']) != remote['plan']:
                raise RuntimeError('Сервер и ноутбук видят разные планы А/Б.')
        verification = json.loads((self.state / 'optimization_equivalence.json').read_text(encoding='utf-8'))
        if verification.get('info', {}).get('fingerprint') != fingerprint or not all(
            verification.get(key) for key in ['per_block_all_columns', 'independence', 'immutable', 'ties']
        ):
            raise RuntimeError('Отчёт эквивалентности устарел или не завершён. Полный поиск запускать нельзя.')
        active = [j for j in jobs if j['status'] in ('queued', 'running')]
        if len(active) > 1 or any(not self._matching(j, fingerprint) for j in active):
            raise RuntimeError('Выполняется другой расчёт. Он не остановлен; дождитесь его завершения.')
        if not quiet:
            print(f'Проверено: {info["count"]:,} кварталов; 120 мест / 1000 жителей; 15 минут; BlocksNet 1.0.0a9.')
            print('А: № 9345 +600; Б: № 9736 +300 и № 477 +300. Обоснования сохранены.')
            print('Подпись данных и метода:', fingerprint)
            print('База результатов:', self.state / 'application.sqlite')
            print('Совместимый поиск уже выполняется; следующая ячейка подключится к нему.' if active
                  else 'Других активных расчётов нет. Поиск этой ячейкой не запускается.')
        return dict(info=info, summary=base['summary'], jobs=jobs)

    @staticmethod
    def _matching(job, fingerprint):
        return job['kind'] == 'search' and job['capacity'] == 600 and job['fingerprint'] == fingerprint

    def start_or_resume(self, preferred_job_id=None):
        context = self.preflight(quiet=True)
        fingerprint = context['info']['fingerprint']
        compatible = [j for j in context['jobs'] if self._matching(j, fingerprint)]
        if preferred_job_id:
            selected = next((j for j in compatible if j['id'] == preferred_job_id), None)
            if selected is None:
                raise RuntimeError('Указанный поиск не найден или относится к другой вместимости/версии данных.')
        else:
            active = [j for j in compatible if j['status'] in ('queued', 'running')]
            selected = max(active or compatible, key=lambda j: j['created']) if compatible else None
        active_other = [j for j in context['jobs'] if j['status'] in ('queued', 'running')
                        and (selected is None or j['id'] != selected['id'])]
        if active_other:
            raise RuntimeError('Другой расчёт уже активен. Новое задание не создано.')
        if selected is None:
            job_id = self.api('/api/search', {'capacity': 600})['job']
            print('Создан ПОЛНЫЙ поиск по всем кварталам: школа 600 мест. ID:', job_id)
        else:
            job_id = selected['id']
            if selected['status'] in ('interrupted', 'cancelled', 'failed'):
                self.api('/api/resume', {'job': job_id})
                print(f'Продолжается поиск {job_id}. Уже сохранено вариантов: {selected["processed"]}.')
            elif selected['status'] in ('queued', 'running', 'complete', 'complete_with_errors'):
                print(f'Используется существующий поиск {job_id}; новое задание не создаётся.')
            else:
                raise RuntimeError('Неизвестное состояние поиска: ' + selected['status'])
        local, _ = self._snapshot()
        if job_id not in {j['id'] for j in local}:
            raise RuntimeError('Задание не найдено в локальной базе. Проверьте адрес сервера.')
        return job_id

    def job(self, job_id):
        job = self.api('/api/job?id=' + job_id)
        if not self._matching(job, self.api('/api/base')['info']['fingerprint']) or not job['compatible']:
            raise RuntimeError('Поиск несовместим с текущими данными.')
        return job

    @staticmethod
    def duration(seconds):
        if seconds is None:
            return 'пока нет оценки'
        seconds = max(0, int(seconds))
        return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'

    def progress_html(self, job):
        ranking = job.get('ranking', [])
        errors = [row for row in ranking if row['error'] is not None]
        elapsed = sum(row['seconds'] or 0 for row in ranking)
        remaining = (job['total'] - job['processed']) * elapsed / len(ranking) if ranking else None
        percent = 100 * job['processed'] / job['total']
        status = {'running': 'Выполняется', 'queued': 'В очереди', 'complete': 'Обход завершён',
                  'complete_with_errors': 'Обход завершён с ошибками', 'cancelled': 'Остановлен',
                  'interrupted': 'Прерван', 'failed': 'Ошибка задания'}.get(job['status'], job['status'])
        result = (f'<div><b>{html.escape(status)}</b> · школа 600 мест<br>'
                  f'<progress value="{job["processed"]}" max="{job["total"]}" style="width:100%"></progress>'
                  f'<b>{job["processed"]:,} / {job["total"]:,} кварталов · {percent:.2f}%</b><br>'
                  f'Прошедшее время расчётов: {self.duration(elapsed)} · Осталось примерно: {self.duration(remaining)}<br>'
                  f'Ошибок: {len(errors)} · Из совместимого кэша: {job.get("cached", 0)}<br>'
                  f'<small>ID: {html.escape(job["id"])}</small>')
        if job.get('error'):
            result += '<p>' + html.escape(str(job['error'])) + '</p>'
        if errors:
            result += '<ul>' + ''.join(f'<li>№ {r["id"]}: {html.escape(str(r["error"]))}</li>' for r in errors[:5]) + '</ul>'
        if job.get('exhaustive'):
            result += f'<p><b>ПОЛНЫЙ ПЕРЕБОР ЗАВЕРШЁН: {job["total"]:,} из {job["total"]:,}. Ошибок: 0.</b></p>'
            best = job.get('best', [])
            if best:
                result += f'<p>Равноценных лучших: {len(best)}. Сокращение дефицита: {best[0]["reduction"]:g} мест.</p>'
        elif job['status'] not in ('queued', 'running'):
            result += '<p><b>Успешный полный перебор не подтверждён. Глобальный лучший вариант не установлен.</b></p>'
        return result + f'<a href="{self.base_url}" target="_blank">Открыть приложение</a> · <a href="{self.base_url}/api/job?id={job["id"]}&amp;csv=1" target="_blank">Полный сохранённый рейтинг CSV</a></div>'

    def watch(self, job_id, interval=3):
        from IPython.display import HTML, display
        fingerprint = self.api('/api/base')['info']['fingerprint']
        handle = None
        try:
            while True:
                job = self.api('/api/job?id=' + job_id)
                if not self._matching(job, fingerprint) or not job['compatible']:
                    raise RuntimeError('Поиск относится к другим данным или вместимости.')
                output = HTML(self.progress_html(job))
                if handle is None:
                    handle = display(output, display_id=True)
                else:
                    handle.update(output)
                if job['status'] not in ('queued', 'running'):
                    if job.get('exhaustive'):
                        print(f'ПОЛНЫЙ ПЕРЕБОР ЗАВЕРШЁН: {job["processed"]} / {job["total"]}; ошибок 0.')
                    else:
                        print('Поиск остановлен/завершён без подтверждения полного успешного перебора. '
                              'Для прерванного задания повторите ячейки 3 и 4.')
                    return job
                time.sleep(interval)
        except KeyboardInterrupt:
            print('Остановлено только наблюдение. Сервер продолжает считать. '
                  'Для наблюдения повторите ячейку 4; для остановки поиска используйте ячейку 6.')
            return None
        except urllib.error.URLError as exc:
            raise RuntimeError('Потеряна связь с сервером. Результаты уже рассчитанных кварталов сохранены. '
                               'Повторите ячейки 2–4; новое задание вместо существующего не создавайте.') from exc

    def stop(self, job_id):
        job = self.job(job_id)
        if job['status'] in ('queued', 'running'):
            self.api('/api/cancel', {'job': job_id})
            print('Запрошена остановка после текущего квартала. Продолжение: ячейки 3 и 4.')
        else:
            print('Этот поиск уже не выполняется:', job['status'])
