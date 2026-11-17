"""Only the CSV school layer and its dependent results are replaced.

CSV text is metadata, never executable instructions. No scraping, downloads,
school-name deduplication, coordinate snapping, median filling or OSM appending.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.analysis.provision import competitive_provision
from blocksnet.config import service_types_config
from loguru import logger

EXAMPLES=Path(__file__).resolve().parent
DATA=EXAMPLES/'data_tyumen'/'refined'
PREVIOUS=EXAMPLES/'data_tyumen'/'previous_school_version'
REQUIRED=['№','Школа / лицей / гимназия','Корпус','Адрес в Тюмени','Широта','Долгота','Оценочная ёмкость','Метод оценки','Год/дата','Статус в модели','Статус координат','Достоверность координат','Источник координат','Исходный ID','Статус относительно старого списка','Категория','Достоверность ёмкости','Источник численности/мест','Источник адреса','Примечания']
MODEL_COLUMNS=['capacity','provision_strong','provision_weak']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_blocks(name):
    frame=gpd.read_file(DATA/name).set_index('id')
    frame.index=frame.index.astype(int)
    return frame

def load_csv(path,expected_count=82):
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as f:
        raw=list(csv.reader(f,delimiter=';'))
    if not raw or raw[0]!=REQUIRED:raise ValueError('Неожиданная структура CSV: нужны исходные 20 столбцов')
    if any(len(row)!=len(REQUIRED) for row in raw[1:]):raise ValueError('Нарушено число полей в строке CSV')
    frame=pd.read_csv(path,sep=';',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    if len(frame)!=expected_count:raise ValueError(f'Ожидалось {expected_count} записей, получено {len(frame)}')
    errors=[]
    for i,row in frame.iterrows():
        for column in ['№','Школа / лицей / гимназия','Адрес в Тюмени','Широта','Долгота','Оценочная ёмкость','Метод оценки']:
            if not row[column].strip():errors.append(f'Строка {i+2}: пустое поле {column}')
    numeric={name:pd.to_numeric(frame[name],errors='coerce') for name in ['№','Широта','Долгота','Оценочная ёмкость']}
    for name,values in numeric.items():
        for i in frame.index[~np.isfinite(values)]:errors.append(f'Строка {i+2}: некорректное число {name}')
    for name in ['№','Оценочная ёмкость']:
        values=numeric[name]
        if ((values<=0)|(values%1!=0)).any():errors.append(f'{name}: требуются положительные целые числа')
    if not numeric['Широта'].between(-90,90).all() or not numeric['Долгота'].between(-180,180).all():errors.append('Координаты вне допустимых диапазонов')
    if frame['№'].duplicated().any():errors.append('Повторяются номера записей')
    if frame['Адрес в Тюмени'].str.strip().str.casefold().duplicated().any():errors.append('Повторяются адреса корпусов')
    if pd.DataFrame({'lat':numeric['Широта'],'lon':numeric['Долгота']}).duplicated().any():errors.append('Повторяются координаты')
    if not (frame['Статус в модели']=='Основная модель').all():errors.append('Есть строки вне основной модели — требуется явное решение')
    if errors:raise ValueError('\n'.join(errors))
    frame=frame.copy()
    frame['id']=numeric['№'].astype(int)
    frame['name']=frame['Школа / лицей / гимназия']
    frame['building']=frame['Корпус'];frame['address']=frame['Адрес в Тюмени']
    frame['latitude']=numeric['Широта'];frame['longitude']=numeric['Долгота']
    frame['capacity']=numeric['Оценочная ёмкость'].astype(int)
    frame['capacity_source']='user_csv_estimate'
    frame['capacity_method']=frame['Метод оценки'];frame['capacity_confidence']=frame['Достоверность ёмкости']
    frame['capacity_reference']=frame['Источник численности/мест'];frame['capacity_year']=frame['Год/дата']
    frame['n_objects']=1
    return gpd.GeoDataFrame(frame,geometry=gpd.points_from_xy(frame.longitude,frame.latitude),crs=4326).set_index('id')

def aggregate_csv(schools,blocks):
    working=schools.to_crs(blocks.crs)
    # Keep precisely the old intersects spatial join; do not move points.
    b=blocks[['geometry']].copy();b.index.name=None
    join=gpd.sjoin(working[['geometry']],b,how='left',predicate='intersects')
    if len(join)!=len(schools) or not join.index.is_unique or join['index_right'].isna().any():
        raise ValueError('Каждая школа должна принадлежать ровно одному существующему кварталу')
    school_blocks,excess=aggregate_objects(blocks[['geometry']],working[['capacity','n_objects','geometry']])
    if len(excess):raise ValueError('Найдены школы вне кварталов')
    school_blocks['capacity']=school_blocks.capacity.astype(int)
    school_blocks['count']=school_blocks['count'].astype(int)
    school_blocks['n_objects']=school_blocks.n_objects.astype(int)
    enriched=schools.copy();enriched['block_id']=join['index_right'].astype(int).reindex(schools.index)
    assert school_blocks.capacity.sum()==schools.capacity.sum()
    assert school_blocks['count'].sum()==len(schools)
    return enriched,school_blocks

def save_gdf(frame,path):
    out=frame.copy();out.index.name='id'
    out.reset_index().to_file(path,driver='GeoJSON',encoding='UTF-8')

def run_update(csv_path=None):
    logger.disable('blocksnet')
    if importlib.metadata.version('blocksnet')!='1.0.0a9':raise RuntimeError('Требуется BlocksNet 1.0.0a9')
    path=Path(csv_path) if csv_path else DATA/'tyumen_schools_82.csv'
    schools=load_csv(path)
    blocks=read_blocks('blocks_refined.geojson')
    population=read_blocks('buildings_blocks.geojson')[['population']]
    schools,school_blocks=aggregate_csv(schools,blocks)
    boundaries=gpd.read_file(DATA/'boundaries.geojson').to_crs(4326)
    if not schools.geometry.covered_by(boundaries.geometry.union_all()).all():raise ValueError('Есть точки вне неизменной территории')
    matrix=pd.read_parquet(DATA/'accessibility_matrix.parquet')
    matrix.index=matrix.index.astype(int);matrix.columns=matrix.columns.astype(int)
    inputs=population.join(school_blocks[['capacity']])
    if not inputs.index.equals(matrix.index) or not inputs.index.equals(matrix.columns):raise ValueError('Идентификаторы кварталов не совпадают с матрицей')
    config=service_types_config['school']
    provision,links=competitive_provision(inputs,matrix,int(config['accessibility']),demand=int(config['demand']),self_supply=True,max_depth=1)
    np.testing.assert_array_equal(provision.capacity,school_blocks.capacity)
    # Read the previous version as baseline on every run: never append old points.
    previous=pd.read_csv(DATA/'results_table.csv' if (DATA/'results_table.csv').exists() else PREVIOUS/'results_table.csv',index_col=0)
    results=previous.copy()
    for column in MODEL_COLUMNS:results[column]=provision[column].reindex(results.index)
    unaffected=[c for c in previous.columns if c not in MODEL_COLUMNS]
    pd.testing.assert_frame_equal(results[unaffected],previous[unaffected])
    result_geo=read_blocks('blocks_land_use.geojson')[['geometry']].join(results)
    summary_columns=pd.read_csv(PREVIOUS/'results_summary.csv',index_col=0).index.tolist()
    summary=results[summary_columns].describe().T.round(3)
    # Independent sensitivity check: capacity must affect the actual solver.
    zero=inputs.copy();zero['capacity']=0
    no_capacity,_=competitive_provision(zero,matrix,int(config['accessibility']),demand=int(config['demand']),self_supply=True,max_depth=1)
    np.testing.assert_array_equal(no_capacity.demand_left,no_capacity.demand)
    assert provision.demand_within.sum()>0 and provision.demand_left.sum()<no_capacity.demand_left.sum()
    checks=dict(csv_82_rows=len(schools)==82,csv_ids_unique=schools.index.is_unique,positive_integer_capacity=bool((schools.capacity>0).all()),all_points_inside_territory=True,all_points_exactly_one_block=True,coordinates_preserved=True,no_legacy_points_appended=True,school_count_conserved=int(school_blocks['count'].sum())==82,capacity_sum_conserved=int(school_blocks.capacity.sum())==int(schools.capacity.sum()),capacity_reaches_solver=True,zero_capacity_changes_result=True,matrix_ids_match=True,unrelated_result_columns_unchanged=True)
    quality=json.loads((DATA/'quality_report.json' if (DATA/'quality_report.json').exists() else PREVIOUS/'quality_report.json').read_text(encoding='utf-8'))
    quality['checks'].update(checks)
    quality.update(schools=82,capacity=int(schools.capacity.sum()),school_source='tyumen_schools_82.csv',school_source_sha256=digest(path),capacity_interpretation='Оценочные вместимости из CSV; методы смешанные. Не все значения являются проектными мощностями.',capacity_methods=schools.capacity_method.value_counts().to_dict(),total_demand=int(provision.demand.sum()),total_deficit=int(provision.demand_left.sum()),served_within=int(provision.demand_within.sum()),original_capacity=17500,original_schools=70)
    validation=dict(checks=checks,rows=82,capacity=int(schools.capacity.sum()),matrix_shape=list(matrix.shape),csv_sha256=digest(path),method=dict(blocksnet='1.0.0a9',demand=int(config['demand']),accessibility=int(config['accessibility']),self_supply=True,max_depth=1),capacity_methods=quality['capacity_methods'],warnings=['Исходные оценки вместимости смешивают проектные места, численность учащихся и расчётные распределения. Значения сохранены без переоценки.','Ссылки, даты и достоверность приняты из переданного CSV; повторная внешняя проверка источников не проводилась.'])
    audit=schools.drop(columns='geometry').reset_index()[['id','name','building','address','latitude','longitude','capacity','capacity_method','capacity_confidence','block_id','capacity_year','capacity_reference']]
    save_gdf(schools,DATA/'schools.geojson')
    save_gdf(school_blocks,DATA/'schools_blocks.geojson')
    provision.to_csv(DATA/'provision_schools.csv');links.to_csv(DATA/'provision_links.csv')
    results.to_csv(DATA/'results_table.csv');summary.to_csv(DATA/'results_summary.csv')
    save_gdf(result_geo,DATA/'results_blocks.geojson')
    audit.to_csv(DATA/'schools_import_audit.csv',index=False,encoding='utf-8-sig')
    (DATA/'quality_report.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'schools_csv_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(schools=82,capacity=quality['capacity'],demand=quality['total_demand'],deficit=quality['total_deficit'],served=quality['served_within'],all_checks=all(checks.values())),ensure_ascii=False,indent=2),flush=True)
    return schools,school_blocks,provision,results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--csv',type=Path);args=parser.parse_args()
    run_update(args.csv)
