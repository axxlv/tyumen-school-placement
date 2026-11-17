import importlib.util
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import unittest
import tempfile

ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=ROOT/'examples';DATA=EXAMPLES/'data_tyumen'/'refined'
spec=importlib.util.spec_from_file_location('school_csv_update',EXAMPLES/'update_schools_csv.py')
update=importlib.util.module_from_spec(spec);spec.loader.exec_module(update)

def test_all_82_records_preserve_coordinates_capacity_and_provenance():
    source=update.load_csv(DATA/'tyumen_schools_82.csv')
    saved=gpd.read_file(DATA/'schools.geojson').set_index('id')
    assert len(source)==len(saved)==82
    assert saved.index.is_unique
    np.testing.assert_array_equal(saved.index,source.index)
    np.testing.assert_allclose(saved.geometry.x,source.geometry.x,atol=1e-12,rtol=0)
    np.testing.assert_allclose(saved.geometry.y,source.geometry.y,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(saved.capacity,source.capacity)
    assert saved.capacity.sum()==106942
    for name in update.REQUIRED:
        assert saved[name].fillna('').astype(str).tolist()==source[name].fillna('').astype(str).tolist()

def test_aggregation_keeps_82_and_correct_capacity_per_block():
    schools=update.load_csv(DATA/'tyumen_schools_82.csv')
    blocks=update.read_blocks('blocks_refined.geojson')
    first,aggregated=update.aggregate_csv(schools,blocks)
    second,repeated=update.aggregate_csv(schools,blocks)
    pd.testing.assert_frame_equal(first,second)
    pd.testing.assert_frame_equal(aggregated,repeated)
    assert aggregated['count'].sum()==82
    assert aggregated.capacity.sum()==106942
    expected=first.groupby('block_id').capacity.sum()
    np.testing.assert_array_equal(aggregated.loc[expected.index,'capacity'],expected)
    changed=schools.copy();chosen=changed.index[0];changed.loc[chosen,'capacity']+=137
    _,modified=update.aggregate_csv(changed,blocks)
    delta=modified.capacity-aggregated.capacity
    assert delta.sum()==137 and delta.loc[first.loc[chosen,'block_id']]==137
    assert (delta!=0).sum()==1

def check_reject_duplicate_and_invalid_capacity(tmp_path):
    frame=pd.read_csv(DATA/'tyumen_schools_82.csv',sep=';',encoding='utf-8-sig',dtype=str,keep_default_na=False)
    for value in ['0','-1','12.5','not-a-number','']:
        edited=frame.copy();edited.loc[0,'Оценочная ёмкость']=value
        path=tmp_path/'invalid.csv';edited.to_csv(path,sep=';',index=False,encoding='utf-8-sig')
        with unittest.TestCase().assertRaises(ValueError):update.load_csv(path)
    edited=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    edited.to_csv(tmp_path/'duplicate.csv',sep=';',index=False,encoding='utf-8-sig')
    with unittest.TestCase().assertRaises(ValueError):update.load_csv(tmp_path/'duplicate.csv')

def test_capacity_is_solver_input_and_all_unrelated_metrics_unchanged():
    result=pd.read_csv(DATA/'results_table.csv',index_col=0)
    old=pd.read_csv(DATA.parent/'previous_population_version'/'results_table.csv',index_col=0)
    provision=pd.read_csv(DATA/'provision_schools.csv',index_col=0)
    schools=update.read_blocks('schools_blocks.geojson')
    np.testing.assert_array_equal(provision.capacity,schools.capacity)
    np.testing.assert_array_equal(result.capacity,schools.capacity)
    assert int(provision.capacity.sum())==106942
    quality=json.loads((DATA/'quality_report.json').read_text(encoding='utf-8'))
    assert int(provision.demand.sum())==quality['total_demand']
    assert int(provision.demand_left.sum())==quality['total_deficit']
    assert int(provision.demand_within.sum())==quality['served_within']
    independent=['capacity','land_use','share','area_accessibility','residential','business','recreation','industrial','transport','special','agriculture']
    pd.testing.assert_frame_equal(result[independent],old[independent],check_dtype=False)
    assert all(json.loads((DATA/'schools_csv_validation.json').read_text(encoding='utf-8'))['checks'].values())

def test_dashboard_embeds_every_school_once_with_exact_capacity_coordinates():
    page=(EXAMPLES/'data_tyumen'/'dashboard_refined.html').read_text(encoding='utf-8')
    begin=page.index('var schoolData = ')+len('var schoolData = ')
    geo,_=json.JSONDecoder().raw_decode(page[begin:])
    source=update.load_csv(DATA/'tyumen_schools_82.csv')
    assert len(geo['features'])==82
    assert {f['properties']['id'] for f in geo['features']}==set(source.index)
    for feature in geo['features']:
        row=source.loc[feature['properties']['id']]
        assert feature['properties']['capacity']==row.capacity
        assert feature['properties']['capacity_method']==row.capacity_method
        np.testing.assert_allclose(feature['geometry']['coordinates'],[row.longitude,row.latitude],atol=1e-12,rtol=0)
    assert 'schoolLayer.addTo(map)' in page
    assert '250 мест' not in page and '70 школ' not in page

def test_reject_duplicate_and_invalid_capacity():
    with tempfile.TemporaryDirectory() as folder:
        check_reject_duplicate_and_invalid_capacity(Path(folder))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for name,f in globals().items() if name.startswith('test_') and callable(f))

if __name__=='__main__':unittest.main(verbosity=2)
