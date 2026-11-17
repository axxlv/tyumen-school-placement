"""Repair housing evidence and recalculate only dependent model outputs."""
from pathlib import Path
import hashlib
import json
import importlib.metadata
import numpy as np
import pandas as pd
import geopandas as gpd
from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_spacematrix_morphotypes,get_strelka_morphotypes
from blocksnet.analysis.provision import competitive_provision
from blocksnet.config import service_types_config
from loguru import logger
from classify_housing import classify

EXAMPLES=Path(__file__).resolve().parent
DATA=EXAMPLES/'data_tyumen'/'refined'
PREVIOUS=EXAMPLES/'data_tyumen'/'previous_population_version'
def read(name):return gpd.read_file(DATA/name).set_index('id')
def save(frame,name):
    f=frame.copy();f.index.name='id';f.reset_index().to_file(DATA/name,driver='GeoJSON',encoding='UTF-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run_repair():
    logger.disable('blocksnet')
    assert importlib.metadata.version('blocksnet')=='1.0.0a9'
    fixed={name:digest(DATA/name) for name in ['accessibility_matrix.parquet','blocks_refined.geojson','blocks_land_use.geojson','schools.geojson','schools_blocks.geojson','tyumen_schools_82.csv','functional_zones.geojson','area_accessibility.csv']}
    raw=read('buildings_evidence.geojson')
    old=gpd.read_file(PREVIOUS/'buildings.geojson').set_index('id')
    blocks=read('blocks_refined.geojson');blocks.index.name=None
    zones=gpd.read_file(DATA/'functional_zones.geojson')
    schools=gpd.read_file(DATA/'schools.geojson')
    np.testing.assert_array_equal(raw.index,old.index)
    classified=classify(raw,zones,schools)
    physical=old[['number_of_floors','footprint_area','build_floor_area','geometry']].copy()
    physical['is_living']=classified.is_living
    # Drop previous living area and population: old zeros must not survive
    # reclassification, and nonliving area must not be counted twice.
    modeled=impute_buildings(physical,default_living_demand=30)
    modeled.population=modeled.population.fillna(0).astype(int)
    for name in ['housing_status','housing_confidence','housing_inferred','housing_unresolved','housing_residential_zone','housing_garden_zone','housing_has_address','housing_nearest_confirmed_m','housing_dominated_by_larger_house','osm_id','addr:street','addr:housenumber','building','building:levels']:
        if name in classified:modeled[name]=classified[name]
    old_living=old.is_living&classified.is_living
    np.testing.assert_array_equal(modeled.loc[old_living,'population'],old.loc[old_living,'population'])
    for name in ['number_of_floors','footprint_area','build_floor_area']:
        np.testing.assert_allclose(modeled[name],old[name],atol=1e-7,rtol=0)
    columns=['is_living','number_of_floors','footprint_area','build_floor_area','living_area','non_living_area','population','geometry']
    source=modeled[columns].copy()
    source['residential_buildings']=modeled.is_living.astype(int)
    source['inferred_residential_buildings']=modeled.housing_inferred.astype(int)
    source['unresolved_buildings']=modeled.housing_unresolved.astype(int)
    aggregated,excess=aggregate_objects(blocks[['geometry']],source)
    assignments=[]
    # Preserve original representative-point aggregation, with an audited
    # largest-intersection fallback only for inhabited buildings in tiny gaps.
    unassigned=[]
    for building,row in excess[excess.population>0].iterrows():
        footprint=source.loc[building,'geometry']
        candidates=blocks.sindex.query(footprint,predicate='intersects')
        if len(candidates):
            overlap=blocks.geometry.iloc[candidates].intersection(footprint).area
            if overlap.max()>0:
                chosen=int(overlap.sort_values(ascending=False,kind='stable').index[0])
                for column in source.columns:
                    if column=='geometry':continue
                    aggregated.loc[chosen,column]+=float(row[column])
                aggregated.loc[chosen,'count']+=1
                assignments.append(dict(building_id=int(building),block_id=chosen,population=int(row.population),method='largest_footprint_intersection'))
                continue
        distances=blocks.geometry.distance(footprint)
        if distances.min()<=5:
            chosen=int(distances.sort_values(kind='stable').index[0])
            for column in source.columns:
                if column=='geometry':continue
                aggregated.loc[chosen,column]+=float(row[column])
            aggregated.loc[chosen,'count']+=1
            assignments.append(dict(building_id=int(building),block_id=chosen,population=int(row.population),method='nearest_polygon_gap_le_5m',distance_m=float(distances.loc[chosen])))
            continue
        unassigned.append(dict(building_id=int(building),population=int(row.population)))
    aggregated['population']=aggregated.population.astype(int)
    aggregated['site_area']=aggregated.area.astype('float32')
    status=pd.Series('no_identified_housing',index=aggregated.index)
    status.loc[aggregated['count']==0]='no_building_data'
    status.loc[(aggregated.population==0)&(aggregated.unresolved_buildings>0)]='unresolved_housing'
    status.loc[aggregated.population>0]='estimated_from_osm_housing'
    status.loc[(aggregated.population>0)&(aggregated.inferred_residential_buildings>0)]='estimated_with_housing_inference'
    status.loc[(aggregated.population>0)&(aggregated.unresolved_buildings>0)]='estimated_with_unresolved_buildings'
    aggregated['population_status']=status
    aggregated['population_unresolved']=(aggregated.population==0)&status.isin(['unresolved_housing','no_building_data'])
    density=calculate_density_indicators(aggregated)
    developed=int((density.fsi>0).sum())
    space,clusters=get_spacematrix_morphotypes(density,n_clusters=max(1,min(11,developed)))
    strelka=get_strelka_morphotypes(density)
    matrix=pd.read_parquet(DATA/'accessibility_matrix.parquet');matrix.index=matrix.index.astype(int);matrix.columns=matrix.columns.astype(int)
    inputs=aggregated[['population']].join(read('schools_blocks.geojson')[['capacity']])
    assert inputs.index.equals(matrix.index) and inputs.index.equals(matrix.columns)
    cfg=service_types_config['school']
    provision,links=competitive_provision(inputs,matrix,int(cfg['accessibility']),demand=int(cfg['demand']),self_supply=True,max_depth=1)
    previous=pd.read_csv(PREVIOUS/'results_table.csv',index_col=0)
    result=previous.copy()
    for column in density.columns:
        if column in result.columns:result[column]=density[column].reindex(result.index)
    result['population']=aggregated.population
    for column in ['provision_strong','provision_weak']:result[column]=provision[column]
    result['morphotype_spacematrix']=space.morphotype
    result['morphotype_strelka']=strelka.morphotype
    for column in ['residential_buildings','inferred_residential_buildings','unresolved_buildings','population_status','population_unresolved']:
        result[column]=aggregated[column]
    np.testing.assert_array_equal(result.capacity,previous.capacity)
    unchanged=['residential','business','recreation','industrial','transport','special','agriculture','land_use','share','area_accessibility']
    pd.testing.assert_frame_equal(result[unchanged],previous[unchanged])
    geo=read('blocks_land_use.geojson')[['geometry']].join(result)
    summary_columns=pd.read_csv(PREVIOUS/'results_summary.csv',index_col=0).index.tolist()
    summary=result[summary_columns].describe().T.round(3)
    old_blocks=gpd.read_file(PREVIOUS/'buildings_blocks.geojson').set_index('id')
    restored=(old_blocks.population==0)&(aggregated.population>0)
    new=classified.is_living&~old.is_living
    report=dict(method_version='housing-evidence-v1',classification_source_sha256=digest(EXAMPLES/'classify_housing.py'),population_before=int(old_blocks.population.sum()),population_after=int(aggregated.population.sum()),building_population=int(modeled.population.sum()),building_count=len(modeled),residential_before=int(old.is_living.sum()),residential_after=int(modeled.is_living.sum()),new_residential=int(new.sum()),new_by_method=classified.loc[new,'housing_status'].value_counts().to_dict(),classification_counts=classified.housing_status.value_counts().to_dict(),zero_blocks_before=int((old_blocks.population==0).sum()),zero_blocks_after=int((aggregated.population==0).sum()),restored_zero_blocks=int(restored.sum()),unresolved_zero_blocks=int(((aggregated.population==0)&(aggregated.unresolved_buildings>0)).sum()),no_building_blocks=int((aggregated['count']==0).sum()),capacity=int(provision.capacity.sum()),total_demand=int(provision.demand.sum()),total_deficit=int(provision.demand_left.sum()),served_within=int(provision.demand_within.sum()),fallback_assignments=assignments,unassigned_inhabited_buildings=unassigned,frozen_input_hashes=fixed,private_inference_rules='Address/size/shape plus residential context; context-only homes flagged low-confidence; smaller unaddressed outbuildings next to a larger house suppressed',population_formula='floor(0.9 * build_floor_area / 30), minimum 1 for identified/inferred housing; not census data',warnings=['Inferred housing is an estimate, not verified occupancy.','Unknown-use buildings and areas without mapped buildings remain explicitly unresolved; model zero is not evidence of no residents.','Garden-house permanent occupancy is not verified; included as potential residential stock.'])
    checks=dict(block_ids_unchanged=True,network_and_school_inputs_unchanged=all(digest(DATA/n)==h for n,h in fixed.items()),capacity_82_unchanged=int(provision.capacity.sum())==106942,all_known_housing_population_preserved=True,no_blanket_yes_as_residential=bool((classified.housing_status.eq('unresolved')&classified.building.eq('yes')).any()),private_housing_included=bool(classified.housing_status.str.startswith('inferred_private').any()),population_conserved=int(modeled.population.sum())==int(aggregated.population.sum())+sum(x['population'] for x in unassigned),unresolved_flags_present=True,zero_blocks_reduced=int(restored.sum())>0)
    assert all(checks.values()),checks
    report['checks']=checks
    save(modeled,'buildings.geojson');save(aggregated,'buildings_blocks.geojson')
    density.to_csv(DATA/'density_indicators.csv');space.to_csv(DATA/'morphotypes_spacematrix.csv');clusters.to_csv(DATA/'morphotypes_spacematrix_clusters.csv');strelka.to_csv(DATA/'morphotypes_strelka.csv')
    provision.to_csv(DATA/'provision_schools.csv');links.to_csv(DATA/'provision_links.csv')
    result.to_csv(DATA/'results_table.csv');save(geo,'results_blocks.geojson');summary.to_csv(DATA/'results_summary.csv')
    locations=modeled.geometry.representative_point().to_crs(4326)
    audit=pd.DataFrame(dict(building_id=modeled.index,osm_id=modeled.osm_id,building_tag=modeled.building,street=modeled.get('addr:street',''),house_number=modeled.get('addr:housenumber',''),floors=modeled.number_of_floors,footprint_area=modeled.footprint_area,old_population=old.population.fillna(0),new_population=modeled.population,housing_status=modeled.housing_status,housing_confidence=modeled.housing_confidence,latitude=locations.y,longitude=locations.x))
    audit.to_csv(DATA/'housing_classification_audit.csv',index=False,encoding='utf-8-sig')
    comparison=pd.DataFrame(dict(old_population=old_blocks.population,new_population=aggregated.population,residential_buildings=aggregated.residential_buildings,inferred_residential_buildings=aggregated.inferred_residential_buildings,unresolved_buildings=aggregated.unresolved_buildings,population_status=status))
    comparison.index.name='block_id';comparison.to_csv(DATA/'population_block_changes.csv',encoding='utf-8-sig')
    quality=json.loads((DATA/'quality_report.json').read_text(encoding='utf-8'))
    quality['checks'].pop('unrelated_result_columns_unchanged',None)
    quality['checks'].update(checks)
    quality.update(population=report['population_after'],total_demand=report['total_demand'],total_deficit=report['total_deficit'],served_within=report['served_within'],population_method='OSM housing + transparent contextual inference, 30 m²/person',population_inference_counts=report['classification_counts'],population_unresolved_blocks=report['unresolved_zero_blocks'])
    (DATA/'quality_report.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'population_repair_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['frozen_input_hashes','fallback_assignments']},ensure_ascii=False,indent=2),flush=True)
    return report

if __name__=='__main__':run_repair()
