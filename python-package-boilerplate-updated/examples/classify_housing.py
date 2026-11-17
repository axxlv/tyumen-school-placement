"""Transparent housing classification; building=yes never means nonresidential.

Only contextual rules infer housing. Existing tagged residential buildings are
retained; explicit nonresidential/service/garage/construction objects excluded.
Unresolved buildings retain an explicit unknown status and are not silently
described as uninhabited.
"""
import json
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

RESIDENTIAL={'residential','house','apartments','detached','terrace','dormitory','semidetached_house','semi-detached','bungalow','cabin','houseboat','allotment_house'}
UNKNOWN={'','yes','nan','none'}
NONRES_WORDS=re.compile(r'гараж|паркинг|автосервис|шин[оа]монтаж|торгов|магазин|гипермаркет|супермаркет|бизнес|офис|администрац|поликлин|больниц|медицин|детский сад|детскийсад|школ|гимназ|лицей|университет|институт|колледж|гостиниц|отель|храм|церков|склад|котельн|подстанц|\bбанк\b|теплиц',re.I)

def strings(frame,name):
    if name not in frame:return pd.Series('',index=frame.index)
    return frame[name].fillna('').astype(str).str.strip().str.casefold().replace({'nan':'','none':''})

def classify(raw,zones,school_points=None):
    frame=raw.copy()
    tags=strings(frame,'building')
    token_sets=tags.map(lambda text:set(text.split(';')))
    explicit=token_sets.map(lambda parts:bool(parts&RESIDENTIAL))
    unknown=tags.isin(UNKNOWN)
    flats=pd.to_numeric(frame.get('building:flats',pd.Series('',index=frame.index)),errors='coerce').fillna(0)>0
    blocked_type=~unknown&~explicit
    abandoned=strings(frame,'abandoned').isin({'yes','true','1'})
    # Explicit non-residential use overrides inference, not an apartments tag
    # with a shop on the ground floor.
    facility=pd.Series(False,index=frame.index)
    for column in ['amenity','office','shop','tourism','leisure','building:use']:
        facility|=strings(frame,column).ne('')&~strings(frame,column).isin({'no','false','residential','apartments','house'})
    facility|=strings(frame,'name').str.contains(NONRES_WORDS,na=False)
    facility|=strings(frame,'building:part').eq('yes')
    # Use only actual polygon membership, not the majority land use of a block.
    points=frame[['geometry']].copy();points.geometry=points.geometry.representative_point()
    join=gpd.sjoin(points,zones[['functional_zone','geometry']].to_crs(frame.crs),how='left',predicate='intersects')
    sets=join.groupby(level=0).functional_zone.agg(lambda s:set(s.dropna().astype(str)))
    residential_zone=sets.map(lambda s:'residential' in s).reindex(frame.index,fill_value=False)
    garden_zone=sets.map(lambda s:'allotments' in s).reindex(frame.index,fill_value=False)
    incompatible_zone=sets.map(lambda s:bool(s&{'industrial','commercial','retail','garages','railway','military','cemetery','education','parking'})).reindex(frame.index,fill_value=False)
    contextual_zone=(residential_zone|garden_zone)&~incompatible_zone
    xy=np.column_stack([points.geometry.x,points.geometry.y])
    confirmed=explicit|flats
    nearest=cKDTree(xy[confirmed]).query(xy,k=1)[0] if confirmed.any() else np.full(len(frame),np.inf)
    area=frame.geometry.area
    floors=pd.to_numeric(frame.get('building:levels',pd.Series('',index=frame.index)),errors='coerce')
    number=strings(frame,'addr:housenumber')
    address=(number.ne('')&(strings(frame,'addr:street').ne('')|strings(frame,'addr:place').ne('')))|strings(frame,'addr:full').ne('')
    compact=4*np.pi*area/frame.geometry.length.pow(2)
    rectangles=frame.geometry.minimum_rotated_rectangle()
    short_sides=[];aspects=[]
    for rectangle in rectangles:
        coords=np.asarray(rectangle.exterior.coords)[:4]
        lengths=np.linalg.norm(np.roll(coords,-1,axis=0)-coords,axis=1)
        short=max(float(lengths.min()),.01)
        short_sides.append(short);aspects.append(float(lengths.max())/short)
    residential_shape=pd.Series(short_sides,index=frame.index).ge(3.5)&pd.Series(aspects,index=frame.index).le(5)
    # A CSV point strictly inside a generic building is additional non-housing
    # evidence. Existing explicit housing labels are preserved for review.
    school_site=pd.Series(False,index=frame.index)
    if school_points is not None:
        matched=gpd.sjoin(frame[['geometry']],school_points[['geometry']].to_crs(frame.crs),how='inner',predicate='contains')
        school_site.loc[matched.index.unique()]=True
    excluded=blocked_type|abandoned|((facility|school_site)&~confirmed)
    candidate=unknown&~excluded
    apartment=candidate&floors.ge(4)&area.between(80,10000)&compact.ge(.08)&(contextual_zone|((nearest<=100)&address&~incompatible_zone))
    private_address=candidate&~apartment&floors.fillna(1).between(1,3)&area.between(25,450)&compact.ge(.12)&residential_shape&address&(contextual_zone|((nearest<=70)&~incompatible_zone))
    # Unaddressed outbuildings next to a larger house must not be populated.
    possible_main=candidate&floors.fillna(1).between(1,3)&area.between(45,300)&compact.ge(.25)&residential_shape&contextual_zone
    anchors=(confirmed|apartment|private_address|possible_main).to_numpy()
    dominated=np.zeros(len(frame),dtype=bool)
    if anchors.any():
        positions=np.flatnonzero(anchors);tree=cKDTree(xy[anchors]);neighbors=tree.query_ball_point(xy,r=25)
        sizes=area.to_numpy()
        for i,neighbors_i in enumerate(neighbors):
            dominated[i]=any(positions[k]!=i and sizes[positions[k]]>=sizes[i]*1.5 for k in neighbors_i)
    private_context=possible_main&~apartment&~private_address&~address&((nearest<=100)|garden_zone)&~dominated
    result=pd.Series('unresolved',index=frame.index)
    result.loc[excluded]='non_residential'
    result.loc[private_context]='inferred_private_context'
    result.loc[private_address]='inferred_private_address'
    result.loc[apartment]='inferred_multistorey'
    result.loc[flats&~blocked_type&~abandoned]='osm_flats'
    result.loc[explicit&~abandoned]='osm_residential'
    confidence=result.map({'osm_residential':'выше: явный жилой тип OSM','osm_flats':'выше: число квартир OSM','inferred_multistorey':'средняя: этажность и жилое окружение','inferred_private_address':'средняя: адрес, размер и жилое окружение','inferred_private_context':'ниже: размер и жилое окружение','non_residential':'нежилое по явным признакам','unresolved':'не установлено'})
    frame['housing_status']=result
    frame['housing_confidence']=confidence
    frame['is_living']=result.isin({'osm_residential','osm_flats','inferred_multistorey','inferred_private_address','inferred_private_context'})
    frame['housing_inferred']=result.str.startswith('inferred_')
    frame['housing_unresolved']=result.eq('unresolved')
    frame['housing_residential_zone']=residential_zone
    frame['housing_garden_zone']=garden_zone
    frame['housing_nearest_confirmed_m']=nearest
    frame['housing_has_address']=address
    frame['housing_dominated_by_larger_house']=dominated
    return frame
