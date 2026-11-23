"""Complete positive-area crystal cells; children replace complete parents."""
from pathlib import Path
from magthermo.common import read
import math
import numpy as np

AREA = math.pi/12


def area(vertices):
    a,b,c=np.asarray(vertices,dtype=np.float64)
    return 2*math.atan2(abs(float(np.linalg.det(np.array([a,b,c])))),1+a@b+b@c+c@a)


def initial(level=2):
    if level not in (1,2):
        raise ValueError('INITIAL_LEVEL_1_OR_2')
    source=read(Path(__file__).resolve().parents[1] / 'config/crystal_orientations.json')
    rows=[dict(v) for v in source['nodes'] if v['level']==level]
    assert len(rows)==4**level
    validate(rows)
    return rows


def children(parent):
    a,b,c=np.asarray(parent['vertices'],dtype=np.float64)
    ab,ac,bc=[v/np.linalg.norm(v) for v in (a+b,a+c,b+c)]
    vertices=[(a,ab,ac),(ab,b,bc),(ac,bc,c),(ab,bc,ac)]
    result=[]
    for i,triangle in enumerate(vertices):
        center=sum(triangle)
        center=center/np.linalg.norm(center)
        cell_id=parent['cell_id']+str(i)
        cell_area=area(triangle)
        result.append(dict(node_id='CRYSTAL_'+cell_id,level=parent['level']+1,
            cell_id=cell_id,parent_cell_id=parent['cell_id'],vertices=np.array(triangle).tolist(),
            c_x=float(center[0]),c_y=float(center[1]),c_z=float(center[2]),
            c_hex=[float(v).hex() for v in center],area_sr=cell_area,weight=cell_area/AREA))
    assert abs(math.fsum(v['area_sr'] for v in result)-parent['area_sr'])<1e-12
    return result


def refine(rule, selected_cells):
    selected=set(selected_cells)
    assert selected.issubset({v['cell_id'] for v in rule})
    result=[w for v in rule for w in (children(v) if v['cell_id'] in selected else [dict(v)])]
    validate(result)
    return result


def validate(rule):
    assert len({v['cell_id'] for v in rule})==len(rule)
    assert all(v['weight']>0 and v['area_sr']>0 for v in rule)
    assert abs(math.fsum(v['area_sr'] for v in rule)-AREA)<=1e-12
    assert abs(math.fsum(v['weight'] for v in rule)-1.)<=1e-12
    for v in rule:
        assert abs(area(v['vertices'])-v['area_sr'])<1e-12
        c=np.array([v[k] for k in ('c_x','c_y','c_z')])
        assert abs(c@c-1.)<1e-12 and c[0]>=c[1]>=c[2]>=0.
    return True
