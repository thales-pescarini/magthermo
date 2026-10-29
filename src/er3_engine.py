"""Prospective common thermal mesh; unchanged exact field and transport."""
import hashlib
import functools
import math
from pathlib import Path
import time
import numpy as np
import er2_engine as base

ALGORITHM_VERSION = 'ER3_COMMON_MESH_2p5_EXACT_FIELD_V1'


def mesh_nodes(T0, T1, step=2.5, origin=20.):
    """Global interior nodes plus exact endpoints, in chronological order."""
    if not all(math.isfinite(float(v)) for v in [T0,T1,step,origin]) or step <= 0:
        raise ValueError('INVALID_COMMON_THERMAL_MESH')
    if T0 == T1:
        return [float(T0)]
    lo,hi=sorted((float(T0),float(T1)))
    first=math.floor((lo-origin)/step)+1
    last=math.ceil((hi-origin)/step)-1
    interior=[origin+k*step for k in range(first,last+1)]
    nodes=[float(T0)]+(interior if T1>T0 else interior[::-1])+[float(T1)]
    if any(not min(T0,T1) <= t <= max(T0,T1) for t in nodes):
        raise ValueError('COMMON_MESH_OUTSIDE_SEGMENT')
    return nodes


@functools.lru_cache(maxsize=1)
def source_id():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SharedCaches(base.SharedCaches):
    def physical_key(self,pm,kind,args,kwargs):
        return base.digest(dict(base_key=super().physical_key(pm,kind,args,kwargs),
                                er3_algorithm=ALGORITHM_VERSION,er3_source_sha256=source_id()))


class GeneralTape(base.GeneralTape):
    def prefix_geometry_key(self,preparation_graph):
        return base.digest(dict(base_key=super().prefix_geometry_key(preparation_graph),
                                er3_algorithm=ALGORITHM_VERSION,er3_source_sha256=source_id()))

    def laboratory_geometry_key(self,land,p,protocol):
        return base.digest(dict(base_key=super().laboratory_geometry_key(land,p,protocol),
                                er3_algorithm=ALGORITHM_VERSION,er3_source_sha256=source_id()))

    def ramp(self,p,land,T0,B0,T1,B1,seconds,label):
        if T1 == T0:
            # Keep the four contracted field substeps and exact vector history.
            return super().ramp(p,land,T0,B0,T1,B1,seconds,label)
        spec=self.level.get('common_thermal_mesh')
        if spec != dict(origin_C=20.,step_C=2.5):
            raise ValueError('COMMON_MESH_REQUIRES_EXPLICIT_FROZEN_RESOLUTION')
        nodes=mesh_nodes(T0,T1,spec['step_C'],spec['origin_C'])
        started=time.monotonic()
        previous_phase=self.shared.current_phase
        self.shared.current_phase=label
        self.shared.er2_counts['ramps:'+label]+=1
        self.shared.er2_counts['common_mesh_thermal_segments']+=len(nodes)-1
        try:
            B0=np.asarray(B0);B1=np.asarray(B1)
            constant_field=np.array_equal(B0,B1)
            for k,(ta,tb) in enumerate(zip(nodes[:-1],nodes[1:])):
                a=(ta-T0)/(T1-T0);b=(tb-T0)/(T1-T0)
                ba=B0 if constant_field else B0+a*(B1-B0)
                bb=B1 if constant_field else B0+b*(B1-B0)
                delta_seconds=seconds*abs(tb-ta)/abs(T1-T0)
                p,land=self.advance(p,land,ta,ba,tb,bb,delta_seconds,label)
                if k%16==0:self.snapshot(p,land,label,k,len(nodes)-1)
            self.trace.append(dict(segment=label,T_start=T0,T_end=T1,seconds=seconds,
                                   substeps=len(nodes)-1,global_thermal_mesh=spec,state=p,states=len(p)))
            self.snapshot(p,land,label,len(nodes)-1,len(nodes)-1)
            return p,land
        finally:
            self.shared.current_phase=previous_phase
            self.cost['ramps']+=time.monotonic()-started


prepare_response=base.prepare_response
evaluate_duration=base.evaluate_duration
