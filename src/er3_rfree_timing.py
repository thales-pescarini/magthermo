"""Exclusive timing of published R-independent numerical work, without physics changes."""
import contextlib
import functools
import time
import numpy as np
import r1_landscape as landscape
import g1_geometry as geometry

VERSION='ER3_RFREE_PUBLISHED_PRODUCER_UNION_V1'


class RfreeTiming:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rfree_intervals=[]
        self.rfree_lab_verified_this_case=None
        self._rfree_pending={}
        self._rfree_failed_pending={}

    def publish(self,bank,key,value):
        span=self._rfree_pending.pop((bank,id(value)),None)
        super().publish(bank,key,value)
        if span is not None and key in self.banks[bank]:
            self.rfree_intervals.append(dict(span,cache_kind=bank,cache_key=key,
                publication='complete producer accepted by unchanged E-R2 guard'))

    def publish_failed_attempt(self,kind,key,exc):
        span=self._rfree_failed_pending.pop((kind,id(exc)),None)
        super().publish_failed_attempt(kind,key,exc)
        if span is not None and key in self.failed_attempts[kind]:
            self.rfree_intervals.append(dict(span,cache_kind=kind,cache_key=key,
                publication='same bounded failed numerical attempt only; no root or physical absence inferred'))

    @contextlib.contextmanager
    def instrument_producers(self):
        """Enter BEFORE E-R2 wraps producers, so only actual misses are timed.

        Direct zero-field enumeration has only R-independent physical inputs.
        Other zero-field producers are credited only in the laboratory after
        the caller proves canonical zero-field entry and no prior zero events.
        Raw nested spans are kept; the consumer takes their UNION, never sum.
        """
        originals={}
        self._rfree_pending={};self._rfree_failed_pending={}
        def timed(kind,producer):
            @functools.wraps(producer)
            def call(pm,*args,**kwargs):
                zero=np.array_equal(np.asarray(pm.B),np.zeros(3))
                lab=(self.rfree_lab_verified_this_case is True and str(self.current_phase).startswith('lab_'))
                independent=bool(zero and (kind=='enumerate_land' or lab))
                start=time.monotonic()
                try:
                    value=producer(pm,*args,**kwargs)
                except Exception as exc:
                    if independent:
                        self._rfree_failed_pending[(kind,id(exc))]=dict(start_s=start,end_s=time.monotonic(),
                            dependency_proof='zero-field laboratory inputs verified R-independent' if lab else 'direct zero-field parameter-only enumeration')
                    raise
                if independent:
                    self._rfree_pending[(kind,id(value))]=dict(start_s=start,end_s=time.monotonic(),
                        dependency_proof='zero-field laboratory inputs verified R-independent' if lab else 'direct zero-field parameter-only enumeration')
                return value
            return call
        try:
            for module,name in [(landscape,'enumerate_land'),(landscape,'continuous_points'),(landscape,'flow'),(geometry,'refine_kkt')]:
                original=getattr(module,name);originals[(module,name)]=original
                setattr(module,name,timed(name,original))
            yield
        finally:
            for (module,name),original in originals.items():setattr(module,name,original)
            self._rfree_pending={};self._rfree_failed_pending={}
