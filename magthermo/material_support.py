"""Extend the zero-field magnetite laboratory calculation to 550 °C."""
import hashlib
from magthermo.common import sha
import ast
import contextlib
import inspect
import textwrap
import numpy as np
import grain_energy as core
import thermal_executor as thermal
import census_refinement as repair

VERSION='ZERO_FIELD_LAB_MATERIAL_DOMAIN_V1'
MAXIMUM_C=550.


def widen_temperature_guards(function,variables):
    """Clone the verified function changing only named 20<=T<=350 guards."""
    raw=getattr(function,'__wrapped__',function)
    assert not raw.__code__.co_freevars,'DOMAIN_ADAPTER_DOES_NOT_ACCEPT_CLOSURES'
    source=textwrap.dedent(inspect.getsource(raw))
    tree=ast.parse(source)
    before=ast.dump(tree,include_attributes=False)
    changed=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and len(node.ops)==2 and
            all(isinstance(op,ast.LtE) for op in node.ops) and
            isinstance(node.left,ast.Constant) and node.left.value==20 and
            isinstance(node.comparators[0],ast.Name) and node.comparators[0].id in variables and
            isinstance(node.comparators[1],ast.Constant) and node.comparators[1].value==350):
            changed.append(node.comparators[0].id)
            node.comparators[1].value=MAXIMUM_C
    assert sorted(changed)==sorted(variables),'TEMPERATURE_GUARD_PATTERN_CHANGED'
    after=ast.dump(tree,include_attributes=False)
    namespace=dict(raw.__globals__)
    exec(compile(tree,'<'+VERSION+':'+raw.__module__+'.'+raw.__name__+'>','exec'),namespace)
    replacement=namespace[raw.__name__]
    metadata=dict(module=raw.__module__,function=raw.__qualname__,temperature_variables=changed,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        domain_adjusted_AST_sha256=hashlib.sha256(after.encode()).hexdigest(),
        changed_only_upper_temperature_guard_C=[350.,MAXIMUM_C],
        material_arithmetic_and_event_transport_unchanged=True)
    return replacement,metadata


@contextlib.contextmanager
def laboratory_support(enabled=True):
    if not enabled:
        yield dict(active=False,legacy_history_keeps_original_domain=True)
        return
    if sha(core.legacy.source.__file__) != '353101e74bf5780e849b5ad383996f8ef001716d0d1e22e40d4866d2d28bddab':
        raise ValueError('Unexpected SDCC energy module for laboratory temperatures.')
    legacy=core.legacy
    old_material=legacy.material
    old_model=core.PointModel
    old_event=thermal.Executor._event_reference
    old_repair=repair.recover_anchor_censuses
    material,material_info=widen_temperature_guards(old_material,['T'])
    event,event_info=widen_temperature_guards(old_event,['Tl0','Tr0'])
    recover,recover_info=widen_temperature_guards(old_repair,['Tl','Tr'])
    class LaboratoryPointModel(old_model):
        def __init__(self,s,c,T,B=(0.,0.,0.),TM=0):
            temperature=float(T)
            if not 20.<=temperature<=MAXIMUM_C:
                raise core.Unresolved('MATERIAL_T_OUTSIDE_20_550C')
            if TM!=0:
                raise core.Unresolved('MATERIAL_ONLY_TM00')
            if temperature>350. and not np.array_equal(np.asarray(B),np.zeros(3)):
                raise core.Unresolved('HIGH_T_REQUIRES_ZERO_LABORATORY_FIELD')
            super().__init__(s,c,temperature,B,TM)
    metadata=dict(version=VERSION,active=True,temperature_range_C=[20.,MAXIMUM_C],
        event_temperature_range_C=[40.,300.],above_350_only_exact_zero_field=True,
        guard_adapters=[material_info,event_info,recover_info],
        same_archived_get_material_parms=True,new_fit_or_physical_law=False,
        empirical_high_temperature_SD_validity_not_established=True)
    try:
        legacy.material=material
        core.PointModel=LaboratoryPointModel
        thermal.Executor._event_reference=event
        repair.recover_anchor_censuses=recover
        yield metadata
    finally:
        legacy.material=old_material
        core.PointModel=old_model
        thermal.Executor._event_reference=old_event
        repair.recover_anchor_censuses=old_repair
        old_material.cache_clear()
