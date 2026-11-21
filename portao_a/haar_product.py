"""Explicit numerical product rules for the same normalized continuous Haar law.

The frozen adapter and its R12/R48/R192 rules remain unchanged. New rules
require both a source-bound unit contract and a verified worker opt-in.
Manufactured moment tests do not certify thermomagnetic quadrature accuracy.
"""
import numpy as np
from portao_a.common import sha, digest

VERSION = 'PORTAO_A_HAAR_PRODUCT_RULE_V1'
PROFILES = {100: (5, 4, 5), 150: (5, 6, 5)}
ARCHIVED_COUNTS = (12, 48, 192)


def product(count):
    if count not in PROFILES:
        raise ValueError('HAAR_PRODUCT_PROFILE_NOT_DECLARED')
    na, nz, ng = PROFILES[count]
    zs, ws = np.polynomial.legendre.leggauss(nz)

    def rz(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    rows = []
    for ia in range(na):
        for iz, (z, wz) in enumerate(zip(zs, ws)):
            beta = np.arccos(z)
            ry = np.array([[z, 0., np.sin(beta)], [0., 1., 0.],
                           [-np.sin(beta), 0., z]])
            for ig in range(ng):
                rows.append(dict(id=f'PA_PRODUCT_R{count}_{len(rows):03}',
                    R_fs=(rz(2*np.pi*ia/na) @ ry @ rz(2*np.pi*ig/ng)).tolist(),
                    number_fraction=float(wz/(2*na*ng)),
                    euler_haar=dict(alpha_index=ia, alpha_count=na,
                        cos_beta_index=iz, cos_beta_count=nz,
                        gamma_index=ig, gamma_count=ng)))
    return rows


def contract(count):
    rows = product(count)
    return dict(version=VERSION, count=count, product_counts=list(PROFILES[count]),
        rotations_and_weights_sha256=digest([
            dict(R_fs=row['R_fs'], number_fraction=row['number_fraction']) for row in rows]),
        measure='normalized_continuous_Haar_SO3',
        population_or_physics_changed=False)


def validate_spec(spec):
    count = spec.get('R_count', 12)
    extensions = spec.get('numeric_extensions', {})
    if count in PROFILES:
        if extensions.get(VERSION) != sha(__file__):
            raise ValueError('HAAR_PRODUCT_EXPLICIT_CURRENT_SOURCE_REQUIRED')
        if spec.get('haar_product_rule') != contract(count):
            raise ValueError('HAAR_PRODUCT_EXPLICIT_RULE_CONTRACT_REQUIRED')
    elif count in ARCHIVED_COUNTS:
        if VERSION in extensions or 'haar_product_rule' in spec:
            raise ValueError('HAAR_PRODUCT_NOT_AN_ARCHIVED_RULE_OVERRIDE')
    else:
        raise ValueError('HAAR_PRODUCT_PROFILE_NOT_DECLARED')
    if not isinstance(spec['rotation_index'], int) or not 0 <= spec['rotation_index'] < count:
        raise ValueError('HAAR_ROTATION_INDEX_OUTSIDE_DECLARED_RULE')


def resolve_rule(count, extensions=None):
    extensions = extensions or {}
    if count in PROFILES:
        if extensions.get(VERSION) != sha(__file__):
            raise ValueError('HAAR_PRODUCT_EXPLICIT_CURRENT_SOURCE_REQUIRED')
        return product(count)
    if VERSION in extensions:
        raise ValueError('HAAR_PRODUCT_NOT_AN_ARCHIVED_RULE_OVERRIDE')
    if count not in ARCHIVED_COUNTS:
        raise ValueError('HAAR_PRODUCT_PROFILE_NOT_DECLARED')
    return archived_rule(count)


def archived_rule(n):
    if n in (12,48):
        from e1_core import haar_orientations
        return haar_orientations(n)
    assert n == 192
    na,nz,ng = 6,4,8
    zs,ws = np.polynomial.legendre.leggauss(nz)
    def rz(a):
        c,s = np.cos(a),np.sin(a)
        return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    rows = []
    for ia in range(na):
        for iz,(z,wz) in enumerate(zip(zs,ws)):
            beta = np.arccos(z)
            ry = np.array([[z,0.,np.sin(beta)],[0.,1.,0.],[-np.sin(beta),0.,z]])
            for ig in range(ng):
                rows.append(dict(id=f'F1_R192_{len(rows):03}',
                    R_fs=(rz(2*np.pi*ia/na) @ ry @ rz(2*np.pi*ig/ng)).tolist(),
                    number_fraction=float(wz/(2*na*ng)),
                    euler_haar=dict(alpha_index=ia,alpha_count=na,cos_beta_index=iz,
                                    cos_beta_count=nz,gamma_index=ig,gamma_count=ng)))
    assert len(rows)==192 and abs(sum(r['number_fraction'] for r in rows)-1.) < 1e-14
    return rows
