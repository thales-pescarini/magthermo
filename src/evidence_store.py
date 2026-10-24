"""Self-contained content-addressed evidence, not a production model bank."""
from pathlib import Path
import json,hashlib,os
import numpy as np
import e1_core as core

class Store:
    def __init__(self,root):self.root=Path(root);self.folder=self.root/'results/objects';self.folder.mkdir(parents=True,exist_ok=True)
    def put(self,kind,value):
        raw=(json.dumps(core.plain(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
        digest=hashlib.sha256(raw).hexdigest();p=self.folder/(digest+'.json')
        if not p.exists():p.write_bytes(raw)
        return {'$ref':str(p.relative_to(self.root)),'sha256':digest,'kind':kind}
    def encode(self,value,key=None):
        if isinstance(value,dict):
            if '$ref' in value:return value
            kind='graph' if {'identity','points','minima','channels','status'}<=set(value) else 'event' if {'T_segment','B_segment_T','status'}<=set(value) else None
            result={str(k):self.encode(v,k) for k,v in value.items()}
            return self.put(kind,result) if kind else result
        if key=='map' and isinstance(value,(np.ndarray,list)):
            return self.put('probability_map',value)
        if isinstance(value,(list,tuple)):return [self.encode(v) for v in value]
        return core.plain(value)
    def save(self,path,value):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(self.encode(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def resolve(value,root,verify=True):
    if isinstance(value,dict) and '$ref' in value:
        p=Path(root)/value['$ref'];raw=p.read_bytes()
        if verify and hashlib.sha256(raw).hexdigest()!=value['sha256']:raise ValueError('EVIDENCE_REFERENCE_HASH_MISMATCH')
        return resolve(json.loads(raw),root,verify)
    if isinstance(value,dict):return {k:resolve(v,root,verify) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v,root,verify) for v in value]
    return value
