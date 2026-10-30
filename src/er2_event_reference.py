"""R2-scoped full reference event producer with one guarded owner-authorized column hook."""
from thermal_executor import time, np, core, geom, ls, math, copy, brentq, root

def event_reference_er2(self,T0,B0,T1,B1,old):
    t=time.monotonic();T0=float(T0);T1=float(T1);B0=np.asarray(B0);B1=np.asarray(B1)
    temperature=abs(T1-T0)>0;tol=1e-6/abs(T1-T0) if temperature else 1e-8/max(float(np.linalg.norm((B1-B0)/5e-5)),1e-300)
    event_key=(float(T0),tuple(B0),float(T1),tuple(B1))
    if event_key in self.event_solutions:
        lo,hi,left,right,P,template=self.event_solutions[event_key]
        rec=copy.deepcopy(template);rec['graph_verification_reused_at_identical_segment']=True
        rec.pop('occupation_before_map',None);rec.pop('occupation_after_map',None);self.events.append(rec)
        return lo,hi,left,right,P.copy()
    hint=self.pending_critical_hint;self.pending_critical_hint=None
    early=dict(T_segment=[T0,T1],B_segment_T=[B0,B1],status='NUMERICAL_UNRESOLVED',fraction_bracket=[0.,1.],reason='CRITICAL_LOCATION_IN_PROGRESS')
    self.events.append(early)
    if self.checkpoint:self.checkpoint(dict(events=self.events,checkpoint=getattr(self,'current',None)))
    lo=0.;hi=1.;left=old;right=None;history=[]
    mirror_critical=None
    if hint is None and np.array_equal(B0,np.zeros(3)) and np.array_equal(B1,np.zeros(3)):
        mirror_critical=geom.locate_mirror_critical(lambda f:self.pm(T0+f*(T1-T0),B0),T0,T1)
    for iteration in range(64):
        if hint is not None or mirror_critical is not None:break
        if hi-lo<=tol:break
        mid=(lo+hi)/2;T=T0+mid*(T1-T0);B=B0+mid*(B1-B0);pm=self.pm(T,B)
        pts,valid=ls.continuous_points(pm,left['points'])
        eig=min((abs(e) for p in pts for e in p['eig']),default=math.inf)
        history.append(dict(fraction=mid,same_stationary_branches=valid,minimum_absolute_curvature=eig,
            geometry_is_current_verified_roots=valid))
        if valid:
            try:left=ls.from_points(pm,pts,left);lo=mid
            except core.Unresolved:hi=mid
        else:hi=mid
    if hint is not None:
        fc=(hint['T_C']-T0)/(T1-T0);lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
    elif mirror_critical is not None:
        fc=mirror_critical['fraction'];lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
    if hi-lo>tol:raise core.Unresolved('EVENT_BISECTION_LIMIT')
    # Resolve the critical root itself, separately from the Morse sides.
    # The diagonal zero-field axial branch exists analytically on both
    # sides; other folds use the stationary KKT plus det(H_tangent)=0.
    critical=dict(hint['critical'],fraction=fc,reversed_known_critical_location=True) if hint is not None else mirror_critical
    if critical is None and np.linalg.norm(B0)==0 and np.linalg.norm(B1)==0 and np.count_nonzero(abs(self.c)>1e-14)==1:
        axis=int(np.argmax(abs(self.c)));u=np.eye(3)[axis]
        def curvature(f):
            pm=self.pm(T0+f*(T1-T0),B0)
            return 2*geom.axial_A(pm) if geom.applicable(pm) else float(np.linalg.eigvalsh(ls.gate.riem(pm,u)[1])[0])
        ca,cb=curvature(0.),curvature(1.)
        if ca*cb<0:
            fc,info=brentq(curvature,0.,1.,xtol=max(1e-14,tol*.01),maxiter=64,full_output=True)
            critical=dict(kind='analytic_axial_branch_eigenvalue_root',fraction=fc,root_direction=u,
                eigenvalue_at_root=curvature(fc),initial_signed_eigenvalues=[ca,cb],iterations=info.iterations)
    if critical is None:
        candidate=min(left['points'],key=lambda p:min(abs(np.array(p['eig']))));u=np.array(candidate['m']);sc=ls.scale(self.pm(T0,B0))
        def F(x):
            pm=self.pm(T0+x[4]*(T1-T0),B0+x[4]*(B1-B0));v=x[:3];vn=v/np.linalg.norm(v)
            g=pm.grad_np(v)/sc-x[3]*v;H=ls.gate.riem(pm,vn)[1]/sc
            return np.r_[g,.5*(v@v-1),np.linalg.det(H)]
        pmc=self.pm(T0+lo*(T1-T0),B0+lo*(B1-B0));initial=np.r_[u,u@pmc.grad_np(u)/sc,(lo+hi)/2]
        sol=root(F,initial,tol=1e-11,options={'maxfev':160});res=float(np.linalg.norm(F(sol.x)))
        fc=float(sol.x[4]);pmc=self.pm(T0+fc*(T1-T0),B0+fc*(B1-B0));eig=np.linalg.eigvalsh(ls.gate.riem(pmc,sol.x[:3]/np.linalg.norm(sol.x[:3]))[1])
        if res<1e-10 and 0<fc<1 and lo-4*tol<fc<hi+4*tol and min(abs(eig))<1e-5:
            critical=dict(kind='fold_stationary_KKT',fraction=fc,root_direction=sol.x[:3]/np.linalg.norm(sol.x[:3]),KKT_residual=res,eigenvalues=eig,scipy_success=bool(sol.success))
    record=dict(T_segment=[T0,T1],B_segment_T=[B0,B1],fraction_bracket=[lo,hi],
        history=history,critical_root=critical,status='NUMERICAL_UNRESOLVED',anchors=[],
        transactional_input_restored=True,physical_transport_version=self.er2_transport_version)
    early.clear();early.update(record);record=early
    def persist():
        self.diagnostic=dict(kind='EVENT_IN_PROGRESS',event=record)
        if self.checkpoint:self.checkpoint(dict(events=self.events,diagnostic=self.diagnostic,checkpoint=getattr(self,'current',None)))
    persist()
    if critical is None:
        record['reason']='CRITICAL_ROOT_NOT_RESOLVED';persist();raise core.Unresolved(record['reason'])
    fc=critical['fraction'];lo=max(0.,fc-tol*.49);hi=min(1.,fc+tol*.49)
    record.update(fraction_bracket=[lo,hi],temperature_bracket_C=[T0+lo*(T1-T0),T0+hi*(T1-T0)],field_fraction_width=hi-lo)
    offsets=[.1,.02,.004] if temperature else [.01,.002,.0004]
    retry_offsets=getattr(self,'_retry_lateral_anchor_offsets',None)
    if retry_offsets is not None:offsets=retry_offsets
    anchor_left=None;anchor_right=None
    for offset in offsets:
        delta=offset/abs(T1-T0) if temperature else offset/max(np.linalg.norm((B1-B0)/5e-5),1e-300)
        fl,fr=fc-delta,fc+delta
        al=dict(offset=offset,fractions=[fl,fr],status='RUNNING');record['anchors'].append(al);persist()
        if not 0<fl<fr<1:
            al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_ORIGINAL_SEGMENT');persist();continue
        Tl0=T0+fl*(T1-T0);Tr0=T0+fr*(T1-T0);Bl0=B0+fl*(B1-B0);Br0=B0+fr*(B1-B0)
        if not (20<=Tl0<=350 and 20<=Tr0<=350) or max(np.linalg.norm(Bl0),np.linalg.norm(Br0))>5e-5*(1+1e-12):
            al.update(status='NUMERICAL_UNRESOLVED',reason='ANCHOR_OUTSIDE_PHYSICAL_RAMP_DOMAIN');persist();continue
        anchor_left=ls.enumerate_land(self.pm(Tl0,Bl0),self.census_levels(Tl0,Bl0));al['left_graph']=anchor_left;persist()
        anchor_right=ls.enumerate_land(self.pm(Tr0,Br0),self.census_levels(Tr0,Br0));al['right_graph']=anchor_right;persist()
        continued,consistent=self.stationary_points(Tl0,Bl0,old)
        same=consistent and ls.gate.same_census(dict(points=continued,counts={k:sum(p['index']==k for p in continued) for k in range(3)}),dict(points=anchor_left.get('points',[]),counts={k:sum(p['index']==k for p in anchor_left.get('points',[])) for k in range(3)}))
        al['continued_old_roots_agree_with_full_left_census']=bool(same)
        if anchor_left['status']=='COMPUTED_VALID' and anchor_right['status']=='COMPUTED_VALID' and same and anchor_left['counts']!=anchor_right['counts']:
            al['status']='COMPUTED_VALID';break
        al.update(status='NUMERICAL_UNRESOLVED',reason='LATERAL_CENSUS_CONNECTION_OR_TOPOLOGY_NOT_RESOLVED');persist()
    else:
        record['reason']='FINITE_LATERAL_ANCHORS_EXHAUSTED';persist()
        from a2_census_repair import recover_anchor_censuses
        repaired=recover_anchor_censuses(self,T0,B0,T1,B1,old,offsets,record,persist)
        if repaired is None:raise core.Unresolved(record['reason'])
        anchor_left,anchor_right,fl,fr,offset,Tl0,Tr0,Bl0,Br0=repaired
        if self.pm(Tl0,Bl0).identity()!=anchor_left['identity'] or self.pm(Tr0,Br0).identity()!=anchor_right['identity']:
            raise core.Unresolved('A2_SELECTED_ANCHOR_MODEL_IDENTITY_MISMATCH')
        record['A2_selected_anchor_model_identity_verified']=True
        record.pop('reason',None)
    record['selected_anchor_offset']=offset
    def continue_side(anchor,f):
        # A finite geometric continuation to the requested one-sided limit.
        af=fl if anchor is anchor_left else fr;land=anchor;checks=[]
        distance=abs(af-fc);target=abs(f-fc);side=-1 if af<fc else 1;fractions=[]
        for j in range(23):
            if distance<=2*target:break
            distance/=2;fractions.append(fc+side*distance)
        if distance>2*target:raise core.Unresolved('B2_LATERAL_24_STEP_CAP')
        fractions.append(f)
        for frac in fractions:
            pm=self.pm(T0+frac*(T1-T0),B0+frac*(B1-B0));pts,ok=ls.continuous_points(pm,land['points'])
            checks.append(dict(fraction=frac,valid=ok,counts=[sum(p['index']==k for p in pts) for k in range(3)] if ok else None,
                maximum_gradient_residual=max(p['gradient_residual'] for p in pts) if ok else None,minimum_absolute_curvature=min(abs(v) for p in pts for v in p['eig']) if ok else None,
                minimum_point_separation=min(np.linalg.norm(p['m']-q['m']) for j,p in enumerate(pts) for q in pts[:j]) if ok else None))
            record['lateral_continuation_in_progress']=checks;persist()
            if not ok:raise core.Unresolved('LATERAL_ROOT_CONTINUATION_UNRESOLVED')
            land=ls.from_points(pm,pts,land)
        return land,checks
    try:
        left,lchecks=continue_side(anchor_left,lo);right,rchecks=continue_side(anchor_right,hi)
    except core.Unresolved as exc:
        record['reason']=str(exc);persist();raise
    record['lateral_continuation']={'left':lchecks,'right':rchecks}
    Tl=T0+lo*(T1-T0);Bl=B0+lo*(B1-B0);Tr=T0+hi*(T1-T0);Br=B0+hi*(B1-B0)
    self.cache[self.key(Tl,Bl)]=left;self.cache[self.key(Tr,Br)]=right
    record.update(pre_graph=left,post_graph=right);persist()
    pmr=self.pm(Tr,Br);oldmin=left['minima'];newmin=right['minima'];U=np.array([p['m'] for p in oldmin]);V=np.array([p['m'] for p in newmin]);P=np.zeros((len(V),len(U)))
    zero_exact=geom.applicable(self.pm(Tl,Bl)) and geom.applicable(pmr)
    if zero_exact:
        P,maps,zero_proof=geom.zero_field_map(self.pm(Tl,Bl),pmr,left,right);born=[]
        record['zero_field_map_proof']=zero_proof
    else:
        cont,ok=ls.batch_newton(pmr,U);survive={};dead=[]
        record['minimum_continuation_checks']=[]
        for i,u in enumerate(cont):
            check=dict(old=i,Newton_converged=bool(ok[i]),candidate_direction=u.copy())
            record['minimum_continuation_checks'].append(check)
            if not ok[i]:
                # A failed continuation is not a stationary root. The
                # destination still requires the two verified lateral flows.
                check['disposition']='requires_verified_lateral_destination'
                dead.append(i);continue
            p=ls.point(pmr,u);u=p['m'];dist=np.linalg.norm(V-u,axis=1);j=int(np.argmin(dist))
            check.update(corrected_direction=u,index=p['index'],nearest_distance=float(dist[j]))
            if p['index']==0 and dist[j]<1e-6 and np.linalg.norm(u-U[i])<.15:survive[i]=j
            else:dead.append(i)
        for i,j in survive.items():P[j,i]=1
        born=[j for j in range(len(V)) if j not in survive.values()];maps=[]
        # Symmetry reverse: a surviving stationary parent can become unstable;
        # daughter roots must form an orbit of verified energy symmetries.
        for i in dead:
            distances=np.linalg.norm(V-U[i],axis=1);near=[j for j in born if distances[j]<.1]
            symmetric,group=ls.cubic_symmetries(pmr,U[i],[V[j] for j in near]) if near else (False,[])
            if symmetric:
                for j in near:P[j,i]=1/len(near)
                maps.append(dict(old=i,kind='verified_symmetric_reverse',destinations=near,symmetries=group))
            else:
                left_byid={p['point_id']:p for p in anchor_left['minima']}
                start=left_byid[oldmin[i]['point_id']]['m']
                amended=self._er2_c110_column(record,i)
                if amended is not None:
                    P[:,i]=np.asarray(amended['column'])
                    maps.append(dict(old=i,kind='E1_TQ01_R1_c110_symmetric_exit',destinations=amended['receivers'],amendment_proof=amended))
                    continue
                flows=[ls.flow(self.pm(Tr0,Br0),start,anchor_right['minima'],tolr) for tolr in [1e-9,1e-11]]
                if flows[0]['destination'] is None or flows[0]['destination']!=flows[1]['destination'] or not all(f['energy_monotone_decreasing'] for f in flows):
                    record['reason']='DISAPPEARING_MINIMUM_FLOW_DESTINATION_UNRESOLVED';record['failed_old_minimum']=i;record['flows']=flows
                    raise core.Unresolved(record['reason'])
                j=flows[-1]['destination'];P[j,i]=1;maps.append(dict(old=i,kind='death_flow_at_verified_lateral_reference',destinations=[j],flows=flows,reference_fraction=fr))
    if np.max(abs(P.sum(axis=0)-1))>1e-12 or np.min(P)<-1e-12:raise core.Unresolved('EVENT_MAP_NOT_STOCHASTIC')
    # Evidence for the critical root: tangential eigenvalue bracket of a
    # continued root, or a located fold KKT with opposite min/saddle sides.
    evidence=[]
    pmleft=self.pm(T0+lo*(T1-T0),B0+lo*(B1-B0))
    for a in left['points']:
        v,okay=ls.batch_newton(pmr,[a['m']])
        if not okay[0]:continue
        try:b=ls.point(pmr,v[0])
        except core.Unresolved:continue
        if okay[0] and np.linalg.norm(v[0]-a['m'])<.1:
            ea=np.array(a['eig']);eb=np.array(b['eig'])
            if np.any(ea*eb<0):evidence.append(dict(kind='signed_tangent_eigenvalue_bracket',before=ea,after=eb,direction=a['m']))
    if critical is not None and critical['kind']=='fold_stationary_KKT':
        direction=np.array(critical['root_direction']);existence=left if len(left['points'])>len(right['points']) else right
        near=sorted(existence['points'],key=lambda p:np.linalg.norm(p['m']-direction))[:2]
        if len(near)==2 and abs(near[0]['index']-near[1]['index'])==1 and max(np.linalg.norm(p['m']-direction) for p in near)<.01:
            signed=[float(min(p['eig'],key=abs)) for p in near]
            if signed[0]*signed[1]<0:evidence.append(dict(kind='fold_pair_opposite_tangent_curvatures_and_KKT_root',existence_side_eigenvalues=signed,temperature_enclosure_empirical=True))
    record.update(old_minima=U,new_minima=V,map=P,mapping_evidence=maps,isolated_births=born,curvature_evidence=evidence)
    if not evidence:
        record['reason']='EVENT_HAS_NO_DEMONSTRATED_SIGNED_CURVATURE_BRACKET'
        raise core.Unresolved(record['reason'])
    record['status']='COMPUTED_VALID';record.pop('reason',None);persist();self.cost['events']+=time.monotonic()-t
    if temperature and np.linalg.norm(B1-B0)==0 and not any(abs(x['T_C']-(T0+fc*(T1-T0)))<1e-6 and np.linalg.norm(np.array(x['B_sample_T'])-B0)<1e-15 for x in self.critical_catalog):
        self.critical_catalog.append(dict(T_C=T0+fc*(T1-T0),B_sample_T=B0.copy(),critical=copy.deepcopy(critical)))
    self.event_solutions[event_key]=(lo,hi,left,right,P.copy(),copy.deepcopy(record))
    return lo,hi,left,right,P
