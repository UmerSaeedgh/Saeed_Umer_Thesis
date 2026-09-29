"""Independent invariants and edge cases for the new scientific calculations."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import nbformat
from guided_analysis import Study, metrics, vec, EI, ROOT

checks=[]
def check(name,condition):
    assert condition,name
    checks.append({'check':name,'passed':True})

s=Study()
check('Human grid: 300 texts × 2 personas',len(s.h)==600)
check('Model grid: 300 texts × 2 personas × 6 configurations',len(s.l)==3600)
check('Matched raw model pair counts',len(s.individual)==24736 and int(s.individual.changed.sum())==5263)
check('Original votes retain the documented 27 off-inventory exclusions',int(s.original.vec_n.sum())==1473)
check('All comparison probabilities normalised',all(np.isclose(r.p.sum(),1) and np.isclose(r.q.sum(),1) for _,r in s.cells.iterrows()))
check('All delta vectors sum to zero',all(np.isclose(r.delta.sum(),0) for _,r in s.cells.iterrows()))
check('TV equals positive moved mass',all(np.isclose(r.tv,r.delta[r.delta>0].sum()) for _,r in s.cells.iterrows()))
check('All model comparisons have equal counts on both sides',s.l.neutral_n.eq(s.l.persona_n).all())
check('Five event outcomes partition each source',s.events().groupby('source').comparisons.sum().to_dict()=={'Human':600,'LLM':3600})
check('Strict hidden proportion changes are a subset of same-support changes',not (s.cells.strict_proportion & ~s.cells.same_support_change).any())
check('Complete-top-set hidden changes also hide under the fixed selected label',not (s.cells.hidden_top & ~s.cells.hidden_label).any())
check('Original human proportion-only count reconstructed',int(s.h.same_support_change.sum())==13 and int(s.h.strict_proportion.sum())==3)
check('Common-model proportion-only count reconstructed',int(s.l.same_support_change.sum())==543 and int(s.l.strict_proportion.sum())==310)

# Synthetic independently calculable cases.
a,b=EI['anger'],EI['sadness']
r=metrics(np.array([a,a,b]),np.array([a,b,b]))
check('One label flip can occur without any support change',r['flip']==1 and r['same_support_change'] and np.isclose(r['tv'],1/3))
r=metrics(np.array([a,a,b]),np.array([a,a,a,b,b]))
check('Stable winner and support can conceal a share change',r['strict_proportion'] and np.isclose(r['tv'],1/15))
r=metrics(np.array([a,a,a]),np.array([b,b,b]))
check('Disjoint supports have TV one and both appearance and disappearance',r['tv']==1 and r['gain_any'] and r['loss_any'])

# Exercise the exact null calculator on independent toy data.
toy=Study.__new__(Study);toy.null_cache={}
toy.h=pd.DataFrame([{'neutral_votes':np.array([a,a,a]),'persona_votes':np.array([b,b,b])}])
toy.l=pd.DataFrame([{'neutral_votes':np.array([a,a,a]),'persona_votes':np.array([a,a,b])}])
hn=toy._null('Human',0);ln=toy._null('LLM',0)
check('All 20 human allocations included for 3+3 votes',len(hn['tv'])==20)
check('Balanced pooled toy: exact mean TV is 0.4',np.isclose(hn['tv'].mean(),.4))
check('Balanced odd-sized groups can always have different selected labels under null',np.all(hn['flip']==1))
check('One discordant model gives constant swap-null TV',len(ln['tv'])==8 and np.allclose(ln['tv'],1/3))
check('Concordant majority can remain fixed despite a changed model',np.all(ln['flip']==0))

for p in sorted((ROOT/'distribution_analysis_v1').glob('0*.ipynb')):
    n=nbformat.read(p,as_version=4);nbformat.validate(n)
    codes=[c for c in n.cells if c.cell_type=='code']
    check(p.name+': sequential completed execution', [c.execution_count for c in codes]==list(range(1,len(codes)+1)))
    check(p.name+': no stored errors', not any(o.output_type=='error' for c in codes for o in c.outputs))
    check(p.name+': all cells executed in this method',n.metadata.get('clean_execution',{}).get('errors')==0)

previous=ROOT.parent/'submission_project'/'data'
if previous.exists():
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file():
            rel=p.relative_to(ROOT/'data')
            check('Raw input unchanged: '+str(rel),hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((previous/rel).read_bytes()).digest())

(ROOT/'verification'/'analysis_checks.json').write_text(json.dumps(checks,indent=2))
print(f'{len(checks)}/{len(checks)} scientific and execution checks passed.')
