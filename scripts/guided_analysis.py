"""Comparable one-label and distribution analyses of the supplied original votes.

No collection or sampling is rerun. All model contrasts use the same model identities
valid in BOTH conditions. Human votes are unpaired within a text. Permutation nulls
are conditional references, not causal estimators; independent cell randomisation
requires joint exchangeability. Bootstrap units are texts, not votes or cells.
"""
from pathlib import Path
from itertools import combinations, product
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from build_emotion_distributions import (
    EMOTIONS, MODEL_COLS, MODEL_RAW_COLS, POOL_TO_KEY, _normalize_label_fallback,
    build_original_neutral_vectors,
)
from stats_utils import holm_correct, bh_fdr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
CATS = ['unambiguous', 'author_independent', 'author_relevant']
CAT_NAMES = ['Unambiguous', 'Author-independent', 'Author-relevant']
CAT_NAME = dict(zip(CATS, CAT_NAMES))
COLORS = {'Human': '#177e89', 'LLM': '#d66b35'}
E = list(EMOTIONS)
EI = {e:i for i,e in enumerate(E)}
N_RANDOM = 10000

def setup():
    plt.rcParams.update({'figure.dpi':115,'font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':13,
        'figure.figsize':(10,5),'savefig.bbox':'tight'})
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.width', 150)

def explain(text):
    display(Markdown(text))

def table(df, name, digits=3):
    df.to_csv(RESULTS / (name+'.csv'), index=False)
    shown=df.round(digits).copy()
    # A Monte Carlo p-value at the simulation floor must never display as zero.
    for c in df.columns:
        if str(c).startswith('p_') or c in ['p_value','p_BH']:
            shown[c]=df[c].map(lambda x: f'{x:.4g}' if pd.notna(x) else '')
    display(shown)

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / (name+'.png'), dpi=170)
    plt.show()

def vec(votes):
    return np.bincount(votes, minlength=len(E)).astype(float)/len(votes)

def entropy(v):
    v=np.asarray(v); return -(np.where(v>0,v,1)*np.log(np.where(v>0,v,1))).sum(axis=-1)

def cosine(a,b):
    d=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(axis=-1),d,out=np.full_like(d,np.nan,dtype=float),where=d>1e-12)

def boot_mean(x, seed=42, n=N_RANDOM):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(n,len(x)))].mean(axis=1)
    return float(x.mean()), *np.quantile(means,[.025,.975])

def metrics(a,b):
    p,q=vec(a),vec(b); ps=p>0; qs=q>0
    tp=p==p.max(); tq=q==q.max()
    tv=np.abs(q-p).sum()/2; gain=qs&~ps; loss=ps&~qs
    same_support=np.array_equal(ps,qs); same_top=np.array_equal(tp,tq)
    w0=int(np.argmax(p));w1=int(np.argmax(q))
    wr0=len(E)-1-int(np.argmax(p[::-1]));wr1=len(E)-1-int(np.argmax(q[::-1]))
    if tv<1e-12: change='No distribution change'
    elif same_support: change='Same emotions; proportions change'
    elif gain.any() and loss.any(): change='Emotions appear AND disappear'
    elif gain.any(): change='Emotions appear only'
    else: change='Emotions disappear only'
    return dict(p=p,q=q,delta=q-p,neutral_votes=a,persona_votes=b,
        neutral_n=len(a),persona_n=len(b),winner0=w0,winner1=w1,
        flip=float(w0!=w1),flip_reverse=float(wr0!=wr1),tv=tv,
        neutral_tie=tp.sum()>1,persona_tie=tq.sum()>1,
        same_top=same_top,same_support=same_support,gained=gain,lost=loss,
        gain_any=bool(gain.any()),loss_any=bool(loss.any()),n_gained=int(gain.sum()),n_lost=int(loss.sum()),
        same_support_change=bool(same_support and tv>1e-12),
        hidden_label=bool(w0==w1 and tv>1e-12),hidden_top=bool(same_top and tv>1e-12),
        strict_proportion=bool(same_top and same_support and tv>1e-12),
        entropy_change=float(entropy(q)-entropy(p)),change=change)

class Study:
    def __init__(self):
        self.master=pd.read_csv(DATA/'sampled_texts.csv')
        self.master['text_id']=self.master.text_id.astype(str)
        assert self.master.text_id.is_unique and len(self.master)==300
        self.ids=sorted(self.master.text_id,key=int)
        self.category=self.master.set_index('text_id').classification.map(POOL_TO_KEY).to_dict()
        assert pd.Series(self.category).value_counts().reindex(CATS).eq(100).all()
        self.text=self.master.set_index('text_id').text.to_dict()
        self.human_raw=pd.read_csv(DATA/'human_annotations.csv')
        self.human_raw['text_id']=self.human_raw.text_id.astype(str)
        assert not self.human_raw.duplicated(['text_id','side']).any()
        self.hvotes={}
        self.personas={}
        for _,r in self.human_raw.iterrows():
            self.hvotes[(r.text_id,r.side)]=np.array([EI[x.strip().lower()] for x in str(r.raters_raw).split('|') if x.strip().lower() in EI])
            self.personas[(r.text_id,r.side)]=str(r.persona)
        assert len(self.hvotes)==900 and all(len(v)>=3 for v in self.hvotes.values())
        assert set(self.hvotes)==set(product(self.ids,['neutral','a','b']))
        raw=pd.read_csv(DATA/'experiment_results_all.csv');raw['text_id']=raw.text_id.astype(str)
        keys=['text_id','framing_condition','prompt_variant','label_format']
        assert len(raw)==7200 and not raw.duplicated(keys).any()
        self.configs=sorted(set(zip(raw.prompt_variant,raw.label_format)))
        assert len(self.configs)==6
        self.mvotes={}; exclusions={m:0 for m in MODEL_COLS};fallbacks=0
        for _,r in raw.iterrows():
            votes={}
            for m,c in MODEL_COLS.items():
                v=r[c]
                if pd.isna(v) or not str(v).strip():
                    v=_normalize_label_fallback(r[MODEL_RAW_COLS[m]]);fallbacks+=1
                v=str(v).strip().lower()
                if v in EI: votes[m]=EI[v]
                else: exclusions[m]+=1
            self.mvotes[(r.text_id,r.framing_condition,r.prompt_variant,r.label_format)]=votes
        assert set(self.mvotes)=={(tid,fr,pv,lf) for tid in self.ids for fr in ['neutral','generic_human','persona_a','persona_b'] for pv,lf in self.configs}
        self.exclusions=pd.DataFrame([dict(model=m,excluded=n,total=7200,percent=100*n/7200) for m,n in exclusions.items()])
        self.fallbacks=fallbacks
        h=[];l=[];individual=[]
        for tid in self.ids:
            for side in ['a','b']:
                h.append(dict(text_id=tid,category=self.category[tid],source='Human',side=side,
                    **metrics(self.hvotes[(tid,'neutral')],self.hvotes[(tid,side)])))
                for pv,lf in self.configs:
                    n=self.mvotes[(tid,'neutral',pv,lf)];p=self.mvotes[(tid,'persona_'+side,pv,lf)]
                    common=sorted(n.keys()&p.keys());assert len(common)>=3
                    a=np.array([n[m] for m in common]);b=np.array([p[m] for m in common])
                    l.append(dict(text_id=tid,category=self.category[tid],source='LLM',side=side,
                        config=pv+'/'+lf,models='|'.join(common),**metrics(a,b)))
                    individual.extend(dict(text_id=tid,category=self.category[tid],side=side,
                        config=pv+'/'+lf,model=m,changed=int(n[m]!=p[m])) for m in common)
        self.h=pd.DataFrame(h);self.l=pd.DataFrame(l);self.individual=pd.DataFrame(individual)
        self.cells=pd.concat([self.h,self.l],ignore_index=True)
        assert len(self.h)==600 and len(self.l)==3600
        assert self.l.groupby(['text_id','side']).size().eq(6).all()
        self.null_cache={};self.test_cache={}
        self.original=build_original_neutral_vectors(self.ids)
        self.original['text_id']=self.original.text_id.astype(str)
        self.old_results={};self.vector_results={}
        self.cells.drop(columns=['p','q','delta','neutral_votes','persona_votes','gained','lost']).to_csv(RESULTS/'comparison_audit.csv',index=False)

    def overview(self):
        rows=[]
        for src,d in [('Human',self.h),('LLM',self.l)]:
            rows.append(dict(source=src,texts=d.text_id.nunique(),neutral_persona_comparisons=len(d),
                observations_per_text=2 if src=='Human' else 12,
                minimum_votes=int(min(d.neutral_n.min(),d.persona_n.min())),
                maximum_votes=int(max(d.neutral_n.max(),d.persona_n.max()))))
        return pd.DataFrame(rows)

    def baseline(self):
        rows=[]
        orig=self.original.set_index('text_id')
        for tid in self.ids:
            h=vec(self.hvotes[(tid,'neutral')]);o=orig.loc[tid,[f'vec_prob_{e}' for e in E]].to_numpy(float)
            rows.append(dict(text_id=tid,category=self.category[tid],comparison='Original vs new human neutral',
                label_agrees=np.argmax(h)==np.argmax(o),tv=np.abs(h-o).sum()/2,cosine=cosine(h,o)))
            for pv,lf in self.configs:
                m=vec(list(self.mvotes[(tid,'neutral',pv,lf)].values()))
                rows.append(dict(text_id=tid,category=self.category[tid],comparison='Human vs LLM neutral',
                    label_agrees=np.argmax(h)==np.argmax(m),tv=np.abs(h-m).sum()/2,cosine=cosine(h,m)))
        return pd.DataFrame(rows)

    def summary(self,metric,by_category=True):
        groups=['source']+(['category'] if by_category else [])
        rows=[]
        for key,g in self.cells.groupby(groups,sort=False):
            key=key if isinstance(key,tuple) else (key,)
            x=g.groupby('text_id')[metric].mean().to_numpy(float)
            mean,lo,hi=boot_mean(x)
            rows.append(dict(zip(groups,key))|dict(comparisons=len(g),texts=len(x),value_pct=100*mean,lo_pct=100*lo,hi_pct=100*hi))
        return pd.DataFrame(rows)

    def bar(self,metric,name,title,ylabel):
        d=self.summary(metric);fig,ax=plt.subplots(figsize=(10,5))
        for j,src in enumerate(['Human','LLM']):
            q=d[d.source==src].set_index('category').reindex(CATS);x=np.arange(3)+(j-.5)*.35
            ax.bar(x,q.value_pct,.33,label=src,color=COLORS[src])
            ax.errorbar(x,q.value_pct,yerr=[q.value_pct-q.lo_pct,q.hi_pct-q.value_pct],fmt='none',color='#303030',capsize=4)
            for xx,yy in zip(x,q.value_pct):ax.text(xx,yy+1.5,f'{yy:.1f}%',ha='center',fontsize=10)
        ax.set_xticks(range(3),CAT_NAMES);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend()
        ax.set_ylim(0,min(110,max(d.hi_pct)*1.22+3));finish(fig,name);table(d,name)
        return d

    def _null(self,source,index):
        key=(source,index)
        if key in self.null_cache:return self.null_cache[key]
        r=(self.h if source=='Human' else self.l).iloc[index]
        a,b=r.neutral_votes,r.persona_votes
        if source=='Human':
            pool=np.concatenate([a,b]);n=len(a)
            total=np.bincount(pool,minlength=len(E))
            counts=np.array([np.bincount(pool[list(c)],minlength=len(E)) for c in combinations(range(len(pool)),n)])
            pa=counts/n;pb=(total-counts)/len(b)
        else:
            bits=np.array(list(product([False,True],repeat=len(a))))
            av=np.where(bits,b,a);bv=np.where(bits,a,b)
            pa=np.eye(len(E))[av].mean(axis=1);pb=np.eye(len(E))[bv].mean(axis=1)
        result={'flip':(pa.argmax(axis=1)!=pb.argmax(axis=1)).astype(float),'tv':np.abs(pa-pb).sum(axis=1)/2}
        self.null_cache[key]=result;return result

    def hypotheses(self,metric):
        if metric in self.test_cache:return self.test_cache[metric].copy()
        rows=[]
        for source,d in [('Human',self.h),('LLM',self.l)]:
            for side in ['a','b']:
                for cat in CATS if source=='Human' else ['all']:
                    sub=d[(d.side==side)&((d.category==cat) if cat!='all' else True)]
                    rng=np.random.default_rng(4200+len(rows));null=np.zeros(N_RANDOM);expect=0
                    for index in sub.index:
                        values=self._null(source,index)[metric]
                        null+=values[rng.integers(0,len(values),N_RANDOM)]
                        expect+=values.mean()
                    null/=len(sub);expect/=len(sub);obs=sub[metric].mean()
                    p=(1+np.count_nonzero(null>=obs-1e-12))/(N_RANDOM+1)
                    mean,lo,hi=boot_mean(sub.groupby('text_id')[metric].mean())
                    rows.append(dict(hypothesis='H1' if source=='Human' else 'H2',source=source,side=side,
                        category=cat,n_cells=len(sub),observed_pct=100*obs,conditional_null_pct=100*expect,
                        excess_pp=100*(obs-expect),ci_lo=100*lo,ci_hi=100*hi,p_raw=p))
        out=pd.DataFrame(rows);out['p_holm']=np.nan
        for h in ['H1','H2']:
            use=out.hypothesis==h;out.loc[use,'p_holm']=holm_correct(out.loc[use,'p_raw'])
        out['reject_05']=out.p_holm<.05
        self.test_cache[metric]=out
        return out.copy()

    def h3(self,metric):
        d=self.h.copy()
        d['expected']=[self._null('Human',i)[metric].mean() for i in d.index]
        d['excess']=d[metric]-d.expected
        t=d.groupby(['text_id','category'])[[metric,'expected','excess']].mean().reset_index()
        rows=[]
        for mode,col in [('Raw',metric),('Conditional-null referenced','excess')]:
            for a,b in [('author_relevant','author_independent'),('author_independent','unambiguous'),('author_relevant','unambiguous')]:
                x=t.loc[t.category==a,col].to_numpy();y=t.loc[t.category==b,col].to_numpy();obs=x.mean()-y.mean()
                rng=np.random.default_rng(102+len(rows));pooled=np.r_[x,y]
                indices=np.argsort(rng.random((N_RANDOM,len(pooled))),axis=1)
                perms=pooled[indices];null=perms[:,:len(x)].mean(axis=1)-perms[:,len(x):].mean(axis=1)
                p=(1+(np.abs(null)>=abs(obs)-1e-12).sum())/(N_RANDOM+1)
                boots=x[rng.integers(0,len(x),(N_RANDOM,len(x)))].mean(axis=1)-y[rng.integers(0,len(y),(N_RANDOM,len(y)))].mean(axis=1)
                lo,hi=np.quantile(boots,[.025,.975])
                rows.append(dict(analysis=mode,contrast=CAT_NAME[a]+' minus '+CAT_NAME[b],difference_pp=100*obs,ci_lo=100*lo,ci_hi=100*hi,p_raw=p))
        out=pd.DataFrame(rows);out['p_holm']=np.nan
        for mode in out.analysis.unique():
            use=out.analysis==mode;out.loc[use,'p_holm']=holm_correct(out.loc[use,'p_raw'])
        return out,t

    def h4(self,metric):
        h=self.h.groupby('text_id')[metric].mean().reindex(self.ids)
        l=self.l.groupby('text_id')[metric].mean().reindex(self.ids)
        point,lo,hi=boot_mean(h-l)
        return pd.DataFrame([dict(metric=metric,n_texts=300,human_mean_pct=100*h.mean(),llm_mean_pct=100*l.mean(),
            human_minus_llm_pp=100*point,ci_lo=100*lo,ci_hi=100*hi)])

    def hypothesis_plot(self,metric,name):
        d=self.hypotheses(metric);fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[2,1]})
        for ax,h in zip(axes,['H1','H2']):
            q=d[d.hypothesis==h].reset_index(drop=True);y=np.arange(len(q))
            ax.errorbar(q.observed_pct,y,xerr=[q.observed_pct-q.ci_lo,q.ci_hi-q.observed_pct],fmt='o',color='#177e89',capsize=3,label='Observed; 95% text CI')
            ax.scatter(q.conditional_null_pct,y,marker='x',color='#d66b35',label='Conditional-null mean')
            ax.set_yticks(y,[('All texts' if r.category=='all' else CAT_NAME[r.category])+' / '+r.side.upper() for _,r in q.iterrows()])
            for i,r in q.iterrows():ax.annotate(f'p={r.p_holm:.4f}',(r.observed_pct,i),xytext=(0,10),textcoords='offset points',fontsize=8)
            ax.set_xlim(0,100);ax.set_title(h+': '+('one-label changes' if metric=='flip' else 'distribution distances'));ax.set_xlabel('Changed comparisons (%)' if metric=='flip' else 'TV × 100')
            ax.set_ylim(-.7,len(q)-.3)
        axes[0].legend(fontsize=8,loc='lower left');finish(fig,name);table(d,name)
        return d

    def ties(self):
        rows=[]
        for src,g in self.cells.groupby('source',sort=False):
            unique=~g.neutral_tie&~g.persona_tie
            rows.append(dict(source=src,comparisons=len(g),either_condition_tied=int((~unique).sum()),
                alphabetic_change_pct=100*g.flip.mean(),reverse_order_change_pct=100*g.flip_reverse.mean(),
                changed_decision=int((g.flip!=g.flip_reverse).sum()),unique_winners_only_n=int(unique.sum()),
                unique_winners_only_change_pct=100*g.loc[unique,'flip'].mean()))
        return pd.DataFrame(rows)

    def label_agreement(self):
        h=self.h[['text_id','side','flip','winner0','winner1']].rename(columns={c:'h_'+c for c in ['flip','winner0','winner1']})
        j=self.l.merge(h,on=['text_id','side'],validate='many_to_one')
        j['status']=np.select([(j.h_flip==0)&(j.flip==0),(j.h_flip==1)&(j.flip==0),(j.h_flip==0)&(j.flip==1)],['Both keep their label','Only humans change label','Only LLMs change label'],default='Both change label')
        both=(j.h_flip==1)&(j.flip==1)
        agree=(j.winner0==j.h_winner0)&(j.winner1==j.h_winner1)
        return j,pd.DataFrame([dict(comparisons=len(j),both_changed=int(both.sum()),same_transition_among_both_changed=int((both&agree).sum()),
            same_transition_pct=100*agree[both].mean(),change_status_agreement_pct=100*(j.h_flip==j.flip).mean())])

    def direction(self,n_perm=N_RANDOM):
        # Retain persona sides separately. Config order fixed across all texts.
        h_index=self.h.set_index(['text_id','side'])
        l_index=self.l.set_index(['text_id','side','config'])
        hd=np.stack([h_index.loc[(tid,side),'delta'] for tid in self.ids for side in ['a','b']]).reshape(300,2,11)
        ld=np.stack([l_index.loc[(tid,side,pv+'/'+lf),'delta'] for tid in self.ids for side in ['a','b'] for pv,lf in self.configs]).reshape(300,2,6,11)
        hs=np.repeat(hd[:,:,None,:],6,axis=2)
        cos=cosine(hs,ld);mask=np.isfinite(cos)
        sums=np.nansum(cos,axis=(1,2));counts=mask.sum(axis=(1,2));obs=sums.sum()/counts.sum()
        rng=np.random.default_rng(301);idx=rng.integers(0,300,(N_RANDOM,300))
        boot=sums[idx].sum(axis=1)/counts[idx].sum(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        null=np.empty(n_perm)
        for b in range(n_perm):
            perm=rng.permutation(300);cs=cosine(hs,ld[perm]);null[b]=np.nanmean(cs)
        p=(1+(null>=obs-1e-12).sum())/(n_perm+1)
        return pd.DataFrame([dict(mean_cosine=obs,ci_lo=lo,ci_hi=hi,defined_comparisons=int(mask.sum()),all_comparisons=3600,
            texts_with_any_defined=int((counts>0).sum()),null_mean=float(null.mean()),p_permutation=p)]),cos,null

    def label_direction(self):
        # Categorical proxy: same selected origin AND destination, among pairs
        # in which BOTH sources change. Text profiles permute as whole blocks.
        j,_=self.label_agreement()
        j=j.sort_values(['text_id','side','config'],key=lambda x:x.astype(int) if x.name=='text_id' else x)
        h0=j.h_winner0.to_numpy().reshape(300,12);h1=j.h_winner1.to_numpy().reshape(300,12)
        l0=j.winner0.to_numpy().reshape(300,12);l1=j.winner1.to_numpy().reshape(300,12)
        eligible=(h0!=h1)&(l0!=l1);same=(h0==l0)&(h1==l1)&eligible
        sums=same.sum(axis=1);counts=eligible.sum(axis=1);obs=sums.sum()/counts.sum()
        rng=np.random.default_rng(302);idx=rng.integers(0,300,(N_RANDOM,300))
        boots=sums[idx].sum(axis=1)/counts[idx].sum(axis=1);lo,hi=np.quantile(boots,[.025,.975])
        null=np.empty(N_RANDOM)
        for b in range(N_RANDOM):
            ix=rng.permutation(300);ok=(h0!=h1)&(l0[ix]!=l1[ix]);match=(h0==l0[ix])&(h1==l1[ix])&ok
            null[b]=match.sum()/ok.sum() if ok.any() else np.nan
        p=(1+np.count_nonzero(null>=obs-1e-12))/(1+np.isfinite(null).sum())
        return pd.DataFrame([dict(eligible_comparisons=int(counts.sum()),same_selected_transition=int(sums.sum()),
            agreement_pct=100*obs,ci_lo=100*lo,ci_hi=100*hi,null_mean_pct=100*np.nanmean(null),p_permutation=p)])

    def transitions(self):
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,(src,d) in zip(axes,[('Human',self.h),('LLM',self.l)]):
            mat=pd.crosstab(d.winner0,d.winner1).reindex(index=range(11),columns=range(11),fill_value=0)
            arr=mat.to_numpy();ax.imshow(arr,cmap='Blues')
            for i in range(11):
                for j in range(11):
                    if arr[i,j]:ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=7,color='white' if arr[i,j]>arr.max()/2 else 'black')
            ax.set_xticks(range(11),E,rotation=60,ha='right',fontsize=8);ax.set_yticks(range(11),E,fontsize=8)
            ax.set_title(f'{src}: {len(d):,} comparisons');ax.set_xlabel('Selected persona label');ax.set_ylabel('Selected neutral label')
        finish(fig,'old_transitions')

    def events(self):
        types=['No distribution change','Same emotions; proportions change','Emotions appear only','Emotions disappear only','Emotions appear AND disappear']
        rows=[]
        for src,d in [('Human',self.h),('LLM',self.l)]:
            for change in types:
                flag=d.change==change
                rows.append(dict(source=src,event=change,comparisons=int(flag.sum()),denominator=len(d),percent=100*flag.mean(),
                    texts_with_event=int(d.loc[flag,'text_id'].nunique()),text_denominator=300))
        return pd.DataFrame(rows)

    def events_plot(self):
        d=self.events();fig,ax=plt.subplots(figsize=(11,5))
        events=d.event.drop_duplicates().tolist();y=np.arange(len(events))
        for j,src in enumerate(['Human','LLM']):
            q=d[d.source==src];ax.barh(y+(j-.5)*.36,q.percent,.34,label=src,color=COLORS[src])
            for yy,(_,r) in zip(y+(j-.5)*.36,q.iterrows()):ax.text(r.percent+.5,yy,f'{r.percent:.1f}% ({r.comparisons:,}/{r.denominator:,})',va='center',fontsize=9)
        ax.set_yticks(y,events);ax.invert_yaxis();ax.set_xlim(0,min(100,d.percent.max()+26));ax.set_xlabel('Share of neutral–persona comparisons (%)')
        ax.set_title('What changed inside the distribution?');ax.legend();finish(fig,'vector_change_types');table(d,'vector_change_types')
        return d

    def emotion_events(self):
        rows=[];fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,(src,d) in zip(axes,[('Human',self.h),('LLM',self.l)]):
            gained=np.stack(d.gained).mean(axis=0)*100;lost=np.stack(d.lost).mean(axis=0)*100
            y=np.arange(11);ax.barh(y-.18,gained,.35,color='#177e89',label='Appears (zero → positive)');ax.barh(y+.18,lost,.35,color='#d66b35',label='Disappears (positive → zero)')
            ax.set_yticks(y,E);ax.invert_yaxis();ax.set_title(src);ax.set_xlabel('Comparisons (%)');ax.legend(fontsize=8)
            for e,g,l in zip(E,gained,lost):rows.append(dict(source=src,emotion=e,appears_pct=g,disappears_pct=l))
        finish(fig,'emotion_appearance_disappearance');out=pd.DataFrame(rows);table(out,'emotion_appearance_disappearance');return out

    def count_sensitivity(self,draws=200):
        rng=np.random.default_rng(42);rows=[]
        for src,d in [('Human',self.h),('LLM',self.l)]:
            for _,r in d.iterrows():
                a,b=r.neutral_votes,r.persona_votes
                ia=np.argsort(rng.random((draws,len(a))),axis=1)[:,:3]
                ib=np.argsort(rng.random((draws,len(b))),axis=1)[:,:3] if src=='Human' else ia
                p=np.eye(11)[a[ia]].mean(axis=1);q=np.eye(11)[b[ib]].mean(axis=1)
                rows.append(dict(source=src,text_id=r.text_id,category=r.category,
                    tv_raw=r.tv,tv_standardised=np.abs(q-p).sum(axis=1).mean()/2,
                    entropy_raw=r.entropy_change,entropy_standardised=(entropy(q)-entropy(p)).mean()))
        out=pd.DataFrame(rows);summary=[]
        for src,g in out.groupby('source',sort=False):
            for col in ['tv_raw','tv_standardised','entropy_raw','entropy_standardised']:
                point,lo,hi=boot_mean(g.groupby('text_id')[col].mean())
                summary.append(dict(source=src,quantity=col,mean=point,ci_lo=lo,ci_hi=hi))
        return out,pd.DataFrame(summary)

    def prompt(self,metric):
        d=self.l.groupby(['text_id','config'])[metric].mean().unstack('config').reindex(self.ids)
        stat,p=stats.friedmanchisquare(*[d[c] for c in d])
        return d,pd.DataFrame([dict(metric=metric,texts=300,configurations=6,friedman_stat=stat,p_value=p)])

    def example(self,event,source,name):
        d=self.h if source=='Human' else self.l
        choose=d[d[event]].sort_values(['tv','text_id'],ascending=[False,True])
        if choose.empty:explain(f'No {source} comparison meets this definition.');return
        r=choose.iloc[0];p,q=r.p,r.q
        active=np.flatnonzero((p+q)>0);x=np.arange(len(active));fig,ax=plt.subplots(figsize=(10,4.5))
        ax.bar(x-.18,p[active]*100,.35,label='Neutral',color='#586f7c');ax.bar(x+.18,q[active]*100,.35,label='Persona',color=COLORS[source])
        for xx,pp,qq in zip(x,p[active]*100,q[active]*100):
            ax.text(xx-.18,pp+1,f'{pp:.0f}%',ha='center',fontsize=9);ax.text(xx+.18,qq+1,f'{qq:.0f}%',ha='center',fontsize=9)
        ax.set_xticks(x,[E[i] for i in active]);ax.set_ylim(0,110);ax.set_ylabel('Vote share (%)');ax.legend()
        ax.set_title(f'{source}, text {r.text_id}, persona {r.side.upper()} — TV {r.tv*100:.1f}%')
        finish(fig,name)
        explain(f'**Actual text:** {self.text[r.text_id]}\n\n**Persona:** {self.personas[(r.text_id,r.side)]}\n\n'
            f'**Selected label:** {E[r.winner0]} → {E[r.winner1]}. **Votes:** {r.neutral_n} → {r.persona_n}. '
            f'**Observed emotions added:** {", ".join(np.array(E)[r.gained]) or "none"}; '
            f'**removed:** {", ".join(np.array(E)[r.lost]) or "none"}. '
            +(f'**Configuration:** {r.config}. ' if source=='LLM' else '')+
            'This is the largest-TV example meeting the stated rule, selected to illustrate the pattern. It is not a representative effect estimate.')

def record_hashes():
    rows=[]
    for p in sorted(DATA.rglob('*')):
        if p.is_file():rows.append(dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(RESULTS/'input_sha256.csv',index=False)
