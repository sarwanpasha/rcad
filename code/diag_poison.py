import os,glob,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
EMB=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','emb')
ALPHA=0.10; SEEDS=[0,1,2,3,4,5,6,7,8,9]; KNN=5; MAHA_MAXDIM=100
CONFIGS=['clipvit_cifar10_c1','clipvit_cifar10_c9','minilm_news20_c7']
def knn_fit(X):return NearestNeighbors(n_neighbors=KNN).fit(X)
def knn_score(nn,Xq):d,_=nn.kneighbors(Xq);return d.mean(1)
def maha_fit(X):
    p=None
    if X.shape[1]>MAHA_MAXDIM:
        p=PCA(n_components=MAHA_MAXDIM,random_state=0).fit(X);X=p.transform(X)
    mu=X.mean(0);cov=np.cov(X-mu,rowvar=False)+1e-3*np.eye(X.shape[1]);return (p,mu,np.linalg.pinv(cov))
def maha_score(m,Xq):
    p,mu,P=m
    if p is not None:Xq=p.transform(Xq)
    Q=Xq-mu;return np.einsum('ij,jk,ik->i',Q,P,Q)
print('config,detector,cal_eps,clean_power_meanseed,poison_frac_above_calmax,poison_pctl_in_cal')
for name in CONFIGS:
    f=os.path.join(EMB,name+'.npz'); d=np.load(f); X=d['X'].astype(np.float64); y=d['y'].astype(int)
    for det in ['knn','maha']:
        for ce in [0.0,0.20]:
            cleanpows=[]; frac_list=[]; pctl_list=[]
            for s in SEEDS:
                rng=np.random.RandomState(s)
                nm=np.where(y==0)[0].copy(); am=np.where(y==1)[0].copy()
                rng.shuffle(nm); rng.shuffle(am)
                nfit=len(nm)//2; rest=nm[nfit:]; fit_i=nm[:nfit]
                ncal=len(rest)//2; cal_i=rest[:ncal]; tno_i=rest[ncal:]
                na_inj=int(round(ce*len(cal_i)/(1-ce))) if ce>0 else 0
                na_inj=min(na_inj,len(am)//2); inj_i=am[:na_inj]; ta_i=am[na_inj:]
                sc=StandardScaler().fit(X[fit_i]); tf=lambda idx:sc.transform(X[idx]); Xf=tf(fit_i)
                mdl=knn_fit(Xf) if det=='knn' else maha_fit(Xf)
                scorer=knn_score if det=='knn' else maha_score
                s_cal_clean=scorer(mdl,tf(cal_i))
                s_inj=scorer(mdl,tf(inj_i)) if na_inj>0 else np.array([])
                if len(ta_i)==0: continue
                s_ta=scorer(mdl,tf(ta_i))
                thr=np.quantile(s_cal_clean,1-ALPHA)
                cleanpows.append(float((s_ta>thr).mean()))
                if na_inj>0:
                    cmax=s_cal_clean.max()
                    frac_list.append(float((s_inj>cmax).mean()))
                    # percentile position of median poison within clean cal
                    pctl_list.append(float((s_cal_clean<np.median(s_inj)).mean()))
            cp=np.mean(cleanpows)
            fr=np.mean(frac_list) if frac_list else float('nan')
            pc=np.mean(pctl_list) if pctl_list else float('nan')
            print('%s,%s,%.2f,%.3f,%.3f,%.3f'%(name,det,ce,cp,fr,pc))
