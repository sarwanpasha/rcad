import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
import close_open as C, rcad_v2 as R
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

EMB=os.environ.get("EMB_OUT", os.path.dirname(os.path.abspath(__file__))+"/assets/emb")
ALPHA=0.10; SEEDS=[0,1,2,3,4,5,6,7,8,9]; KNN=5
CAL_EPS=[0.0,0.05,0.10,0.20]     # contamination injected into CALIBRATION normals (mislabeled anomalies)
KEEP=os.environ.get("KEEP_PROBLEMS","").split(",") if os.environ.get("KEEP_PROBLEMS") else None

def knn_fit(X): return NearestNeighbors(n_neighbors=KNN).fit(X)
def knn_score(nn,Xq): d,_=nn.kneighbors(Xq); return d.mean(1)
MAHA_MAXDIM=100
def maha_fit(X):
    p=None
    if X.shape[1]>MAHA_MAXDIM:
        p=PCA(n_components=MAHA_MAXDIM,random_state=0).fit(X); X=p.transform(X)
    mu=X.mean(0); cov=np.cov(X-mu,rowvar=False)+1e-3*np.eye(X.shape[1]); return (p,mu,np.linalg.pinv(cov))
def maha_score(m,Xq):
    p,mu,P=m
    if p is not None: Xq=p.transform(Xq)
    Q=Xq-mu; return np.einsum('ij,jk,ik->i',Q,P,Q)

rows=[]
for f in sorted(glob.glob(EMB+"/*.npz")):
    name=os.path.basename(f).replace(".npz","")
    if KEEP and name not in KEEP: continue
    d=np.load(f); X=d["X"].astype(np.float64); y=d["y"].astype(int)
    for det in ["knn","maha"]:
        for ce in CAL_EPS:
            for s in SEEDS:
                rng=np.random.RandomState(s)
                nm=np.where(y==0)[0].copy(); am=np.where(y==1)[0].copy()
                rng.shuffle(nm); rng.shuffle(am)
                nfit=len(nm)//2; rest=nm[nfit:]; fit_i=nm[:nfit]
                ncal=len(rest)//2; cal_i=rest[:ncal]; tno_i=rest[ncal:]
                # split anomalies: pool for injection vs test
                na_inj=int(round(ce*len(cal_i)/(1-ce))) if ce>0 else 0
                na_inj=min(na_inj,len(am)//2)
                inj_i=am[:na_inj]; ta_i=am[na_inj:]
                sc=StandardScaler().fit(X[fit_i])
                tf=lambda idx: sc.transform(X[idx])
                Xf=tf(fit_i)
                fitter=knn_fit if det=="knn" else maha_fit
                scorer=knn_score if det=="knn" else maha_score
                mdl=fitter(Xf)
                s_cal_clean=scorer(mdl,tf(cal_i))
                s_inj=scorer(mdl,tf(inj_i)) if na_inj>0 else np.array([])
                s_cal=np.concatenate([s_cal_clean,s_inj]) if na_inj>0 else s_cal_clean
                s_ref=scorer(mdl,tf(tno_i))   # clean reference = held-out test normals proxy
                s_tn=scorer(mdl,tf(tno_i)); s_ta=scorer(mdl,tf(ta_i))
                ehat,Dks,crit=C.eps_hat_ks(s_cal,s_ref)
                for rule,kind,e in [("Standard","std",0.0),("Robust","rob",ce),("Auto","rob",ehat)]:
                    thr=C.thr_standard(s_cal,ALPHA) if kind=="std" else C.thr_robust(s_cal,ALPHA,max(e,1e-9))
                    cov=float(np.mean(s_tn<=thr)); pw=float(np.mean(s_ta>thr))
                    loc,_=R.clopper_pearson(int(np.sum(s_tn<=thr)),len(s_tn),alpha=0.10)
                    lop,_=R.clopper_pearson(int(np.sum(s_ta>thr)),len(s_ta),alpha=0.10)
                    rows.append(dict(dataset=name,detector=det,cal_eps=ce,rule=rule,seed=s,
                        coverage=cov,cov_lcb90=float(loc),power=pw,power_lcb90=float(lop),
                        eps_hat=float(ehat),threshold=float(thr),n_test_norm=len(s_tn),n_test_anom=len(s_ta)))
df=pd.DataFrame(rows)
out=EMB+"/../grid2b_raw.csv"; df.to_csv(out,index=False)
print("WROTE %s rows=%d"%(out,len(df)))
g=df.groupby(["dataset","detector","cal_eps","rule"]).agg(cov=("coverage","mean"),pw=("power","mean"),eps_hat=("eps_hat","mean")).reset_index()
pd.set_option("display.width",200); pd.set_option("display.max_rows",400)
print(g.to_string(index=False,float_format=lambda x:"%.3f"%x))
