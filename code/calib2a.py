import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
import close_open as C
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

EMB = os.environ.get("EMB_OUT", os.path.dirname(os.path.abspath(__file__))+"/assets/emb")
ALPHA=0.10; SEEDS=[0,1,2,3,4]; KNN=5; MAHA_MAXDIM=100

def knn_score(Xfit, Xq, k=KNN):
    nn=NearestNeighbors(n_neighbors=k).fit(Xfit)
    d,_=nn.kneighbors(Xq); return d.mean(1)

def maha_prep(Xfit):
    p=None
    if Xfit.shape[1]>MAHA_MAXDIM:
        p=PCA(n_components=MAHA_MAXDIM, random_state=0).fit(Xfit); Xfit=p.transform(Xfit)
    mu=Xfit.mean(0); cov=np.cov(Xfit-mu,rowvar=False)+1e-3*np.eye(Xfit.shape[1])
    return (p,mu,np.linalg.pinv(cov))
def maha_score(prep,Xq):
    p,mu,P=prep
    if p is not None: Xq=p.transform(Xq)
    Q=Xq-mu; return np.einsum('ij,jk,ik->i',Q,P,Q)

rows=[]
for f in sorted(glob.glob(EMB+"/*.npz")):
    d=np.load(f); X=d["X"].astype(np.float64); y=d["y"].astype(int)
    name=os.path.basename(f).replace(".npz","")
    for det in ["knn","maha"]:
        pw_seed=[]; cov_seed=[]
        for s in SEEDS:
            rng=np.random.RandomState(s)
            nm=np.where(y==0)[0]; am=np.where(y==1)[0]
            rng.shuffle(nm)
            nfit=len(nm)//2; ncal=len(nm)-nfit
            fit_i=nm[:nfit]; cal_i=nm[nfit:nfit+ncal//2]; tno_i=nm[nfit+ncal//2:]
            sc=StandardScaler().fit(X[fit_i])
            Xf=sc.transform(X[fit_i]); Xcal=sc.transform(X[cal_i])
            Xtn=sc.transform(X[tno_i]); Xta=sc.transform(X[am])
            if det=="knn":
                s_cal=knn_score(Xf,Xcal); s_tn=knn_score(Xf,Xtn); s_ta=knn_score(Xf,Xta)
            else:
                mp=maha_prep(Xf); s_cal=maha_score(mp,Xcal); s_tn=maha_score(mp,Xtn); s_ta=maha_score(mp,Xta)
            thr=C.thr_robust(s_cal,ALPHA,1e-9)
            cov=float(np.mean(s_tn<=thr)); pw=float(np.mean(s_ta>thr))
            pw_seed.append(pw); cov_seed.append(cov)
        rows.append(dict(dataset=name,detector=det,clean_power=np.mean(pw_seed),clean_cov=np.mean(cov_seed),n_anom=len(am)))
        print("done %-22s %-5s pw=%.3f cov=%.3f"%(name,det,np.mean(pw_seed),np.mean(cov_seed)),flush=True)
df=pd.DataFrame(rows)
pd.set_option("display.width",160)
print(df.to_string(index=False,float_format=lambda x:"%.3f"%x))
df.to_csv(EMB+"/../calib2a_summary.csv",index=False)
print("\nMID-RANGE [0.25,0.85] problems (keep for eps-grid):")
print(df[(df.clean_power>=0.25)&(df.clean_power<=0.85)].to_string(index=False,float_format=lambda x:"%.3f"%x))
