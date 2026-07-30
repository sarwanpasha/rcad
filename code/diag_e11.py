import pandas as pd, numpy as np
d=pd.read_csv('ablation_results/e11_refsize_raw.csv')
print('columns:',list(d.columns))
print('n_ref values:',sorted(d.n_ref.unique()))
print('rules:',sorted(d.rule.unique()))
# eps_hat and Robust power as a function of n_ref, pooled over datasets/seeds
for det in sorted(d.detector.unique()):
    sub=d[d.detector==det]
    print('\n=== detector=%s ==='%det)
    g=sub.groupby(['rule','n_ref'])[['coverage','power','eps_hat']].mean().round(3)
    print(g.to_string())
