from pathlib import Path
import sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.inference import predict_ensemble
from src.metrics import regression_metrics
data=pd.read_csv(ROOT/'data'/'fixed_split_manifest.csv'); test=data.loc[data.evaluation_split.eq('held_out_test')].copy()
mlp,cnn,ensemble=predict_ensemble(test.Sequence.tolist(),ROOT)
out=test[['Sequence','Weighted_Score']].copy(); out['MLP_prediction']=mlp; out['Residual_CNN_prediction']=cnn; out['Ensemble_prediction']=ensemble; out.to_csv(ROOT/'reproduced_test_predictions.csv',index=False)
for name,prediction in [('MLP',mlp),('Residual CNN',cnn),('Equal-weight ensemble',ensemble)]: print(name,regression_metrics(test.Weighted_Score,prediction))
archive=pd.read_csv(ROOT/'data'/'archived_final_test_predictions.csv')[['Sequence','top2_average']]
check=out[['Sequence','Ensemble_prediction']].merge(archive,on='Sequence',validate='one_to_one')
print('Archived ensemble max_abs_difference=',float((check.Ensemble_prediction-check.top2_average).abs().max()))
