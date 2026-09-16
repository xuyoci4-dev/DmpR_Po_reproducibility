from pathlib import Path
import sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.inference import predict_ensemble
from src.metrics import regression_metrics
data=pd.read_csv(ROOT/'data'/'batch2_cross_batch_evaluation.csv'); _,_,ensemble=predict_ensemble(data.Sequence.tolist(),ROOT)
print('Batch 2 non-WT n=',len(data)); print('Equal-weight ensemble',regression_metrics(data.Weighted_Score,ensemble))
