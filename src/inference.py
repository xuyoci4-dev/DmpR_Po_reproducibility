from pathlib import Path
import joblib,numpy as np,torch
from .encoding import encode_880,encode_cnn
from .model_definitions import MLP880,ResidualCNN
def _predict(paths,kind,matrix):
    values=[]
    for path in sorted(paths):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model=MLP880(payload['config']['dropout']) if kind=='mlp' else ResidualCNN(payload['config']['dropout'])
        model.load_state_dict(payload['state_dict'],strict=True); model.eval()
        with torch.no_grad(): y=np.concatenate([model(torch.from_numpy(matrix[i:i+512])).numpy() for i in range(0,len(matrix),512)])
        values.append(y*payload['target_std']+payload['target_mean'])
    if len(values)!=3: raise RuntimeError(f'Expected three {kind} seed checkpoints')
    return np.mean(values,axis=0)
def predict_ensemble(sequences,root):
    root=Path(root); x=encode_880(sequences); scaler=joblib.load(root/'models'/'mlp_final'/'feature_scaler.joblib')
    mlp=_predict((root/'models'/'mlp_final').glob('seed_*.pt'),'mlp',scaler.transform(x).astype(np.float32))
    cnn=_predict((root/'models'/'residual_cnn_final').glob('seed_*.pt'),'cnn',encode_cnn(sequences))
    return mlp,cnn,.5*mlp+.5*cnn
