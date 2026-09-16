from scipy.stats import spearmanr
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
def regression_metrics(observed,predicted): return {'Spearman':float(spearmanr(observed,predicted).statistic),'R2':float(r2_score(observed,predicted)),'RMSE':float(mean_squared_error(observed,predicted)**.5),'MAE':float(mean_absolute_error(observed,predicted))}
