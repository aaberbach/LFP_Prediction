# Make plot comparing models

import numpy as np
import pdb
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as ss
from statsmodels.graphics.gofplots import qqplot

np.random.seed(10304)

########## Experimental predictions ############
# Load data
file = open('./data/channel0-preds-trues.pkl', "br")
data = pickle.load(file)

# Pick a subset for plotting
idx = np.random.choice(np.arange(data['multi_ensemble'].shape[0]),10000)

# Get y-true
y_true = data['y_true'][idx,:]

models = ['AR','uni_ensemble',\
            'multi_ensemble']
RMSE = np.zeros((idx.shape[0],len(models)))

for j,i in enumerate(models):
    preds = data[i][idx,:]
    RMSE[:,j] = np.log(np.sqrt(np.mean((preds-y_true)**2,axis=1)))

RMSE_df = pd.DataFrame(RMSE,columns=models)




plt.figure()
qqplot(RMSE_df['AR'], line='s')

print('####### EXPERIMENTAL ###########')

stat, p = ss.shapiro(RMSE_df['AR']) #shapiro-wilks test for normality
print('Shapiro-Wilks test for normality\n \
        variable: RMSE_df[AR]\n\
        p = {}'.format(p))
        
stat, p = ss.normaltest(RMSE_df['AR']) #shapiro-wilks test for normality
print('DAgostinos test for normality\n \
        variable: RMSE_df[AR]\n\
        p = {}'.format(p))

p = ss.ranksums(RMSE_df['AR'],RMSE_df['uni_ensemble']).pvalue
print('Wilcoxon rank-sum test p-value between AR and uni_ensemble = {}'.format(p))

p = ss.ranksums(RMSE_df['uni_ensemble'],RMSE_df['multi_ensemble']).pvalue
print('Wilcoxon rank-sum test p-value between uni_ensemble and multi_ensemble = {}'.format(p))

print('###################################')
RMSE_df = RMSE_df.melt(var_name='groups', value_name='vals')

plt.figure()
ax = sns.violinplot(x="groups", y="vals", data=RMSE_df)
ax.set_xlabel([])
ax.set_ylabel('log(RMSE)')
plt.title('experimental')
#################################################

############### Model predictions ###############
# Load data
file = open('./data/model_uni_preds.pkl', "br")
data_uni = pickle.load(file)

file = open('./data/model_multi_preds.pkl', "br")
data_multi = pickle.load(file)

# Pick a subset for plotting
idx = np.random.choice(np.arange(data_uni['ensemble'].shape[0]),10000)

y_true = np.load('./data/model_y_true.npy')
y_true = y_true[idx,:]

models = ['ar','ensemble']
RMSE = np.zeros((idx.shape[0],3))

for j,i in enumerate(models):
    preds = data_uni[i][idx,:]
    RMSE[:,j] = np.log(np.sqrt(np.mean((preds-y_true)**2,axis=1)))

preds = data_multi['ensemble'][idx,:]
RMSE[:,2] = np.log(np.sqrt(np.mean((preds-y_true)**2,axis=1)))

RMSE_df = pd.DataFrame(RMSE,columns=['AR-only LFP','ens.-only LFP','ens.-LFP+FR'])

print('####### MODEL ###########')

stat, p = ss.shapiro(RMSE_df['AR-only LFP']) #shapiro-wilks test for normality
print('Shapiro-Wilks test for normality\n \
        variable: RMSE_df[AR-only LFP]\n\
        p = {}'.format(p))
        
stat, p = ss.normaltest(RMSE_df['AR-only LFP']) #shapiro-wilks test for normality
print('DAgostinos test for normality\n \
        variable: RMSE_df[AR-only LFP]\n\
        p = {}'.format(p))
       

p = ss.ranksums(RMSE_df['AR-only LFP'],RMSE_df['ens.-only LFP']).pvalue
print('Wilcoxon rank-sum test p-value between AR and uni_ensemble = {}'.format(p))

p = ss.ranksums(RMSE_df['ens.-only LFP'],RMSE_df['ens.-LFP+FR']).pvalue
print('Wilcoxon rank-sum test p-value between uni_ensemble and multi_ensemble = {}'.format(p))

print('###################################')


RMSE_df = RMSE_df.melt(var_name='groups', value_name='vals')

plt.figure()
ax = sns.violinplot(x="groups", y="vals", data=RMSE_df)
ax.set_xlabel([])
ax.set_ylabel('RMSE')
plt.title('model')

plt.show()
pdb.set_trace()



