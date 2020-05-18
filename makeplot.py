# Make plot comparing models

import numpy as np
import pdb
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(10304)

# Load data
file = open('./data/channel0-preds-trues.pkl', "br")
data = pickle.load(file)

# Pick a subset for plotting
idx = np.random.choice(np.arange(data['multi_ensemble'].shape[0]),125000)

# Get y-true
y_true = data['y_true'][idx,:]

models = ['AR',\
            'uni_cnn','multi_cnn','all_cnn',\
            'uni_lstm','multi_lstm','uni_ensemble',\
            'multi_ensemble']
RMSE = np.zeros((idx.shape[0],len(models)))

for j,i in enumerate(models):
    preds = data[i][idx,:]
    RMSE[:,j] = np.sqrt(np.mean((preds-y_true)**2,axis=1))

RMSE_df = pd.DataFrame(RMSE,columns=models)

RMSE_df = RMSE_df.melt(var_name='groups', value_name='vals')

ax = sns.violinplot(x="groups", y="vals", data=RMSE_df)


plt.show()

pdb.set_trace()
