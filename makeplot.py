# Make plot comparing models

import numpy as np
import pdb
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

file = open('./data/channel0-results.pkl', "br")
data = pickle.load(file)

pdb.set_trace()

d = np.zeros((250000*2,10))
m = np.empty((250000*2,10)).astype(object)
t = np.zeros((250000*2,10)).astype(object)

models = ['AR','multi_ensemble']
time_samps = ['t+1','t+2']
for i in models:
    d[250000*(c):250000*(c+1),0] = data[models[i]]['mae'][0] # samples
    m[250000*(c):250000*(c+1),0] = np.repeat(models[i],250000) # model
    t[250000*(c):250000*(c+1),0] = np.repeat('t+1',250000) # time sample
    

sns.boxplot(univar_cnn['t+1'])
plt.show()

pdb.set_trace()
