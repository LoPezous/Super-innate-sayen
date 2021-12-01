#!/usr/bin/env python
# coding: utf-8

# In[1]:


import FlowCytometryTools
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from FlowCytometryTools import FCMeasurement, ThresholdGate
import random
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import hdbscan
import umap.plot
from collections import Counter
from matplotlib.pyplot import figure
import matplotlib
from matplotlib.pyplot import cm
from unidip import UniDip
import scipy.stats as stat

def unimodal(dat):
        
        dat = list(dat)       
        dat = np.msort(dat)
        intervals = UniDip(dat, alpha=0.05, ntrials=1).run()
        return len(intervals)
    
def spread(dat):
    IQR = stat.iqr(dat)
    return IQR
    
good_markers = []
bad_markers = []    
sizes = []
os.chdir(r'C:\Users\mp268043\Desktop\QC\files')
for file_BL, file_D28 in zip(os.listdir(r'C:\Users\mp268043\Desktop\QC\files\BL'), os.listdir(r'C:\Users\mp268043\Desktop\QC\files\D28')) :
    print('baseline: ' + file_BL + ' day 28: ' + file_D28)
    
    BL = FCMeasurement(ID='BL', datafile=r'BL/' + file_BL)
    BL = BL.data
    
    D28 = FCMeasurement(ID='D28', datafile=r'D28/' + file_D28)
    D28 = D28.data
    
    indexes = random.sample(range(0, min(len(BL), len(D28))), min(len(BL), len(D28)))
    
    #indexes = random.sample(range(0, min(len(BL), len(D28))), 20000)
    sizes.append(len(indexes))
    BL = BL.iloc[indexes,]
    D28 = D28.iloc[indexes,]
    
    BL = BL.drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','File_Number','102Pd','103Rh','104Pd',
                      '105Pd','106Pd','108Pd','110Pd','190BCKG',
                      '191Ir','193Ir','80ArAr','131Xe_conta','140Ce_Beads',
                      '208Pb_Conta','127I_Conta','138Ba_Conta'], axis = 1)
    
    D28 = D28.drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','File_Number','102Pd','103Rh','104Pd',
                      '105Pd','106Pd','108Pd','110Pd','190BCKG',
                      '191Ir','193Ir','80ArAr','131Xe_conta','140Ce_Beads',
                      '208Pb_Conta','127I_Conta','138Ba_Conta'], axis = 1)
    
    
    for mbl, md28 in zip(BL, D28):
        print(' - '+ str(mbl))

    
        if (unimodal(BL[str(mbl)]) == unimodal(BL[str(md28)])) & ((spread(BL[str(mbl)])<6*spread(D28[str(md28)])) & (spread(D28[str(mbl)]) < 6*spread(BL[str(md28)]))) & ((BL[str(mbl)].mean() < 6*D28[str(md28)].mean()) & (D28[str(md28)].mean() < 6*BL[str(mbl)].mean())):
            
            if mbl in bad_markers:
                print('bad')
            else:
            
                try:
                    good_markers.remove(mbl)
                except ValueError:
                    pass    
                good_markers.append(mbl)
                print('good')
        else:
            try:
                bad_markers.remove(mbl)
            except ValueError:
                pass
            
            try:
                good_markers.remove(mbl)
            except ValueError:
                pass
            bad_markers.append(mbl)
            print('bad')
    
    fig, axes = plt.subplots(6, 7, figsize=(17,18), dpi=100)    
    for p, ax in zip(BL, axes.flatten()):
        if p in bad_markers:
            colors = ['red','red']
        else:
            colors = ['blue', 'green']
        ax.hist(BL[str(p)], bins = 100, density = True, alpha = 0.6)
        ax.hist(D28[str(p)], bins = 100, density = True, alpha = 0.6)
        sns.kdeplot(BL[str(p)], ax = ax, legend = False, c = colors[0])  
        sns.kdeplot(D28[str(p)], ax = ax, legend = False, c = colors[1])
        #ax.set_title(str(p))
    plt.savefig(str(file_BL)+ str(file_D28) +'_'+ r'distrib.png', dpi=300)
    plt.close()
                
textfile = open("QC_results.txt", "w")
textfile.write('GOOD:' + "\n")
for element in np.unique(good_markers):
    
    textfile.write(element + "\n")
textfile.write("\n")   
textfile.write('BAD:' + "\n")

for element in np.unique(bad_markers):
    textfile.write(element + "\n")

textfile.write("\n")
textfile.write("N CELLS:")
textfile.write("\n")
textfile.write(str(sizes))
textfile.write("\n")
textfile.write("PARAMETERS:\means: 3* / spread: 3*")   #CHANGE 
textfile.close()
   
fig, axes = plt.subplots(6, 7, figsize=(17,18), dpi=100)



# In[63]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




