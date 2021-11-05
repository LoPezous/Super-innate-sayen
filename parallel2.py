#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def UMAP_clusters(animals, cells, neighbors, metric, min_sample, min_size):
    
    import FlowCytometryTools
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import glob
    from FlowCytometryTools import FCMeasurement, ThresholdGate
    import random
    from sklearn.decomposition import PCA
    #import umap.umap_ as umap
    import umap
    import hdbscan
    import umap.plot
    from collections import Counter
    from matplotlib.pyplot import figure
    import matplotlib
    from matplotlib.pyplot import cm
    from unidip import UniDip
    import scipy.stats as stat
    import datetime
    import win32com.client as win32
    import shutil
    import sys
    import re
    import traceback2 as traceback
    from more_itertools import one
    
    
   # Quality control tests used in QC section
    def unimodal(dat):
        
        dat = list(dat)       
        dat = np.msort(dat)
        intervals = UniDip(dat, alpha=0.05).run()
        if len(intervals) != 1:
        
            return (False)
        else:
            return (True)
    
    def spread(dat):
        IQR = stat.iqr(dat)
        if IQR < 200:
            return True
        else:
            return False
        
        
        
    
    cwd = sys.path[0]
    os.chdir(cwd)
    
    paths = ['Results', 'Results/metrics', 'Results/metrics/' + str(metric),
             'Results/metrics/' + str(metric) + '/_1', 'Results/metrics/' + str(metric) + '/_1/UMAP','Results/metrics/' + str(metric) + '/_1/UMAP/other_markers', 'Results/metrics/' + str(metric) + '/_1/Clusters',
             'Results/metrics/' + str(metric) + '/_01', 'Results/metrics/' + str(metric) + '/_01/UMAP','Results/metrics/' + str(metric) + '/_01/UMAP/other_markers', 'Results/metrics/' + str(metric) + '/_01/Clusters']
    
    for path in paths:
                
        if os.path.exists(path):                                      #DELETES EXISTING DIRS AND RECREATE THEM
            shutil.rmtree(path)
        os.makedirs(path)
        

    #BASELINE
    files = os.listdir('files')
    
    #ALL BASELINE FILES
    
    baseline_files = []
    
    for file in files:
        if re.match('BL', file):
            baseline_files.append(file)
        
                
    
    sample_BL = FCMeasurement(ID='Test Sample', datafile=r'files/'+ baseline_files[0])
    sample_BL = sample_BL.data
    for animal in animals:
        if animal in baseline_files[0]:
            sample_BL['animal'] = [animal]*len(sample_BL)                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    baseline_files.pop(0)
    
    
    for file in baseline_files:
        
        sample = FCMeasurement(ID='Test Sample', datafile=r'files/'+ file)
        sample = sample.data
        for animal in animals:
            if animal in file:
                sample['animal'] = [animal]*len(sample)
        sample_BL.append(sample)
        del sample
    
        
    # Resampling BASELINE
    
    indexes = random.sample(range(0, len(sample_BL)), cells)
    data_BL = sample_BL
    del sample_BL
    data_BL = data_BL.iloc[indexes,]
    
    #Cleaning BASELINE
    data_BL = data_BL.drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','FileNum','102Pd','103Rh','104Pd',
                      '105Pd','106Pd','108Pd','110Pd','190BCKG',
                      '191Ir','193Ir','80ArAr','131Xe_conta','140Ce_Beads',
                      '208Pb_Conta','127I_Conta','138Ba_Conta'], axis = 1)
    
    data_BL['Timepoint'] = ['BL']*len(data_BL)            #ADD A TIMEPOINT COLUMN
    
    
    
    # ALL D28 files
     
    matches = []
    
    variables = {}
    
    timepoints = data_BL['Timepoint']
    global animales
    animales = data_BL['animal']
    
    del  data_BL['Timepoint']
    del  data_BL['animal']
    global data
    data = data_BL
    
    
    
    for file in files:
        findall = re.findall('(D\d{2}_)', file)
        if len(findall) == 1:
            matches.append(one(findall)) 
        
    matches = np.unique(matches)
    for match in matches:
        match = match.replace('_','')
        timepoint_files = []
        for file in files:
            
            if re.search(match, str(file)):
                timepoint_files.append(file)
                
                
        #match = 'sample_' + str(match)
        key = match
        value = FCMeasurement(ID='Test Sample', datafile=r'files/'+ timepoint_files[0])            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        variables[key] = value.data
        for animal in animals:
            if animal in timepoint_files[0]:
                variables[key]['animal'] = [animal]*len(variables[key])
                
        
        timepoint_files.pop(0)
        for item in timepoint_files:
            sample = FCMeasurement(ID='Test Sample', datafile=r'files/'+ file)
            sample = sample.data
            for animal in animals:
                if animal in item:
                    sample['animal'] = [animal]*len(sample)
            variables[key] = variables[key].append(sample)
            del sample
                    
        #indexes = random.sample(range(0, len(variables[key])), cells)
        indexes = random.sample(range(0, len(variables[key])), cells)
                
                
    
                
        variables[key] = variables[key].iloc[indexes,]
    
                #cleaning D28
    
        variables[key] = variables[key].drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','FileNum','102Pd','103Rh','104Pd',
                      '105Pd','106Pd','108Pd','110Pd','190BCKG',
                      '191Ir','193Ir','80ArAr','131Xe_conta','140Ce_Beads',
                      '208Pb_Conta','127I_Conta','138Ba_Conta'], axis = 1)
    
        variables[key]['Timepoint'] = [match]*len(variables[key])
                    
   
                # RESAMPLING D28
    
    
                
    
                  #ADD A TIMEPOINT COLUMN
    
    
    #ADD OTHER TIMEPOINT CODE IF NECESSARY HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #SECTIONS TO COPY AND CHANGE: ALL {timepoint} files, Resampling {timepoint},  cleaning {timepoint}
    
    #CREATE timepoint DF 
        timepoints = timepoints.append(variables[key]['Timepoint'], ignore_index = True)
        animales = animales.append(variables[key]['animal'], ignore_index = True)
        del  variables[key]['Timepoint']
        del  variables[key]['animal']
        
    
    #DELETE TIMEPOINTS FROM ANALYSIS DF (if other timepoints were added, also delete timepoint column from these dataframes)
    
    
    
    #CREATE ANALYSIS DATAFRAME with same indices as timepoint df
        
        data = data.append(variables[key], ignore_index=True)
                
            
    data.columns = ['CD45','CD66','HLA-DR','CD3',
                'CD64','CD34','H3','CD123','CD101',
                'CD38','CD2','Ki67','CD10','CD117',
                'CX3CR1','E3L','CD172a','CD45RA',
                'CD14','Siglec1', 'CD1C','H4K20me3',
                'CD32','CLEC12A','CD90','H3K27ac','CD16',
               'CD11C','CD33','H4','CD115','BDCA2','CD49d+',
                'H3K27me3','H3K4me3','CADM1','CD20','CD8','CD11b']  #39d
    del data_BL
    #el data_D28
    #drop non-clustering markers and keep them for later use
    global back_up
    back_up = data
    data = data.drop(['Ki67','H3K4me3','H3K27me3','H4','H3K27ac','H4K20me3','E3L','CD64','CD2', 'CD45RA', 'CD20'], axis = 1) #32d 6 --> 29
    
    #DATA PER ANIMAL
  
    for animal in np.unique(animales):
        anml = (animales == animal)
        df = pd.DataFrame(data[anml])
        df.to_csv(str(animal) + '.csv')
        
    
    
    data = np.arcsinh(data)
    #data = (data-data.min())/(data.max()-data.min()) #MINMAX NORMED
                             #FLOAT 323 !!!
    #WHOLE DATA ANALYSIS
    
    print('Running...')
    
    
    #UMAP dimension reduction to 2D
    
    clusterable_embedding_1 = umap.UMAP(
        n_neighbors= neighbors,
        min_dist=0,
        n_components=2, metric = str(metric)
    ).fit_transform(data)
    
    
    
    
    #HDBSCAN clustering over UMAP embedding
    
    _01 = hdbscan.HDBSCAN(
        min_samples= min_sample,
        min_cluster_size= int(min_size*len(data)),     #0.00033
    ).fit_predict(data)
    
    
    
    
    
    
    labels = [_01]
    #names = ['_' + str(cluster_size*100).replace('.','').replace('10', '1')] #to store in folders named _1 : 1% , _01 : 0.1% , _001 : 0.001 %
    names = ['_01']
    
    
    
   
    #UMAP PLOTS
    for labels_1, percent in zip(labels, names):
        
        
        os.chdir(cwd + '/Results/metrics/' + str(metric) + '/' + str(percent))
        
    
        for timepoint in np.unique(timepoints):
            
            color = iter(cm.hsv(np.linspace(0, 1, len(np.unique(labels_1)*2)))) #choosing gradient as discrete colors depending on number of clusters

            
            fig, ax = plt.subplots(figsize = (15,15))
            
            tmp = (timepoints == timepoint) #timepoing condition boolean list
            for cluster in np.unique(labels_1):


                

                clustered = (labels_1 == cluster) #cluster condition boolean list 

                both = (tmp & clustered)

                if cluster == -1:

                    ax.scatter(clusterable_embedding_1[both][:,0],
                            clusterable_embedding_1[both][:,1],
                            s=0.1,
                            color = 'white',
                            cmap='jet',
                            label = cluster)
                    
                    ax.patch.set_facecolor('black')


                else:


                    ax.scatter(clusterable_embedding_1[both][:,0],
                                clusterable_embedding_1[both][:,1],
                                s=0.1,
                                color = next(color),
                                cmap='jet',
                                label = cluster)
                    
                    ax.patch.set_facecolor('black')

            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", markerscale = 15.0, ncol = 3)
            

            
                
            plt.savefig(r'UMAP/' + str(timepoint) + '.png', bbox_inches='tight', dpi=300)
            plt.close()


        #UMAP markers

        path = r'UMAP/other_markers'
                
        if os.path.exists(path):                                      #DELETES EXISTING DIRS AND RECREATE THEM
            shutil.rmtree(path)
        os.makedirs(path)

        for x in back_up:
            #for timepoint in np.unique(timepoints):
                #print(' - ' + str(timepoint))
                fig, ax = plt.subplots(figsize = (12,12))
                #tmp = (timepoints == timepoint)
                for cluster in np.unique(labels_1):
                    

                    clustered = (labels_1 == cluster)


                    plt.scatter(clusterable_embedding_1[clustered][:,0],
                                clusterable_embedding_1[clustered][:,1],
                                s=0.1,
                                c = back_up[clustered][str(x)],
                                cmap='jet', norm=matplotlib.colors.LogNorm())
                    #plt.clim(0.1,1000)
                    plt.clim(vmin = 0.1, vmax = back_up[clustered][str(x)].max())
                plt.colorbar()
                
                
                
                plt.savefig(r'UMAP/other_markers/_' + str(x) + '.png', dpi=300)
                plt.close()






        
        
        hover_df = pd.DataFrame(timepoints)
        hover_df.columns = ['target']

        # SUBSETS
        
        subsets = {}
        
        for match in matches:
            match = match.replace('_','')
            key = 'subset_' + match
            value = hover_df['target'] == match
            subsets[key] = value
            
        
        subset_BL = hover_df['target'] == 'BL'
        
        #subset_D28 = hover_df['target'] == 'D28' 
        #subset_whole = (hover_df['target'] == 'BL') | (hover_df['target'] == 'D28') 




        means = pd.DataFrame(data.mean())
        means.set_axis(['Mean'], axis = 1, inplace = True)
        means = means.sort_values(by = 'Mean', ascending = True)
            
        error = np.std(data)
        
        figure(figsize=(15, 10))
        #plt.barh(means.index, means['Mean'],xerr=error)
        plt.errorbar(means['Mean'], means.index, xerr=[0]*28, fmt = 'o', ecolor = 'blue', c = 'red') #CHANGES WITH DIMENSIONS !!!!!!!!!
        plt.xscale('log')
        plt.xlabel('Mean intensity', fontsize = 18)
        plt.ylabel('Channels', fontsize = 18)


        plt.savefig(r'Clusters/WHOLE_markers.png', dpi=300)
        plt.close()

        files = glob.glob(r'Clusters/*')
        for f in files:
            os.remove(f)
        print('deleted previous cluster results')
        
        good_clusters = []
        bad_clusters = []
        for i in np.unique(labels_1):
            
            
            
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
            #QC

            eight = (labels_1 == i)
            df_group_8 = pd.DataFrame(data.values[eight,])
            markers = list(data.columns)
            df_group_8.set_axis(markers, axis=1, inplace=True)

            means = pd.DataFrame(df_group_8.mean())
            means.set_axis(['Mean'], axis = 1, inplace = True)
            means = means.sort_values(by = 'Mean', ascending = True)

            error = np.std(df_group_8)
            
            #figure(figsize=(15, 10))
            #plt.barh(means.index, means['Mean'],xerr=error)
            ax1.errorbar(means['Mean'], means.index, xerr=error, fmt = 'o', ecolor = 'blue', c = 'red')
            #plt.xscale('log')
            ax1.set_xlabel('Mean intensity', fontsize = 18)
            ax1.set_ylabel('Channels', fontsize = 18)
            ax1.set_title('Quality Control')

            #Signature

            ax2.errorbar(means['Mean'], means.index, xerr=[0]*28, fmt = 'o', ecolor = 'blue', c = 'red')  #CHANGES WITH DIMENSIONS !!!!!
            ax2.set_xscale('log')
            ax2.set_xlabel('Mean intensity (log)', fontsize = 18)

            ax1.set_title('Signature')

            
                
            plt.savefig(r'Clusters/' + str(i) + '_markers.png', dpi=300)
            plt.close()

        #distribution
            
            good_markers = []
            bad_markers = []
            fig, axes = plt.subplots(6, 7, figsize=(17,18), dpi=100)

            for p, ax in zip(data, axes.flatten()):
                

                ax.hist(df_group_8[str(p)], bins = 100, density = True, alpha = 0.6)
                if (unimodal(df_group_8[str(p)]) == True) & (spread(df_group_8[str(p)]) == True):
                    c = 'green'
                    good_markers.append(True)
                else:
                    c = 'red'
                    bad_markers.append(False)
                    
                sns.kdeplot(df_group_8[str(p)], ax = ax, legend = False, c = c)
                
                #ax.set_title(str(p))

            plt.savefig(r'Clusters/' + str(i) + '_distrib.png', dpi=300)
            plt.close()
            
            #good cluster plot
            good = len(good_markers)/(len(good_markers) + len(bad_markers))
            bad = len(bad_markers)/(len(good_markers) + len(bad_markers))
            
            if good > 0.7:
                good_clusters.append(str(i))
            else:
                bad_clusters.append(str(i))
            
        
        ok = len(good_clusters)/(len(good_clusters) + len(bad_clusters))
        not_ok = len(bad_clusters)/(len(good_clusters) + len(bad_clusters))
        
        plt.bar('good_clusters', ok, label = 'good clusters', color = 'green')
        plt.bar('bad_clusters', not_ok, label = 'bad clusters', color = 'red')
        plt.title('Proportion of clusters where all markers pass the cluster QC')
        plt.savefig(r'Clusters/' + 'QC.png', dpi=300)
        plt.close()
        


        fig, axes = plt.subplots(6, 7, figsize=(17,18), dpi=100)

        for p, ax in zip(data, axes.flatten()):

            ax.hist(data[str(p)], bins = 100, density = True, alpha = 0.6)
            sns.kdeplot(data[str(p)], ax = ax, legend = False, c = 'red')  
            #ax.set_title(str(p))
        plt.savefig(r'Clusters/whole_distrib.png', dpi=300)
        plt.close()

        
        #CLUSTER SIZES
        #CREATE DATAFRAME
        df_list = []
        for animal in animals:
            anml = (animales == animal)
            anml_clusters = pd.DataFrame(labels_1[anml,])
            clusters = []
            sizes = []
            for x in np.unique(anml_clusters[0]):
                
                sizes.append(np.shape((anml_clusters[anml_clusters[0] == x]))[0])
                clusters.append(x)
            anml_clusters = pd.DataFrame(sizes)
            anml_clusters['cluster'] = clusters
            anml_clusters.columns = [str(animal), 'cluster']
            anml_clusters.to_csv(str(animal) + '_clusters.csv', index = False)
            df_list.append(str(animal) + '_clusters.csv')
        df = pd.read_csv(df_list[0])
        os.remove(df_list[0])
        df_list.pop(0)
        for n in df_list:
            df_n = pd.read_csv(n)
            monkey = n.split('_')[0]
            df[monkey] = df_n[monkey]
        
        df.to_csv('clusters_per_monkey.csv', index = False)
        for m in df_list:
            os.remove(m)
        
        
        clusters_BL = pd.DataFrame(labels_1[subset_BL,])
        clustersizes_BL = []
        clusters = []
        for i in np.unique(clusters_BL[0]):

            clustersizes_BL.append(np.shape(clusters_BL[clusters_BL[0]==i])[0])
            clusters.append(i)
        global cluster_sizes
        cluster_sizes = pd.DataFrame(clustersizes_BL)
        cluster_sizes.columns = ['BL']
        cluster_sizes['clusters'] = clusters

        for subset_D28 in subsets:
            
            clusters_D28 = pd.DataFrame(labels_1[subsets[subset_D28],])
            
            clustersizes_D28 = []
            for i in np.unique(clusters_D28[0]):
                if any(clusters_D28.isin(i)): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    clustersizes_D28.append(np.shape(clusters_D28[clusters_D28[0]==i])[0])
                else:
                    clustersizes_D28.append(0)
                    
            for match in matches:
                match = match.replace('_','')
                cluster_sizes[match] = clustersizes_D28 
        
        #plotting
        
        figure(figsize=(15, 10))
        width = 0.3
        plt.bar(cluster_sizes['clusters'], cluster_sizes['BL'],width, color = 'b', alpha = 0.7, label = 'Baseline')
        for match in matches:
            match = match.replace('_','')
            plt.bar(cluster_sizes['clusters']+width, cluster_sizes[str(match)],width, color = 'g', alpha = 0.7, label = str(match))
        #plt.yscale("log")


        plt.xticks(cluster_sizes['clusters'] + width / 2, cluster_sizes['clusters'])

        plt.legend()
        plt.xlabel('Clusters', fontsize = 15)
        plt.ylabel('Cell count', fontsize = 15)
        plt.title('Evolution of cluster sizes', fontsize = 18)

        
        plt.savefig(r'Clusters/cluster_composition.png', dpi=300)
        plt.close()
        print('Done')
        







import datetime
import win32com.client as win32
#try:
    
start = datetime.datetime.now().time()
UMAP_clusters(['CDF059','CDI003'],
              5000, 
              10,
              'euclidean',
              20,
              0.00033)
              
                
finish = datetime.datetime.now().time()
delta = datetime.timedelta(hours=finish.hour-start.hour, minutes=finish.minute-start.minute, seconds = finish.second-start.second)
    
for adress in ['martin.pezous@cea.fr', 'martin.pezous-puech@live.fr']:

    outlook = win32.Dispatch('Outlook.Application')
    mail = outlook.CreateItem(0)
    mail.To = adress
    mail.Subject = 'Analysis: Done'
    mail.Body = ''
    mail.HTMLBody = 'The analysis was succesful and ran for ' + str(delta) + ' (Days : Hours : Minutes : seconds)'
    mail.Send()


#except Exception as e:
"""
    
    for adress in ['martin.pezous@cea.fr', 'martin.pezous-puech@live.fr']:

        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)
        mail.To = adress
        mail.Subject = 'ERROR'
        mail.Body = ''
        mail.HTMLBody = 'An error occured during analysis:\n'+ str(e) 
        mail.Send()
    
"""    

      



    
    
    


# In[2]:


for x in back_up:
    print(back_up[str(x)].max())


# In[3]:


matches


# In[ ]:




