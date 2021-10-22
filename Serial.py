#!/usr/bin/env python
# coding: utf-8

# In[2]:


def UMAP_clusters(BL_1, BL_2, D28_1, D28_2, cells, neighbors, metric):
    
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
    import umap.umap_ as umap
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
        
        
        
    
    cwd = os.getcwd()
    os.chdir(cwd)
    
    paths = ['Results', 'Results/metrics', 'Results/metrics/' + str(metric),
             'Results/metrics/' + str(metric) + '/_1', 'Results/metrics/' + str(metric) + '/_1/UMAP','Results/metrics/' + str(metric) + '/_1/UMAP/other_markers', 'Results/metrics/' + str(metric) + '/_1/Clusters',
             'Results/metrics/' + str(metric) + '/_01', 'Results/metrics/' + str(metric) + '/_01/UMAP','Results/metrics/' + str(metric) + '/_01/UMAP/other_markers', 'Results/metrics/' + str(metric) + '/_01/Clusters']
    
    for path in paths:
                
        if os.path.exists(path):                                      #DELETES EXISTING DIRS AND RECREATE THEM
            shutil.rmtree(path)
        os.makedirs(path)

    #BASELINE
    
    sample_BL_1 = FCMeasurement(ID='Test Sample', datafile=BL_1)
    sample_BL_2 = FCMeasurement(ID='Test Sample', datafile=BL_2)
    sample_BL = sample_BL_1.data.append(sample_BL_2.data)
    
    # Resampling
    
    indexes = random.sample(range(0, len(sample_BL)), cells)
    data_BL = sample_BL
    data_BL = data_BL.iloc[indexes,]
    
    #Cleaning  
    data_BL = data_BL.drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','FileNum','Pd102Di','Rh103Di','Pd104Di',
                      'Pd105Di','Pd106Di','Pd108Di','Pd110Di','BCKG190Di',
                      'Ir191Di','Ir193Di','ArAr80Di','Xe131Di','Ce140Di',
                      'Pb208Di','I127Di','Ba138Di'], axis = 1)
    data_BL['Timepoint'] = ['BL']*len(data_BL)
    
    #D28
    
    sample_D28_1 = FCMeasurement(ID='Test Sample', datafile=D28_1)
    sample_D28_2 = FCMeasurement(ID='Test Sample', datafile=D28_2)
    sample_D28 = sample_D28_1.data.append(sample_D28_2.data)
    
    
    # RESAMPLING
    
    #indexes = np.random.random_integers(len(sample_D28.data), size = (1200000,))
    indexes = random.sample(range(0, len(sample_D28)), cells)
    data_D28 = sample_D28
    data_D28 = data_D28.iloc[indexes,]
    data_D28 = data_D28.drop(['Time','Event_length','Center','Offset','Width',
                      'Residual','FileNum','Pd102Di','Rh103Di','Pd104Di',
                      'Pd105Di','Pd106Di','Pd108Di','Pd110Di','BCKG190Di',
                      'Ir191Di','Ir193Di','ArAr80Di','Xe131Di','Ce140Di',
                      'Pb208Di','I127Di','Ba138Di'], axis = 1)
    
    data_D28['Timepoint'] = ['D28']*len(data_D28)
    
    #CREATE timepoint DF with same indices 
    
    
    timepoints = data_BL['Timepoint'].append(data_D28['Timepoint'], ignore_index=True)
    
    #DELETE TIMEPOINTS FROM ANALYSIS DF
    
    del  data_BL['Timepoint']
    del  data_D28['Timepoint']
    
    #CREATE ANALYSIS DATAFRAME with same indices as timepoint df
    
    data = data_BL.append(data_D28, ignore_index=True)
    data.columns = ['CD45','CD66','HLA-DR','CD3',
                'CD64','CD34','H3','CD123','CD101',
                'CD38','CD2','Ki67','CD10','CD117',
                'CX3CR1','E3L','CD172a','CD45RA',
                'CD14','Siglec1', 'CD1C','H4K20me3',
                'CD32','CLEC12A','CD90','H3K27ac','CD16',
               'CD11C','CD33','H4','CD115','BDCA2','CD49d+',
                'H3K27me3','H3K4me3','CADM1','CD20','CD8','CD11b']  #39d
    
    #drop non-clustering markers and keep them for later use
    back_up = data
    data = data.drop(['Ki67','H3K4me3','H3K27me3','H4','H3K27ac','H4K20me3','E3L'], axis = 1) #32d 6
    


    data = (data-data.min())/(data.max()-data.min()) #MINMAX NORMED
    
    #WHOLE DATA ANALYSIS
    
    print('Running...')
    
    #UMAP dimension reduction to 2D
    
    clusterable_embedding_1 = umap.UMAP(
        n_neighbors= neighbors,
        min_dist=0,
        n_components=2, metric = str(metric)
    ).fit_transform(data)
    
    
    
    
    #HDBSCAN clustering over UMAP embedding
    
    _1 = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size= int(0.01*len(data)),
    ).fit_predict(clusterable_embedding_1)
    
    _01 = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size= int(0.001*len(data)),
    ).fit_predict(clusterable_embedding_1)
    
    
    
    
    
    
    labels = [_1, _01]
    #names = ['_' + str(cluster_size*100).replace('.','').replace('10', '1')] #to store in folders named _1 : 1% , _01 : 0.1% , ...
    names = ['_1', '_01']
    
    
    
   
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
                plt.colorbar()
                
                path = r'UMAP/other_markers/_' + str(x)
                
                if os.path.exists(path):                                      #DELETES EXISTING DIRS AND RECREATE THEM
                    shutil.rmtree(path)
                os.makedirs(path)
                
                plt.savefig(r'UMAP/other_markers/_' + str(x) + '.png', dpi=300)
                plt.close()






        

        hover_df = pd.DataFrame(timepoints)
        hover_df.columns = ['target']

        # SUBSETS
        
        subset_BL = hover_df['target'] == 'BL'
        
        subset_D28 = hover_df['target'] == 'D28' 
        subset_whole = (hover_df['target'] == 'BL') | (hover_df['target'] == 'D28') 




        means = pd.DataFrame(data.mean()).astype('float32')
        means.set_axis(['Mean'], axis = 1, inplace = True)
        means = means.sort_values(by = 'Mean', ascending = True)
            
        error = np.std(data)
        
        figure(figsize=(15, 10))
        #plt.barh(means.index, means['Mean'],xerr=error)
        plt.errorbar(means['Mean'], means.index, xerr=[0]*32, fmt = 'o', ecolor = 'blue', c = 'red') #CHANGES WITH DIMENSIONS !!!!!!!!!
        plt.xscale('log')
        plt.xlabel('Mean intensity', fontsize = 18)
        plt.ylabel('Channels', fontsize = 18)


        plt.savefig(r'Clusters/WHOLE_markers.png', dpi=300)
        plt.close()

        files = glob.glob(r'Clusters/*')
        for f in files:
            os.remove(f)
        print('deleted previous cluster results')
        for i in np.unique(labels_1):
            
            
            
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
            #QC

            eight = (labels_1 == i)
            df_group_8 = pd.DataFrame(data.values[eight,]).astype('float32')
            markers = list(data.columns)
            df_group_8.set_axis(markers, axis=1, inplace=True)

            means = pd.DataFrame(df_group_8.mean()).astype('float32')
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

            ax2.errorbar(means['Mean'], means.index, xerr=[0]*32, fmt = 'o', ecolor = 'blue', c = 'red')  #CHANGES WITH DIMENSIONS !!!!!
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
            plt.bar('good', good, label = 'good', color = 'green')
            plt.bar('bad',bad, label = 'bad', color = 'red')
            plt.savefig(r'Clusters/' + str(i) + 'QC.png', dpi=300)
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
        
        clusters_BL = pd.DataFrame(labels_1[subset_BL,])
        clustersizes_BL = []
        clusters = []
        for i in np.unique(clusters_BL[0]):

            clustersizes_BL.append(np.shape(clusters_BL[clusters_BL[0]==i])[0])
            clusters.append(i)

        cluster_sizes = pd.DataFrame(clustersizes_BL).astype('float32')
        cluster_sizes.columns = ['BL']
        cluster_sizes['clusters'] = clusters


        clusters_D28 = pd.DataFrame(labels_1[subset_D28,])
        clustersizes_D28 = []
        for i in np.unique(clusters_D28[0]):

            clustersizes_D28.append(np.shape(clusters_D28[clusters_D28[0]==i])[0])

        cluster_sizes['D28'] = clustersizes_D28
        
        #plotting
        
        figure(figsize=(15, 10))
        width = 0.3
        plt.bar(cluster_sizes['clusters'], cluster_sizes['BL'],width, color = 'b', alpha = 0.7, label = 'Baseline')
        plt.bar(cluster_sizes['clusters']+width, cluster_sizes['D28'],width, color = 'g', alpha = 0.7, label = 'Day 28')
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
try:
    
    start = datetime.datetime.now().time()
    UMAP_clusters(r'files/BL_VAC2022_CDF059.fcs_SecondRand.fcs',
                  r'files/BL_VAC2022_CDI003.fcs_SecondRand.fcs',
                  r'files/D28_VAC2022_CDF059.fcs_SecondRand.fcs',
                  r'files/D28_VAC2022_CDI003.fcs_SecondRand.fcs',
                  5000,
                  10,
                  "euclidean")
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


except Exception as e:
    
    
    for adress in ['martin.pezous@cea.fr', 'martin.pezous-puech@live.fr']:

        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)
        mail.To = adress
        mail.Subject = 'ERROR'
        mail.Body = ''
        mail.HTMLBody = 'An error occured during analysis:\n'+ str(e) 
        mail.Send()

    
    

      



    
    
    


# In[1]:





# In[ ]:




