#!/usr/bin/env python
# coding: utf-8

# In[1]:


def UMAP_clusters(BL_1, BL_2, D28_1, D28_2, directory, cells, neighbors, metric, cluster_size):
    
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
        
        
        
    
    
    os.chdir(directory)

    #BASELINE
    
    sample_BL_1 = FCMeasurement(ID='Test Sample', datafile=BL_1)
    sample_BL_2 = FCMeasurement(ID='Test Sample', datafile=BL_2)
    sample_BL = sample_BL_1.data.append(sample_BL_2.data)
    
    # RESAMPLING
    print('data loaded')
    #indexes = np.random.random_integers(len(sample_BL.data), size = (1200000,))
    indexes = random.sample(range(0, len(sample_BL)), cells)
    data_BL = sample_BL
    data_BL = data_BL.iloc[indexes,]
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
    
    #CREATE TARGET DF
    
    
    timepoints = data_BL['Timepoint'].append(data_D28['Timepoint'], ignore_index=True)
    
    #DELETE TIMEPOINTS FROM ANALYSIS DF
    
    del  data_BL['Timepoint']
    del  data_D28['Timepoint']
    
    #CREATE ANALYSIS DATAFRAME
    
    data = data_BL.append(data_D28, ignore_index=True)
    data.columns = ['CD45','CD66','HLA-DR','CD3',
                'CD64','CD34','H3','CD123','CD101',
                'CD38','CD2','Ki67','CD10','CD117',
                'CX3CR1','E3L','CD172a','CD45RA',
                'CD14','Siglec1', 'CD1C','H4K20me3',
                'CD32','CLEC12A','CD90','H3K27ac','CD16',
               'CD11C','CD33','H4','CD115','BDCA2','CD49d+',
                'H3K27me3','H3K4me3','CADM1','CD20','CD8','CD11b']
    
    #drop non cluster marker
    back_up = data
    data = data.drop(['Ki67','H3K4me3','H3K27me3','H4','H3K27ac','H4K20me3','E3L'], axis = 1)
    
    #WHOLE DATA ANALYSIS
    
    print('0/3')
    
    
    
    clusterable_embedding_1 = umap.UMAP(
        n_neighbors= neighbors,
        min_dist=0,
        n_components=2, metric = str(metric)
    ).fit_transform(data)
    print('1/3')
    
    
    
    #one percent
    
    _1 = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size= int(cluster_size*len(data)),
    ).fit_predict(clusterable_embedding_1)
    
    
    
    
    
    
    labels = [_1]
    names = ['_' + str(cluster_size*100).replace('.','').replace('10', '1')]
    
    
    
    
    print('2/3')
    
    for labels_1, percent in zip(labels, names):
        
        os.chdir(directory + '/' + str(percent))
        #UMAP PLOTS
    
        for timepoint in np.unique(timepoints):
            
            color = iter(cm.hsv(np.linspace(0, 1, len(np.unique(labels_1)*2))))

            print(' - ' + str(timepoint))
            fig, ax = plt.subplots(figsize = (15,15))
            
            tmp = (timepoints == timepoint)
            for cluster in np.unique(labels_1):


                print('   - ' + str(cluster))

                clustered = (labels_1 == cluster)

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
                    print(' - ' + str(cluster))

                    clustered = (labels_1 == cluster)


                    plt.scatter(clusterable_embedding_1[clustered][:,0],
                                clusterable_embedding_1[clustered][:,1],
                                s=0.1,
                                c = back_up[clustered][str(x)],
                                cmap='jet', norm=matplotlib.colors.LogNorm())
                plt.colorbar()
                plt.savefig(r'UMAP/other_markers/_' + str(x) + '.png', dpi=300)
                plt.close()






        print('3/3')

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
        from matplotlib.pyplot import figure
        figure(figsize=(15, 10))
        #plt.barh(means.index, means['Mean'],xerr=error)
        plt.errorbar(means['Mean'], means.index, xerr=[0]*32, fmt = 'o', ecolor = 'blue', c = 'red')
        plt.xscale('log')
        plt.xlabel('Mean intensity', fontsize = 18)
        plt.ylabel('Channels', fontsize = 18)


        plt.savefig(r'Clusters/WHOLE_markers.png', dpi=300)
        plt.close()

        print('Generating marker plots...')
        for i in np.unique(labels_1):
            print('Cluster '+ str(i))
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
            from matplotlib.pyplot import figure
            #figure(figsize=(15, 10))
            #plt.barh(means.index, means['Mean'],xerr=error)
            ax1.errorbar(means['Mean'], means.index, xerr=error, fmt = 'o', ecolor = 'blue', c = 'red')
            #plt.xscale('log')
            ax1.set_xlabel('Mean intensity', fontsize = 18)
            ax1.set_ylabel('Channels', fontsize = 18)
            ax1.set_title('Quality Control')

            #Signature

            ax2.errorbar(means['Mean'], means.index, xerr=[0]*32, fmt = 'o', ecolor = 'blue', c = 'red')
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
                print(' - '+ str(p))

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
        print('Done')


        fig, axes = plt.subplots(6, 7, figsize=(17,18), dpi=100)

        for p, ax in zip(data, axes.flatten()):

            ax.hist(data[str(p)], bins = 100, density = True, alpha = 0.6)
            sns.kdeplot(data[str(p)], ax = ax, legend = False, c = 'red')  
            #ax.set_title(str(p))
        plt.savefig(r'Clusters/whole_distrib.png', dpi=300)
        plt.close()

        print('Generating cluster plots...')
        #CLUSTER SIZES
        #CREATE DATAFRAME
        import pandas as pd
        import numpy as np
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
        print(cluster_sizes) 
        print(cluster_sizes['BL'].sum(),cluster_sizes['D28'].sum())
        #plotting
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
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

        print('Done')
        plt.savefig(r'Clusters/cluster_composition.png', dpi=300)
        plt.close()
        
        
import datetime
data_list = [r'C:\Users\mp268043\Jupyter\tests\VAC2022\BL_VAC2022_CDF059.fcs_SecondRand.fcs',
             r'C:\Users\mp268043\Jupyter\tests\VAC2022\BL_VAC2022_CDI003.fcs_SecondRand.fcs',
             r'C:\Users\mp268043\Jupyter\tests\VAC2022\D28_VAC2022_CDF059.fcs_SecondRand.fcs',
             r'C:\Users\mp268043\Jupyter\tests\VAC2022\D28_VAC2022_CDI003.fcs_SecondRand.fcs', 
             r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results',
            2000000,
            15]


metrics = [['cosine',r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Cosine'], 
           ['euclidean', r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Euclidean']]


import multiprocessing as mp


if __name__ == '__main__':
    
    p1 = mp.Process(target=UMAP_clusters, args=(data_list[0], data_list[1], data_list[2], data_list[3], 
                                                r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Cosine', 
                                                data_list[5], data_list[6], 'cosine', 1/100,))
    
    
    p2 = mp.Process(target=UMAP_clusters, args=(data_list[0], data_list[1], data_list[2], data_list[3], 
                                                r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Euclidean', 
                                                data_list[5], data_list[6], 'euclidean', 1/100,))
    

    #p3 = mp.Process(target=UMAP_clusters, args=(data_list[0], data_list[1], data_list[2], data_list[3], 
                                                #r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Cosine', 
                                                #data_list[5], data_list[6], 'cosine', 1/1000,))

    
    
    #p4 = mp.Process(target=UMAP_clusters, args=(data_list[0], data_list[1], data_list[2], data_list[3], 
                                                #r'C:\Users\mp268043\Jupyter\tests\VAC2022\Results\metrics\Euclidean', 
                                                #data_list[5], data_list[6], 'euclidean', 1/1000,))

    start = datetime.datetime.now().time()

    p1.start()
    p2.start()
    #p3.start()
    #p4.start()

    p1.join()
    p2.join()
    #p3.join()
    #p4.join()
    
    finish = datetime.datetime.now().time()
    delta = datetime.timedelta(hours=finish.hour-start.hour, minutes=finish.minute-start.minute, seconds = finish.second-start.second)
    
    print('Running time: ', delta)
    
    



    
    
    

