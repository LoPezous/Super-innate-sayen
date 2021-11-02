import os
import shutil
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import glob
from utils import unimodal_bool, spread_bool
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from file_params import cwd


def plot_all(labels, names, metric, timepoints, data, animales, animals, clusterable_embedding_1=None, back_up=None,
             matches=None):
    # UMAP PLOTS
    for labels_1, percent in zip(labels, names):

        os.chdir(cwd + '/Results/metrics/' + str(metric) + '/' + str(percent))

        for timepoint in np.unique(timepoints):

            color = iter(cm.hsv(np.linspace(0, 1, len(np.unique(
                labels_1) * 2))))  # choosing gradient as discrete colors depending on number of clusters

            fig, ax = plt.subplots(figsize=(15, 15))

            tmp = (timepoints == timepoint)  # timepoing condition boolean list
            for cluster in np.unique(labels_1):

                clustered = (labels_1 == cluster)  # cluster condition boolean list

                both = (tmp & clustered)

                if cluster == -1:

                    ax.scatter(clusterable_embedding_1[both][:, 0],
                               clusterable_embedding_1[both][:, 1],
                               s=0.1,
                               color='white',
                               cmap='jet',
                               label=cluster)

                    ax.patch.set_facecolor('black')

                else:

                    ax.scatter(clusterable_embedding_1[both][:, 0],
                               clusterable_embedding_1[both][:, 1],
                               s=0.1,
                               color=next(color),
                               cmap='jet',
                               label=cluster)

                    ax.patch.set_facecolor('black')

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", markerscale=15.0, ncol=3)

            plt.savefig(r'UMAP/' + str(timepoint) + '.png', bbox_inches='tight', dpi=300)
            plt.close()

        # UMAP markers

        path = r'UMAP/other_markers'

        if os.path.exists(path):  # DELETES EXISTING DIRS AND RECREATE THEM
            shutil.rmtree(path)
        os.makedirs(path)

        for x in back_up:
            # for timepoint in np.unique(timepoints):
            # print(' - ' + str(timepoint))
            fig, ax = plt.subplots(figsize=(12, 12))
            # tmp = (timepoints == timepoint)
            for cluster in np.unique(labels_1):
                clustered = (labels_1 == cluster)

                plt.scatter(clusterable_embedding_1[clustered][:, 0],
                            clusterable_embedding_1[clustered][:, 1],
                            s=0.1,
                            c=back_up[clustered][str(x)],
                            cmap='jet', norm=matplotlib.colors.LogNorm())
            plt.colorbar()

            plt.savefig(r'UMAP/other_markers/_' + str(x) + '.png', dpi=300)
            plt.close()

        hover_df = pd.DataFrame(timepoints)
        hover_df.columns = ['target']

        # SUBSETS

        subsets = {}

        for match in matches:
            match = match.replace('_', '')
            key = 'subset_' + match
            value = hover_df['target'] == match
            subsets[key] = value

        subset_BL = hover_df['target'] == 'BL'

        # subset_D28 = hover_df['target'] == 'D28'
        # subset_whole = (hover_df['target'] == 'BL') | (hover_df['target'] == 'D28')

        means = pd.DataFrame(data.mean())
        means.set_axis(['Mean'], axis=1, inplace=True)
        means = means.sort_values(by='Mean', ascending=True)

        error = np.std(data)

        plt.figure(figsize=(15, 10))
        # plt.barh(means.index, means['Mean'],xerr=error)
        plt.errorbar(means['Mean'], means.index, xerr=[0] * 29, fmt='o', ecolor='blue',
                     c='red')  # CHANGES WITH DIMENSIONS !!!!!!!!!
        plt.xscale('log')
        plt.xlabel('Mean intensity', fontsize=18)
        plt.ylabel('Channels', fontsize=18)

        plt.savefig(r'Clusters/WHOLE_markers.png', dpi=300)
        plt.close()

        files = glob.glob(r'Clusters/*')
        for f in files:
            os.remove(f)
        print('deleted previous cluster results')
        for i in np.unique(labels_1):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            # QC

            eight = (labels_1 == i)
            df_group_8 = pd.DataFrame(data.values[eight, ])
            markers = list(data.columns)
            df_group_8.set_axis(markers, axis=1, inplace=True)

            means = pd.DataFrame(df_group_8.mean())
            means.set_axis(['Mean'], axis=1, inplace=True)
            means = means.sort_values(by='Mean', ascending=True)

            error = np.std(df_group_8)

            # figure(figsize=(15, 10))
            # plt.barh(means.index, means['Mean'],xerr=error)
            ax1.errorbar(means['Mean'], means.index, xerr=error, fmt='o', ecolor='blue', c='red')
            # plt.xscale('log')
            ax1.set_xlabel('Mean intensity', fontsize=18)
            ax1.set_ylabel('Channels', fontsize=18)
            ax1.set_title('Quality Control')

            # Signature

            ax2.errorbar(means['Mean'], means.index, xerr=[0] * 29, fmt='o', ecolor='blue',
                         c='red')  # CHANGES WITH DIMENSIONS !!!!!
            ax2.set_xscale('log')
            ax2.set_xlabel('Mean intensity (log)', fontsize=18)

            ax1.set_title('Signature')

            plt.savefig(r'Clusters/' + str(i) + '_markers.png', dpi=300)
            plt.close()

            # distribution

            good_markers = []
            bad_markers = []
            fig, axes = plt.subplots(6, 7, figsize=(17, 18), dpi=100)

            for p, ax in zip(data, axes.flatten()):

                ax.hist(df_group_8[str(p)], bins=100, density=True, alpha=0.6)
                # on peut enlever les is True. Par défaut si a est un booléen
                # if a: donne if True/False, rentre dans le if si a is True -> if True
                if (unimodal_bool(df_group_8[str(p)]) is True) & (spread_bool(df_group_8[str(p)]) is True):
                    c = 'green'
                    good_markers.append(True)
                else:
                    c = 'red'
                    bad_markers.append(False)

                sns.kdeplot(df_group_8[str(p)], ax=ax, legend=False, c=c)

                # ax.set_title(str(p))

            plt.savefig(r'Clusters/' + str(i) + '_distrib.png', dpi=300)
            plt.close()

            # good cluster plot
            good = len(good_markers) / (len(good_markers) + len(bad_markers))
            bad = len(bad_markers) / (len(good_markers) + len(bad_markers))
            plt.bar('good', good, label='good', color='green')
            plt.bar('bad', bad, label='bad', color='red')
            plt.savefig(r'Clusters/' + str(i) + 'QC.png', dpi=300)
            plt.close()

        fig, axes = plt.subplots(6, 7, figsize=(17, 18), dpi=100)

        for p, ax in zip(data, axes.flatten()):
            ax.hist(data[str(p)], bins=100, density=True, alpha=0.6)
            sns.kdeplot(data[str(p)], ax=ax, legend=False, c='red')
            # ax.set_title(str(p))
        plt.savefig(r'Clusters/whole_distrib.png', dpi=300)
        plt.close()

        # CLUSTER SIZES
        # CREATE DATAFRAME
        df_list = []
        for animal in animals:
            anml = (animales == animal)
            anml_clusters = pd.DataFrame(labels_1[anml, ])
            clusters = []
            sizes = []
            for x in np.unique(anml_clusters[0]):
                sizes.append(np.shape((anml_clusters[anml_clusters[0] == x]))[0])
                clusters.append(x)
            anml_clusters = pd.DataFrame(sizes)
            anml_clusters['cluster'] = clusters
            anml_clusters.columns = [str(animal), 'cluster']
            anml_clusters.to_csv(str(animal) + '_clusters.csv', index=False)
            df_list.append(str(animal) + '_clusters.csv')
        df = pd.read_csv(df_list[0])
        os.remove(df_list[0])
        df_list.pop(0)
        for n in df_list:
            df_n = pd.read_csv(n)
            monkey = n.split('_')[0]
            df[monkey] = df_n[monkey]

        df.to_csv('clusters_per_monkey.csv', index=False)
        for m in df_list:
            os.remove(m)

        clusters_BL = pd.DataFrame(labels_1[subset_BL, ])
        clustersizes_BL = []
        clusters = []
        for i in np.unique(clusters_BL[0]):
            clustersizes_BL.append(np.shape(clusters_BL[clusters_BL[0] == i])[0])
            clusters.append(i)

        cluster_sizes = pd.DataFrame(clustersizes_BL)
        cluster_sizes.columns = ['BL']
        cluster_sizes['clusters'] = clusters

        for subset_D28 in subsets:

            clusters_D28 = pd.DataFrame(labels_1[subsets[subset_D28], ])

            clustersizes_D28 = []
            for i in np.unique(clusters_D28[0]):
                clustersizes_D28.append(np.shape(clusters_D28[clusters_D28[0] == i])[0])
            for match in matches:
                match = match.replace('_', '')
                cluster_sizes[match] = clustersizes_D28

        # plotting

        plt.figure(figsize=(15, 10))
        width = 0.3
        plt.bar(cluster_sizes['clusters'], cluster_sizes['BL'], width, color='b', alpha=0.7, label='Baseline')
        for match in matches:
            match = match.replace('_', '')
            plt.bar(cluster_sizes['clusters'] + width, cluster_sizes[str(match)], width, color='g', alpha=0.7,
                    label=str(match))
        # plt.yscale("log")

        plt.xticks(cluster_sizes['clusters'] + width / 2, cluster_sizes['clusters'])

        plt.legend()
        plt.xlabel('Clusters', fontsize=15)
        plt.ylabel('Cell count', fontsize=15)
        plt.title('Evolution of cluster sizes', fontsize=18)

        plt.savefig(r'Clusters/cluster_composition.png', dpi=300)
        plt.close()
        print('Done')
