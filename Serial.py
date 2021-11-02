import hdbscan
import umap
import profiler
import traitement_data
from file_params import get_files


@profiler.sayen_logger
@profiler.sayen_timer
def serial(metric, animals, neighbors, cells, scaled=True):
    baseline_files, files = get_files(metric)
    print("Handling data ...")
    data, back_up, timepoints, matches, animales = traitement_data.traitement(baseline_files, files, animals, cells, scaled)
    # UMAP dimension reduction to 2D
    print('Running...')
    clusterable_embedding_1 = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=0,
        n_components=2, metric=str(metric)
    ).fit_transform(data)

    # HDBSCAN clustering over UMAP embedding
    _01 = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size=int(0.001 * len(data)),
    ).fit_predict(clusterable_embedding_1)
    labels = [_01]
    names = ['_1', '_01']

    return clusterable_embedding_1, data, back_up, timepoints, matches, animales, labels, names
