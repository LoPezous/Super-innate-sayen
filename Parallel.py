import hdbscan
import umap
import profiler
import traitement_data
from file_params import get_files


# ALL BASELINE FILES

@profiler.sayen_logger
@profiler.sayen_timer
def parallel(metric, animals, neighbors, cells):
    baseline_files, files = get_files(metric)
    print("Handling data ...")
    data, back_up, timepoints, matches, animales = traitement_data.traitement(baseline_files, files, animals, cells)

    print('Running ...')
    # TODO: Transformer data en tab Numpy, et changer le type (float32 ou float16?)
    # Utiliser les types Numpy qui sont compatibles avec le C

    # UMAP dimension reduction to 2D
    clusterable_embedding_1 = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=0,
        n_components=2, metric=str(metric)
    ).fit_transform(data)

    _01 = hdbscan.HDBSCAN(
        min_samples=20,
        min_cluster_size=int(0.001 * len(data)),
    ).fit_predict(data)

    # df1 = pd.DataFrame(_01)
    # print(df1.memory_usage(index=True).sum())

    labels = [_01]
    names = ['_1', '_01']

    return clusterable_embedding_1, data, back_up, timepoints, matches, animales, labels, names

