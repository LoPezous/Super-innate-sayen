from Serial import serial
from Parallel import parallel
from plotter import plot_all


def main(metric, animals, neighbors, cells, choice="serial", plot=True, scaled=True):
    if choice == "serial":
        # parameters returned by the function serial
        clusterable_embedding_1, data, back_up, timepoints, matches, animales, labels, names = serial(metric,
                                                                                                      animals,
                                                                                                      neighbors,
                                                                                                      cells,
                                                                                                      scaled)

    elif choice == "parallel":
        clusterable_embedding_1, data, back_up, timepoints, matches, animales, labels, names = parallel(metric, animals,
                                                                                                        neighbors,
                                                                                                        cells,
                                                                                                        scaled)

    else:
        raise NotImplementedError("Invalid function choice")

    if plot:
        plot_all(labels, names, metric, timepoints, data, animales, animals, clusterable_embedding_1, back_up, matches)


if __name__ == "__main__":
    metric_ = "euclidean"
    animals_ = ["R2D2"]
    neighbors_ = 10
    cells_ = 5_000
    main(metric_, animals_, neighbors_, cells_, choice="serial", plot=True, scaled=True)
