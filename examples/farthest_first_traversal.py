import click
import numpy as np
from matplotlib import pyplot as plt
from ApproximateGreedyPermutation.algorithms import farthest_first_traversal, incremental_farthest_search


@click.command()

@click.option('-n', '--num_samples', default=100,
              help='Number of samples to run.')

@click.option('-d', '--dimension', default=2,
              help='Dimensions of vectors.')

@click.option('-k', '--selection', default=10,
              help='Number of maximally distant vectors.')

@click.option('-m', '--minkowski', default=2,
              help='Which Minkowski p-norm to use.')

@click.option('-e', '--sample_edge', is_flag=True,
              help='Samples around the edge of the space ' \
                   'an avoids center points.')

def main(num_samples, dimension, selection, minkowski, sample_edge):
    
    data = np.random.normal(size=(num_samples, dimension))

    print(data.shape)

    k_inds = farthest_first_traversal(data, selection, minkowski=minkowski, sample_edge=sample_edge)

    #k_farthest, k_inds = incremental_farthest_search(data, selection)

    farthest = data[k_inds]
    mask = np.ones(num_samples, dtype=bool)
    mask[k_inds] = False
    leftover = data[mask]

    print(farthest.shape)
    print(leftover.shape)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(farthest[:, 0], farthest[:, 1], color='r', label='farthest')
    ax.scatter(leftover[:, 0], leftover[:, 1], color='b', label='leftover')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'K = {selection} farthest points')
    plt.show()

if __name__ == '__main__':
    main()
