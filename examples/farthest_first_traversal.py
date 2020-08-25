import click
import numpy as np
from matplotlib import pyplot as plt
from DiverseSMILES.algorithms import farthest_first_traversal, mitchells_best_candidate


@click.command()
@click.option('-n', '--num_samples', default=100,
              help='Number of samples to run.')
@click.option('-d', '--dimension', default=2,
              help='Dimensions of vectors.')
@click.option('-k', '--selection', default=10,
              help='Number of maximally distant vectors.')
@click.option('-m', '--minkowski', default=2,
              help='Which Minkowski p-norm to use.')
@click.option('-a', '--algorithm', default='fft',
              help='Which algorithm to use [fft, mbc]')
@click.option('-c', '--candidates', default=10,
              help='How many candidates to use in mbc')
@click.option('-e', '--sample_edge', is_flag=True,
              help='Samples around the edge of the space '
                   'an avoids center points.')
def main(num_samples, dimension, selection, minkowski, algorithm, candidates, sample_edge):

    data = np.random.normal(size=(num_samples, dimension))

    print(data.shape)
    print(sample_edge)

    if algorithm == 'fft':
        k_inds = farthest_first_traversal(data, selection, minkowski=minkowski, sample_edge=sample_edge)
    elif algorithm == 'mbc':
        k_inds = mitchells_best_candidate(data, selection, minkowski=minkowski, candidates=candidates)

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
