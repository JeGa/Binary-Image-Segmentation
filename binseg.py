import numpy as np
from scipy import misc
import scipy.stats
import logging
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import progressbar
import maxflow
from examples_utils import plot_graph_2d


class GMM:
    def __init__(self, prob):
        """
        Arguments:
            prob: List with mu, sigma and mix.
        """
        self.mu = prob[0]
        self.sigma = prob[1]
        self.mix = prob[2]

        self.K = self.mu.shape[0]

        self.pdfs = []
        for k in range(self.K):
            self.pdfs.append(scipy.stats.multivariate_normal(self.mu[k], self.sigma[k]))

    def prob(self, x):
        prob = 0
        for k in range(self.K):
            prob += self.mix[k] * self.pdfs[k].pdf(x)
        return prob


def unaryenergy(fg, bg, img):
    """
    Arguments:
        fg: Foreground GMM
        bg: Background GMM
        img: Image array (RGB)
    """
    logging.info("Calculate unary energy functions.")
    ysize, xsize, _ = img.shape
    unary = np.empty((ysize, xsize, 3))

    with progressbar.ProgressBar(max_value=xsize, redirect_stdout=True) as progress:
        for x in range(xsize):
            for y in range(ysize):
                # Background
                unary[y, x, 0] = -np.log(bg.prob(img[y, x]))
                # Foreground
                unary[y, x, 1] = -np.log(fg.prob(img[y, x]))

                # Assign labels
                if unary[y, x, 0] < unary[y, x, 1]:
                    unary[y, x, 2] = 0  # Background
                else:
                    unary[y, x, 2] = 1  # Foreground
            progress.update(x)

    return unary


def pairwiseenergy(unaries, img):
    logging.info("Calculate pairwise energy functions.")
    ysize, xsize, _ = img.shape
    pairwise = np.array()

    l = 0.5
    w = 1.0

    with progressbar.ProgressBar(max_value=xsize, redirect_stdout=True) as progress:
        for x in range(xsize):
            for y in range(ysize):
                pass
                # if unaries[y, x, 3] ==
                #    np.exp(-l * np.linalg.norm())
                # else:
                #    pass
            progress.update(x)

    return pairwise


def readprobfile(filename):
    file = np.load(filename)
    mu = file["arr_0"]
    sigma = file["arr_1"]
    mix = file["arr_2"]

    return [mu, sigma, mix]


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("Read GMM for unaries.")
    probf = readprobfile("prob_foreground.npz")
    probb = readprobfile("prob_background.npz")

    fg = GMM(probf)
    bg = GMM(probb)

    logging.info("Read image.")
    img = misc.imread("banana3.png")
    img = np.array(img, dtype=np.float64)

    # unaries = unaryenergy(fg, bg, img)
    unaries = np.load("unary.npy")
    # np.save("unary", unaries)

    ysize, xsize, _ = img.shape

    def pairwiseenergy(y1, x1, y2, x2):
        nonlocal unaries
        nonlocal img

        l = 0.005
        w = 110

        if unaries[y1, x1, 2] != unaries[y2, x2, 2]:
            delta = 1
        else:
            delta = 0

        # Not same label
        energy = w * np.exp(-l * np.power(np.linalg.norm(img[y1, x1] - img[y2, x2], ord=2), 2)) * delta
        #a = (np.linalg.norm(img[y1, x1] - img[y2, x2], ord=2))
        return energy

    g = maxflow.GraphFloat()
    nodeids = g.add_grid_nodes((ysize, xsize))

    for y in range(ysize - 1):
        for x in range(xsize - 1):
            e_right = pairwiseenergy(y, x, y, x + 1)
            if e_right >= 10.0:
                a = 2
            g.add_edge(nodeids[y, x], nodeids[y, x + 1], e_right, e_right)

            e_down = pairwiseenergy(y, x, y + 1, x)
            g.add_edge(nodeids[y, x], nodeids[y + 1, x], e_down, e_down)

            # Source, sink
            g.add_tedge(nodeids[y, x], unaries[y, x, 0], unaries[y, x, 1])

    for y in range(ysize - 1):
        e_down = pairwiseenergy(y, xsize - 1, y + 1, xsize - 1)
        g.add_edge(nodeids[y, xsize - 1], nodeids[y + 1, xsize - 1], e_down, e_down)

        g.add_tedge(nodeids[y, xsize - 1], unaries[y, xsize - 1, 0], unaries[y, xsize - 1, 1])

    for x in range(xsize - 1):
        e_right = pairwiseenergy(ysize - 1, x, ysize - 1, x + 1)

        g.add_edge(nodeids[ysize - 1, x], nodeids[ysize - 1, x + 1], e_right, e_right)
        g.add_tedge(nodeids[ysize - 1, x], unaries[ysize - 1, x + 1, 0], unaries[ysize - 1, x, 1])
    g.add_tedge(nodeids[ysize - 1, xsize - 1], unaries[ysize - 1, xsize - 1, 0], unaries[ysize - 1, xsize - 1, 1])

    # g = maxflow.Graph[float]()
    # nodeids = g.add_grid_nodes((5, 5))
    #
    # # Edges pointing backwards (left, left up and left down) with infinite
    # # capacity
    # structure = np.array([[0, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 0]
    #                       ])
    # g.add_grid_edges(nodeids, structure=structure, symmetric=False)
    #
    # # Set a few arbitrary weights
    # weights = np.array([[100, 110, 120, 130, 140]]).T + np.array([0, 2, 4, 6, 8])
    #
    # print(weights)
    #
    # structure = np.zeros((3, 3))
    # structure[1, 2] = 1
    # g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # plot_graph_2d(g, nodeids.shape)

    logging.info("Calculate max flow.")
    g.maxflow()

    for y in range(ysize):
        for x in range(xsize):
            if g.get_segment(nodeids[y, x]):
                img[y, x] = np.array([0, 0, 0])
            else:
                img[y, x] = np.array([255, 255, 0])

    logging.info("Save image.")
    img = img.astype(np.uint8)

    plt.imshow(img)
    plt.show()
    plt.imsave("banana_out", img)


if __name__ == '__main__':
    main()
