import numpy as np
from scipy import misc
import scipy.stats
import logging
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import progressbar
import maxflow
from maxflow.examples_utils import plot_graph_2d


class GMM:
    def __init__(self, prob):
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
    logging.info("Calculate unary energy functions.")
    ysize, xsize, _ = img.shape
    unary = np.empty((ysize, xsize, 2))

    with progressbar.ProgressBar(max_value=xsize, redirect_stdout=True) as progress:
        for x in range(xsize):
            for y in range(ysize):
                unary[y, x, 0] = -np.log(bg.prob(img[y, x]))
                unary[y, x, 1] = -np.log(fg.prob(img[y, x]))
            progress.update(x)

    return unary


def pairwiseenergy(img):
    logging.info("Calculate pairwise energy functions.")
    ysize, xsize, _ = img.shape
    pairwise = np.array()

    l = 0.5
    w = 1.0

    with progressbar.ProgressBar(max_value=xsize, redirect_stdout=True) as progress:
        for x in range(xsize):
            for y in range(ysize):
                pass  # if np.exp(-l * np.linalg.norm())
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

    unaries = unaryenergy(fg, bg, img)

    ysize, xsize, _ = img.shape

    g = maxflow.GraphFloat()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    g.add_grid_edges(nodeids, structure=structure, symmetric=True)

    # Source and sink
    g.add_grid_tedges(nodeids, 0, 2)

    plot_graph_2d(g, nodeids.shape)

    logging.info("Save image.")
    img = img.astype(np.uint8)

    plt.imshow(img)
    plt.show()
    plt.imsave("banana_out", img)


if __name__ == '__main__':
    main()
