import numpy as np
from scipy import misc
import scipy.stats
import logging
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import progressbar
import maxflow
import os


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

    Returns:
        Array A (ysize, xsize, 3) with
            A[y, x, 0] = energy background
            A[y, x, 1] = energy foreground
            A[y, x, 2] = label
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


def readprobfile(filename):
    file = np.load(filename)
    mu = file["arr_0"]
    sigma = file["arr_1"]
    mix = file["arr_2"]

    return [mu, sigma, mix]


def generateunaries(img):
    logging.info("Read GMM for unaries.")
    probf = readprobfile("prob_foreground.npz")
    probb = readprobfile("prob_background.npz")

    fg = GMM(probf)
    bg = GMM(probb)

    logging.info("Generate unaries.")
    unaries = unaryenergy(fg, bg, img)
    np.save("unary", unaries)


class Node:
    def __init__(self, nodeid, y, x):
        self.nodeid = nodeid
        self.y = y
        self.x = x


class Nodegrid:
    def __init__(self, ysize, xsize):
        self.g = maxflow.GraphFloat()

        self.nodeids = self.g.add_grid_nodes((ysize, xsize))

        self.ysize = ysize
        self.xsize = xsize

    def loop(self, edgecallback, nodecallback):
        """
        Loops over the grid of nodes. Two callback functions are required:

        :param edgecallback: Called for every edge.
        :param nodecallback: Called for every node.
        """
        logging.info("Iterate through graph.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.getNode(y, x)

                # Node
                nodecallback(node_i, self.g)

                # Right edge
                node_j = self.getNode(y, x + 1)
                edgecallback(node_i, node_j, self.g)

                # Down edge
                node_j = self.getNode(y + 1, x)
                edgecallback(node_i, node_j, self.g)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.getNode(y, self.xsize - 1)

            # Node
            nodecallback(node_i, self.g)

            # Down edge
            node_j = self.getNode(y + 1, self.xsize - 1)
            edgecallback(node_i, node_j, self.g)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.getNode(self.ysize - 1, x)

            # Node
            nodecallback(node_i, self.g)

            # Right edge
            node_j = self.getNode(self.ysize - 1, x + 1)
            edgecallback(node_i, node_j, self.g)

        # Last node
        nodecallback(self.getNode(self.ysize - 1, self.xsize - 1), self.g)

    def loopnodes(self, callback):
        for y in range(self.ysize):
            for x in range(self.xsize):
                callback(self.getNode(y, x), self.g)

    def maxflow(self):
        logging.info("Calculate max flow.")
        self.g.maxflow()

    def getNode(self, y, x):
        return Node(self.nodeids[y, x], y, x)


class Binseg:
    def __init__(self, img, unaries):
        self.img = img
        self.unaries = unaries

        self.nodegrid = Nodegrid(img.shape[0], img.shape[1])

        self.l = 0.0005
        self.w = 10

    def edge(self, node_i, node_j, graph):
        """
        Callback for pairwise energy.
        """
        i = [node_i.y, node_i.x]
        j = [node_j.y, node_j.x]

        # Pixel values
        xi = self.img[i[0], i[1]]
        xj = self.img[j[0], j[1]]

        A = self.pairwiseenergy(0, 0, xi, xj)
        B = self.pairwiseenergy(0, 1, xi, xj)
        C = self.pairwiseenergy(1, 0, xi, xj)
        D = self.pairwiseenergy(1, 1, xi, xj)

        #energy = self.pairwiseenergy(self.unaries[i[0], i[1], 2],
        #                             self.unaries[j[0], j[1], 2],
        #                             xi, xj)

        # print(A, B, C, D)

        graph.add_edge(node_i.nodeid, node_j.nodeid, B + C - A - D, 0.0)

        graph.add_tedge(node_i.nodeid, C, A)
        graph.add_tedge(node_j.nodeid, D, C)

    def node_assign(self, node_i, graph):
        """
        Callback for assigning unary energy.
        """
        graph.add_tedge(node_i.nodeid,
                        self.unaries[node_i.y, node_i.x, 1],
                        self.unaries[node_i.y, node_i.x, 0])

    def node_segment(self, node_i, graph):
        """
        Callback for segmentation.
        """
        if graph.get_segment(node_i.nodeid) == 0:
            self.img[node_i.y, node_i.x] = np.array([0, 0, 0])
        else:
            self.img[node_i.y, node_i.x] = np.array([255, 255, 0])

    def pairwiseenergy(self, y1, y2, x1, x2):
        """
        Returns pairwise energy between node i and node j using the Potts model.

        :param y1: Label of i node.
        :param y2: Label of j node.
        :param x1: Pixel value at node i.
        :param x2: Pixel value at node j.
        :return: Pairwise energy.
        """
        if y1 == y2:
            return 0.0

        # Not same label
        # np.sum(np.power(x1 - x2, 2), 0)
        energy = self.w * np.exp(-self.l * np.power(np.linalg.norm(x1 - x2, 2), 2))
        return energy

    def segment(self):
        self.nodegrid.loop(self.edge, self.node_assign)

        self.nodegrid.maxflow()

        self.nodegrid.loopnodes(self.node_segment)

    def getimg(self):
        return self.img


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("Read image.")
    img = misc.imread("banana3.png")
    img = np.array(img, dtype=np.float64)

    if not os.path.exists("unary.npy"):
        generateunaries(img)

    logging.info("Load unaries.")
    unaries = np.load("unary.npy")

    binseg = Binseg(img, unaries)
    binseg.segment()

    logging.info("Save image.")
    img = binseg.getimg().astype(np.uint8)

    plt.imshow(img)
    plt.show()
    plt.imsave("banana_out", img)


if __name__ == '__main__':
    main()
