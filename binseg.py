import numpy as np
from scipy import misc
import scipy.stats
import logging
import matplotlib.pyplot as plt


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


def unaryenergy():
    pass


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

    unary = np.empty(img.shape[0:2])

    logging.info("Save image.")
    img = img.astype(np.uint8)

    plt.imshow(img)
    plt.show()
    plt.imsave("banana_out", img)


if __name__ == '__main__':
    main()
