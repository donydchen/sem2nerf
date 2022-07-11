import numpy as np


class ToOneHot(object):
    """ Convert the input PIL image to a one-hot torch tensor """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def onehot_initialization(self, a):
        if self.n_classes is None:
            self.n_classes = len(np.unique(a))
        out = np.zeros(a.shape + (self.n_classes, ), dtype=int)
        out[self.__all_idx(a, axis=2)] = 1
        return out

    def __all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def __call__(self, img):
        img = np.array(img)
        one_hot = self.onehot_initialization(img)
        return one_hot
