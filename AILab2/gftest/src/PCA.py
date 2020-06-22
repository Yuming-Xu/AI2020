import numpy as np
import matplotlib.pyplot as plt


def plot_pca_stats(e_val):
    plt.plot(np.array(e_val) / np.sum(np.array(e_val)))
    plt.show()


class PCA(object):
    def __init__(self, threshold=0.9991, first_k=3, use_threshold=False):
        self.train_x = None
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.first_k = first_k
        self.mean_vec = None
        self.W = None
        self.mean_vec = None
        pass

    def fit(self, train_x):
        print(train_x[0])
        self.train_x = np.mat(train_x.copy()).T  # n * m m个样本，列向量
        """
        mean_vec = np.mean(self.train_x, axis=1)
        var_mat = self.train_x - np.tile(mean_vec, self.train_x.shape[1])
        cov_mat = var_mat * var_mat.T
        """
        #print(cov_mat)
        cov_mat = np.cov(self.train_x)
        #print(cov_mat)
        e_val, e_vec = np.linalg.eig(cov_mat)
        tmp = sorted(zip(e_val, e_vec.T), reverse=True)
        e_val, e_vec = zip(*tmp)
        #print(e_val)
        #print(e_vec)
        # plot_pca_stats(e_val)
        if self.use_threshold:
            total = sum(e_val)
            tmp = 0
            W = []
            for i in range(len(e_val)):
                tmp += e_val[i]
                W.append(list(np.squeeze(np.array(e_vec[i]))))
                if tmp / total >= self.threshold:
                    break
        else:
            W = [list(np.squeeze(np.array(e_vec[i]))) for i in range(self.first_k)]

        self.W = np.mat(np.array(W)).T
        #self.mean_vec = mean_vec
        proj_train = self.train_x.T * self.W
        print(proj_train[0])
        return proj_train