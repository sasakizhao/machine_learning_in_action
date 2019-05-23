
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(filename):
    with open(filename, 'r') as f:
        num_feat = len(f.readline().split('\t')) -1
        data_mat = []
        labels_mat = []
        for line in f.readlines():
            line_arr = line.strip().split('\t')
            feats = []
            for i in range(num_feat):
                feats.append(float(line_arr[i]))
            data_mat.append(feats)
            labels_mat.append(float(line_arr[-1]))
        return data_mat, labels_mat


def stand_regress(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    x_t_x = x_mat.T * x_mat
    if np.linalg.det(x_t_x) == 0.0:
        print("x_t_x 是奇异矩阵")
        return
    ws = np.linalg.solve(x_t_x, x_mat.T * y_mat)
    return ws



if __name__ == '__main__':
    data_mat, labels_mat = load_dataset('ex0.txt')
    ws = stand_regress(data_mat, labels_mat)
    x_mat = np.mat(data_mat); y_mat = np.mat(labels_mat)
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat_copy = x_copy * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0])
    ax.plot(x_copy[:, 1], y_hat_copy)
    fig.show()





