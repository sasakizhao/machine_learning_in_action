
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
                feats.append(line_arr[i])
            data_mat.append(feats)
            labels_mat.append(line_arr[-1])
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


#岭回归
def ridge_regress(x_mat, y_mat, lam=0.2):
    x_t_x = x_mat.T * x_mat
    demon = x_t_x + np.eye(x_mat.shape[1]) * lam
    if np.linalg.det(demon) == 0.0:
        print("x_t_x 是奇异矩阵")
        return
    ws = np.linalg.solve(demon, x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = np.mat(x_arr); y_mat = np.mat(y_arr).T

    #标准化
    x_mean = np.mean(x_mat, axis=0)
    x_var = np.var(x_mat, axis=0)
    x_norm = (x_mat - x_mean) / x_var
    y_mean = np.mean(y_mat, axis=0)
    y_mat = y_mat - y_mean

    num_test_plts = 30
    w_mat = np.zeros(shape=(num_test_plts, x_norm.shape[1]))
    for i in range(num_test_plts):
        ws = ridge_regress(x_norm, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat



#前向逐步回归
def stage_wise(x_arr,y_arr, eps=0.01, num_it=500):
    x_mat = np.mat(x_arr); y_mat = np.mat(y_arr)
    #标准化
    x_norm = (x_mat - np.mean(x_mat, axis=0)) / np.var(x_mat, axis=0)
    y_norm = y_mat - np.mean(y_mat, axis=0)
    m, n = x_norm.shape

    return_mat = np.zeros(shape=(num_it, n))
    ws = np.ones(shape=(n, 1)); ws_test = ws.copy(); ws_max = ws.copy()
    lowerst_error = np.inf
    for i in range(num_it):

        print(ws.T)
        for feature_ind in range(n):
            for sign in [-1.0, 1.0 ]:
                ws_test = ws.copy()
                ws_test[feature_ind] += eps * sign  # 做出一丢丢的变化
                y_test = x_norm * ws_test
                test_error = rss_error(y_mat.A, y_test.A)  # array类型
                if test_error < lowerst_error:
                    print(test_error)
                    lowerst_error = test_error
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T


def rss_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


# 测试 直线拟合
def test_1():
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

if __name__ == '__main__':
    # data_arr, labels_arr = load_dataset('abalone.txt')
    # # w_mat = ridge_test(data_arr, labels_arr)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # ax.plot(w_mat)
    # # fig.show()
    # stage_wise(data_arr, labels_arr)
    data_mat, labels_mat = load_dataset('money.txt')
    data_mat = np.mat(data_mat);
    labels_mat = np.mat(labels_mat)
    data = np.hstack((data_mat, labels_mat.T))
    money_count = np.mat(np.zeros(shape=(data.shape[0], 1)))
    for i in range(0, data.shape[0]):
        money_count[i, :] = data[0:i+1, -1].astype(np.float64).sum()
    data = np.hstack((data_mat, money_count))


