import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def fake_map(fun, data):
    '''
        替代py2的 map函数
    :param fun:
    :param data:
    :return:
    '''
    out = []
    for i in range(len(data)):
        out.append(fun(data[i]))
    return out

# def load_dataset(filename, delim='\t'):
#     fr = open(filename, 'r')
#     data_arr = []
#     for line in fr.readlines():
#         line_arr = []
#         for str in line.split(' '):
#             line_arr.append(str)
#         data_arr.append(line_arr)
#     return np.mat(data_arr, dtype=np.float)

def load_dataset(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [fake_map(float,line) for line in stringArr]
    return np.mat(datArr)




def pca(data_mat, top_n_feat=99999999):
    mean_values = np.mean(data_mat, axis=0)
    mean_removed = data_mat - mean_values
    # 协方差
    cov_mat = np.cov(mean_removed, rowvar=0)
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_mat))
    eig_values_ind = np.argsort(eig_values)
    eig_values_ind = eig_values_ind[:-(top_n_feat + 1):-1]
    red_eig_vectors = eig_vectors[:, eig_values_ind]
    #将数据转换到新空间
    low_d_data_mat = mean_removed * red_eig_vectors
    recon_mat = (low_d_data_mat * red_eig_vectors.T) + mean_values
    return low_d_data_mat, recon_mat


def replace_nan_with_mean():
    data_mat = load_dataset('secom.data', ' ')
    num_feat = data_mat.shape[1]
    for i in range(num_feat):
        mean_value = np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:, i].A))[0], i])
        data_mat[np.nonzero(np.isnan(data_mat[:, i].A))[0], i] = mean_value
    return data_mat





if __name__ == '__main__':
    # data_mat = load_dataset('testSet.txt')
    # low_d_data_mat, recon_mat = pca(data_mat, top_n_feat=1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], color='pink')
    #
    # ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], color='blue')
    # plt.show()
    data_mat = replace_nan_with_mean()
    mean_values = np.mean(data_mat, axis=0)
    mean_removed = data_mat - mean_values
    cov_mat = np.cov(mean_removed, rowvar=0)
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_mat))
