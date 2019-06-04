import numpy as np


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def eculid_sim(in_a, in_b):
    '''
        欧氏距离
    :param in_a:
    :param in_b:
    :return:
    '''
    return 1.0 / (1.0 + np.linalg.norm(in_a - in_b))


def pears_sim(in_a, in_b):
    '''
    皮尔逊距离
    :param in_a:
    :param in_b:
    :return:
    '''
    if len(in_a) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(in_a, in_b, rowvar=0)[0][1]


def cos_sim(in_a, in_b):
    num = in_a.T * in_b
    denom = np.linalg.norm(in_a) * np.linalg.norm(in_b)
    return 0.5 + 0.5 * (num / denom)



def stand_est(data_mat, user, sim_meas, item):
    n = data_mat.shape[1]
    sim_total = 0.0
    rate_sim_total = 0.0
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0:
            continue
        over_lap = np.nonzero(np.logical_and(data_mat[:, j].A > 0, data_mat[:, item].A > 0))[0]
        if len(over_lap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(data_mat[over_lap, j], data_mat[over_lap, item])
            print('%d和%d的相似度是：%f' %(j, item, similarity))
            sim_total += similarity
            rate_sim_total += user_rating * similarity
    if sim_total == 0:
        return 0
    else:
        return rate_sim_total / sim_total


def sigmaPct(Sigma, perscent):
    '''
        按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
        后续计算SVD时需要将原始矩阵转换到k维空间
    :param Sigma: 奇异值
    :param perscent: 保留总能量
    :return:
    '''
    Sig2 = Sigma **2
    total_Sig2 = np.sum(Sig2)
    for k in range(len(Sigma)):
        current_per = np.sum(Sig2[:k]) / total_Sig2
        print('当前百分比%f, k+1 = %d' % (current_per, k + 1))
        if current_per < perscent:
            continue
        return k + 1


def svd_est(data_mat, user, sim_meas, item, perscent=0.9):
    n = data_mat.shape[1]
    sim_total = 0.0
    rate_sim_total = 0.0
    U, Sigma, VT = np.linalg.svd(data_mat)
    k = sigmaPct(Sigma, perscent)
    Sig4 = np.mat(np.eye(k) * Sigma[:k])
    x_formed_items = data_mat.T * U[:, :k] * Sig4.I
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_meas(x_formed_items[j, :].T, x_formed_items[item, :].T)
        print('%d和%d的相似度是：%f' %(j, item, similarity))
        sim_total += similarity
        rate_sim_total += user_rating * similarity
    if sim_total == 0:
        return 0
    else:
        return rate_sim_total / sim_total


def recommend(data_mat, user, N=3, sim_meas=eculid_sim, est_method=stand_est):
    un_rated_items = np.nonzero(data_mat[user, :].A == 0)[1]
    if len(un_rated_items) == 0:
        return '你已经对所有物品打分了'
    item_scores = []
    for item in un_rated_items:
        estimated_score = est_method(data_mat, user, sim_meas, item)
        item_scores.append((item, estimated_score))
    # 排序，前N个
    item_scores_sorted = sorted(item_scores, key=lambda parm:parm[1], reverse=True)[:N]
    return item_scores_sorted




if __name__ == '__main__':
    data = loadExData2()
    data_mat = np.mat(data)
    # data_mat[0, 1] = data_mat[0, 0] = data_mat[1, 0] = data_mat[2, 0] = 4
    # data_mat[3, 3] = 2

    r = recommend(data_mat, 2, est_method=svd_est)



