import numpy as np


def load_dataset(filename):
    data_arr = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split('\t')
            data_arr.append(line_arr)
    return data_arr


def dist_eclud(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_cent(dataset, k):
    n = dataset.shape[1]
    centroids = np.mat(np.zeros(shape=(k, n)))
    for i in range(n):
        min_i = dataset[:, i].min()
        rang_i = float(dataset[:, i].max() - min_i)
        centroids[:, i] = min_i + rang_i * np.random.rand(k, 1)
    return centroids


def k_means(dataset, k, dist_means=dist_eclud, create_cent=rand_cent):
    m = dataset.shape[0]
    cluster_assment = np.mat(np.zeros(shape=(m, 2)))
    centroids = create_cent(dataset, k)
    cluster_changed = True

    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf; min_index = -1
            for j in range(k):
                curr_dist = dist_means(centroids[j, :], dataset[i, :])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, curr_dist **2
        print(centroids)
        for cent in range(k):
            pts_in_clust = dataset[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)
    return centroids, cluster_assment



def bi_k_mean(dataset, k, dist_meas=dist_eclud):
    m = dataset.shape[0]
    cluster_assment = np.mat(np.zeros(shape=(m, 2)))
    centroid0 = np.mean(dataset, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(np.mat(centroid0), dataset[j, :]) **2
    while len(cent_list) < k:
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            pts_in_current_cluster = dataset[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_cluster_assment = k_means(pts_in_current_cluster, 2, dist_meas)

            #计算sse
            sse_split = np.sum(split_cluster_assment[:, 1])
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print('sse_split: %f和sse_not_split: %f ' % (sse_split, sse_not_split))
            if (sse_split + sse_not_split) < lowest_sse:
                lowest_sse = sse_split + sse_not_split
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_cluster_assment.copy()
        # 增加一个中心点
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split

        print("best_cent_to_split 是：", best_cent_to_split)
        print("len best_clust_ass 是：", len(cent_list))

        cent_list[best_cent_to_split] = best_new_cents[0, :]  # 修改
        cent_list.append(best_new_cents[1, :].A) # 增加
        # 更新成新中心点的距离
        cluster_assment[np.nonzero(cluster_assment[:, 0] == best_cent_to_split)[0], :] = best_clust_ass

    cent_mat = np.mat(np.zeros(shape=(len(cent_list), cent_list[0].shape[1])))
    for i in range(len(cent_list)):
        cent_mat[i, :] = cent_list[i]
    return cent_mat, cluster_assment



if __name__ == '__main__':
    data_arr = load_dataset('testSet2.txt')
    data_mat = np.mat(data_arr, dtype=np.float)
    cent_list, cluster_assment = bi_k_mean(data_mat, 3)