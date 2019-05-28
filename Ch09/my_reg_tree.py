
import numpy as np


class tree_node():


    def __init__(self, feat, val, right, left):
        feature_to_split_on = feat
        value_of_split = val
        right_branch = right
        left_branch = left



def load_dataset(filename):
    data_arr = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split('\t')
            data_arr.append(line_arr)
    return data_arr


def bin_split_dataset(dataset, feature, value):
    mat_0 = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
    mat_1 = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
    return mat_0, mat_1


def reg_leaf(dataset):
    return np.mean(dataset[:, -1])


def reg_err(dataset):
    x = np.var(dataset[:, -1]) * np.shape(dataset)[0]
    return x


def choose_best_split(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    (tols, toln) = ops
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(dataset)
    m, n = dataset.shape
    S = err_type(dataset)
    best_s = np.inf; best_index = 0; best_val = 0
    for feat_index in range(n - 1):
        for feat_value in set(dataset[:, feat_index].T.A.tolist()[0]):
            mat0, mat1 = bin_split_dataset(dataset, feat_index, feat_value)
            # 判断
            if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
                continue
            s_new = err_type(mat0) + err_type(mat1)
            if s_new < best_s:
                best_s = s_new
                best_index = feat_index
                best_val = feat_value
    if (S - best_s) < tols:
        # 误差没变多少， 未通过容许的误差下降值
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_index, best_val)
    # 再判断一次
    if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
        return None, leaf_type(dataset)
    return best_index, best_val


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat == None:
        return val
    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val
    lset, rset = bin_split_dataset(dataset, feat, val)
    ret_tree['left'] = create_tree(lset, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree



#后剪枝

def is_tree(obj):
    return (type(obj).__name__ == 'dict')

def get_mean(tree):
    if is_tree(tree['left']): tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']): tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testdata):
    if testdata.shape[0] == 0:
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        lset, rset = bin_split_dataset(testdata, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], lset)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], rset)

    if not is_tree(tree['left']) and not is_tree(tree['right']):
        #计算合并前与合并后误差
        lset, rset = bin_split_dataset(testdata, tree['spInd'], tree['spVal'])

        if lset.shape[0] > 0:
            err_left = sum(np.power((lset[:, -1] - tree['left']), 2))
        else:
            err_left = 0
        if rset.shape[0] > 0:
            err_right = sum(np.power((rset[:, -1] - tree['right']), 2))
        else:
            err_right = 0
        err_not_merge = err_left + err_right

        # err_not_merge = sum(np.power((lset[:, -1] - tree['left']), 2)) + sum(np.power((rset[:, -1] - tree['right']), 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        err_merge = sum(np.power(testdata[:, -1] - tree_mean, 2))
        if err_merge < err_not_merge:
            print("合并")
            return tree_mean
        else:
            return tree


def linear_solve(dataset):
    m, n = dataset.shape
    x = np.mat(np.ones(shape=(m,n)))
    y = np.mat(np.ones(shape=(m, 1)))
    x[:, 1:n] = dataset[:, 0:n-1]
    y = dataset[:, -1]

    x_t_x = x.T * x
    if np.linalg.det(x_t_x) == 0.0:
        print("奇异矩阵")
    ws = np.linalg.solve(x_t_x, x.T * y)
    return ws, x, y


def model_leaf(dataset):
    ws, x, y = linear_solve(dataset)
    return ws


def model_err(dataset):
    ws, x, y = linear_solve(dataset)
    y_hat = x * ws
    return np.sum(np.power(y - y_hat, 2))


def reg_tree_eval(model, in_dat):
    return float(model)


def model_tree_eval(model, in_dat):
    n = in_dat.shape[1]
    X = np.mat(np.ones(shape=(1, n+1)))
    X[:, 1:n+1] = in_dat
    return float(X * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_fore_cast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    y_hat = np.mat(np.zeros(shape=(m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, np.mat(test_data[i]), model_eval)
    return y_hat

if __name__ == '__main__':
    # my_data = load_dataset('ex2.txt')
    # my_mat = np.mat(my_data, np.float32)
    # ret_tree = create_tree(my_mat, ops=(0, 1))
    # print('原树', ret_tree)
    #
    # test_data = load_dataset('ex2test.txt')
    # test_mat = np.mat(test_data, np.float32)
    # prune_tree = prune(ret_tree, test_mat)
    # print('剪之后的树', prune_tree)


    # my_dat2 = np.mat(load_dataset('exp2.txt'), dtype=np.float)
    # tree = create_tree(my_dat2, model_leaf, model_err, ops=(1, 10))

    train_mat = np.mat(load_dataset('bikeSpeedVsIq_train.txt'), dtype=np.float)
    test_mat = np.mat(load_dataset('bikeSpeedVsIq_test.txt'), dtype=np.float)
    my_tree = create_tree(train_mat, ops=(1, 20))
    y_hat = create_fore_cast(my_tree, test_mat[:, 0]) #x
    corr = np.corrcoef(y_hat, test_mat[:, 1], rowvar=0)