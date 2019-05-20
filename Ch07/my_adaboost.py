import numpy as np

def load_simple_data():
    data_mat = np.mat([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_calssify(data_matrix, dimen, thresh_val, thresh_ineq):
    '''
    树墩分类 弱分类器
    :param data_matrix:
    :param dimen: 维度
    :param thresh_val: 阈值
    :param thresh_ineq: 阈值不等式
    :return:
    '''
    ret_array = np.ones(shape=(data_matrix.shape[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    '''
        构建树墩
    :param data_arr:
    :param class_labels:
    :param D:
    :return:
    '''
    data_matrix= np.mat(data_arr); label_matrix = np.mat(class_labels).T
    m, n = data_matrix.shape
    num_steps = 10.0; best_stump = {}; best_class_est = np.mat(np.zeros(shape=(m, 1)))
    min_error = np.inf
    #三个循环
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = float(range_max - range_min) / num_steps
        for j in range(-1, int(num_steps+1)):
            for inequal in ['lt', 'gt']:
                thresh_val = range_min + step_size * float(j)
                predicted_values = stump_calssify(data_matrix, i, thresh_val, inequal)
                error_arr = np.mat(np.ones(shape=(m, 1)))
                error_arr[predicted_values == label_matrix] = 0  # 预测结果和实际标记比较
                weight_error = D.T * error_arr
                print('切片：dim %d, thresh %.2f, thresh inequal %s, errorWeight %.2f'\
                      % (i, thresh_val, inequal, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predicted_values.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it):
    week_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones(shape=(m, 1))/m)
    agg_class_est = np.mat(np.zeros(shape=(m, 1)))  #最终打分累加的
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        print('D : ',D.T)
        alpha = float(0.5 * np.log((1.0 - error) / np.max([error, 1e-16])))
        best_stump['alpha'] = alpha
        # 保留过程记录
        week_class_arr.append(best_stump)
        # print('classEst ', class_est.T)
        # 计算D
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est) # m*1
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        #累计
        agg_class_est += alpha * class_est
        # print('agg_class_est: ', agg_class_est.T)
        agg_error = np.multiply(np.sign(agg_class_est) != np.sign(np.mat(class_labels).T), \
                                np.ones(shape=(m, 1)))
        agg_error_rate = float(agg_error.sum() / m)
        print('总体错误率为： %.2f' % agg_error_rate)
        if agg_error_rate == 0.0:
            break
    return week_class_arr


#ada分类函数
def ada_classify(data2class, classifer_arr):
    data_mat = np.mat(data2class)
    m = data_mat.shape[0]
    agg_classifer = np.mat(np.zeros(shape=(m,1)))
    for i in range(len(classifer_arr)):
        stump = classifer_arr[i]
        class_est = stump_calssify(data_mat, stump['dim'], \
                                   stump['thresh'], stump['ineq'])
        agg_classifer += stump['alpha'] * class_est
        print(agg_classifer)
    return np.sign(agg_classifer)


#预测难分类数据

#读取文件
def load_dataset(filename):
    num_feature = len(open(filename, 'r').readline().split('\t'))
    data_arr = []; labels_arr =[]
    for line in open(filename, 'r').readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feature - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        labels_arr.append(float(cur_line[-1]))
    return data_arr, labels_arr





if __name__ == '__main__':
    # data_mat, class_labels = load_simple_data()
    # class_arr = adaboost_train_ds(data_mat, class_labels, 9)
    # print("结果", ada_classify([[5, 5],[0, 0]], class_arr))
    data_arr, labels_arr = load_dataset('horseColicTraining2.txt')
    classifier_arr = adaboost_train_ds(data_arr, labels_arr, 10)
    m = len(labels_arr)
    err_arr = np.mat(np.ones(shape=(m, 1)))
    prediction = ada_classify(data_arr, classifier_arr)
    error_rate = float(err_arr[prediction != np.mat(labels_arr).T].sum() / m)
    print("训练错误率", error_rate)

    test_arr, test_labels_arr = load_dataset('horseColicTest2.txt')
    prediction = ada_classify(test_arr, classifier_arr)

    m = len(test_labels_arr)
    err_arr = np.mat(np.ones(shape=(m, 1)))
    error_rate = float(err_arr[prediction != np.mat(test_labels_arr).T].sum() / m)
    print("测试错误率", error_rate)


