'''

    我自己按照书本的说明自己实现的K-NN算法

'''


import numpy as np
import operator
from os import listdir

def create_data_set():
    '''
    创建数据集
    :return: 元组 （特征，标签）
    '''
    data_set = np.array([[1.0, 1.1], [1.0,1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return data_set, labels

def classify_knn(in_x, data_set, labels, k):
    '''
        KNN算法实现
    :param in_x: 输入待分类的数据
    :param data_set: 训练集
    :param labels: 训练集对应标签
    :param k: 取前k个
    :return: 分类器为输入待分类数据选择的类别
    '''

    data_size = data_set.shape[0] #数据个数
    diff_matrix = np.tile(in_x, (data_size, 1)) - data_set
    diff_matrix_squre = diff_matrix**2
    diff_matrix_squre_sum = diff_matrix_squre.sum(axis=1)
    distances = np.sqrt(diff_matrix_squre_sum)

    sorted_dis_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dis_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def file2matrix(filename):
    with open(filename, 'r') as f:
        array_lines = f.readlines()
        number_of_lines = len(array_lines)
        return_matrix = np.zeros(shape=(number_of_lines, 3))
        class_label_vector = []
        for i, line in enumerate(array_lines):
            line = line.strip()
            list_from_line = line.split('\t')
            return_matrix[i, :] = list_from_line[0:3]
            class_label_vector.append(int(list_from_line[-1]))

    return return_matrix, class_label_vector


def auto_norm(dataset):
    min_value = dataset.min(0)
    print('minvalue is ', min_value)
    max_value = dataset.max(0)
    print('max_value is ', max_value)
    range = max_value - min_value # 标量
    print('range is ', range)
    normed_dataset = np.zeros(shape=np.shape(dataset))
    m = dataset.shape[0]
    normed_dataset = (dataset - np.tile(min_value, (m, 1))) / np.tile(range, (m, 1))
    return normed_dataset, range, min_value


def date_class_test():
    hoRatio = 0.1
    data_fature, data_labels = file2matrix('datingTestSet2.txt')
    normed_dataset, ranges, min_value = auto_norm(data_fature)
    m = normed_dataset.shape[0]
    numTestVecs = int(m*hoRatio)
    print(type(numTestVecs))
    error_count = 0.0
    for i in range(numTestVecs):
        clf_result = classify_knn(normed_dataset[i, :],\
                                  normed_dataset[numTestVecs:m, :], data_labels[numTestVecs:m], k=6)
        print('分类器返回结果: %d 而实际是%d', clf_result, data_labels[i])

        if clf_result != data_labels[i]:
            error_count += 1.0
    print('总错误率是%f' %float(100*(error_count/float(numTestVecs))))


def img2vector(filename):
    return_vector = np.zeros(shape=(1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):
            linestr = f.readline()
            for j in range(32):
                return_vector[0, 32*i +j] = linestr[j]
    return return_vector


def hand_writing_class_test():
    hw_labels = []
    training_file_list = listdir('digits/trainingDigits')
    train_m = len(training_file_list)
    training_matrix = np.zeros(shape=(train_m, 1024))
    #解析训练集
    for i in range(train_m):
        #0_46.txt
        filename = training_file_list[i]
        file_str = filename.split('.')[0]
        label = file_str.split('_')[0]
        hw_labels.append(int(label))
        training_matrix[i, :] = img2vector('digits/trainingDigits/%s' %filename)
    # 解析测试集
    test_file_list = listdir('digits/testDigits')
    test_m = len(test_file_list)
    # test_matrix = np.zeros(shape=(test_m, 1024))
    error_count = 0.0
    for i in range(test_m):
        filename = test_file_list[i]
        file_str = filename.split('.')[0]
        true_label = file_str.split('_')[0]
        test_vector = img2vector('digits/testDigits/%s' %filename)
        clf_result = classify_knn(test_vector, training_matrix, hw_labels, k=3)
        if i%5==0:
            print('分类器返回结果: %d 而实际是%d' % (clf_result, int(true_label)))
        if int(clf_result) != int(true_label):
            error_count += 1
    print('总错误个数有%d' %error_count)
    print('总错误率是%f' % float(100 * (error_count / float(test_m))) + '%')




if __name__ == '__main__':
    hand_writing_class_test()












