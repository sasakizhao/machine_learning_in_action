import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    data_mat = [] #特征 两个
    label_mat = [] #标签
    with open('testSet.txt', 'r') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def grad_ascent(data_mat_in, class_label):
    #类型转换
    data_matrix = np.mat(data_mat_in)
    label_vector = np.mat(class_label).transpose() # 行向量转换成列向量
    m, n = data_matrix.shape
    alpha = 0.01 #学习率
    max_cycles = 500
    w = np.ones(shape=(n, 1))
    error = 0.0
    #计算梯度
    for i in range(max_cycles):
        h = sigmoid(data_matrix * w)
        error = label_vector - h
        w += alpha * data_matrix.transpose() * error
    return w

def stoc_grad_ascent(data_mat_in, class_label):
    m, n = data_mat_in.shape
    alpha = 0.01 #学习率
    w = np.ones(shape=n)
    error = 0.0
    #计算梯度
    for i in range(m):
        h = sigmoid(sum(data_mat_in[i] * w))
        error = class_label[i] - h
        w += alpha * data_mat_in[i] * error
    return w

def stoc_grad_ascent_better(data_mat_in, class_label, num_iter=180):
    m, n = data_mat_in.shape
    w = np.ones(shape=n)
    error = 0.0
    for i in range(num_iter):
        data_index = list(range(m))
        #计算梯度
        for j in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01 # 学习率
            #此处用len(data_index)而不用m的原因是不放回取样
            random_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat_in[random_index] * w))
            error = float(class_label[random_index]) - h
            w += alpha * data_mat_in[random_index] * error
            del(data_index[random_index])
    return w


def plot_best_fit(weights):

    data_mat, label_mat = load_dataset()
    data_arr = np.array(data_mat)
    m, n = data_arr.shape
    x_cords1 = []; y_cords1 = [] #x轴和y轴
    x_cords2 = []; y_cords2 = []
    for i in range(m):
        if int(label_mat[i]) == 1:
            x_cords1.append(data_arr[i, 1])
            y_cords1.append(data_arr[i, 2])
        else:
            x_cords2.append(data_arr[i, 1])
            y_cords2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #画散点图
    ax.scatter(x_cords1, y_cords1, s=30, c='red', marker='s')
    ax.scatter(x_cords2, y_cords2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify_vector(int_x, w):
    prob = sigmoid(sum(int_x * w))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    num_feature = 21
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    # 先处理训练集
    train_set = []
    train_labels =[]
    for line in fr_train.readlines():
        feature_line = line.strip().split('\t')
        line_arr = []
        for i in range(num_feature):
            line_arr.append(float(feature_line[i]))
        train_set.append(line_arr)
        train_labels.append(feature_line[-1])
    #训练
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    train_w = stoc_grad_ascent_better(train_set, train_labels, 500)
    #测试
    error_count = 0
    num_test_vector = 0.0
    for line in fr_test.readlines():
        num_test_vector += 1
        feature_line = line.strip().split('\t')
        line_arr = []
        for i in range(num_feature):
            line_arr.append(float(feature_line[i]))
        if int(classify_vector(line_arr, train_w)) != int(feature_line[-1]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vector)
    print("错误率为： %f" % error_rate)
    return error_rate

def muti_test():
    num_tests = 10
    error_sum = 0.0
    for i in range(num_tests):
        error_sum += colic_test()
    print("在 %d 次迭代后，错误率为 %f" % (num_tests, float(error_sum / num_tests)))



if __name__ == '__main__':
    # data_mat, label_mat = load_dataset()
    # w = stoc_grad_ascent_better(np.array(data_mat), label_mat)
    # plot_best_fit(w)
    muti_test()



