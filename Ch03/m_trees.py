
import numpy as np
from math import log
import operator


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0] #只剩一类了
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_feature_to_split(dataset)
    best_label = labels[best_feature]

    del(labels[best_feature])
    my_tree = {best_label: {}}
    feat_value = [example[best_feature] for example in dataset]
    uni_value = set(feat_value)
    for value in uni_value:
        sub_dataset = spilt_dataset(dataset, best_feature, value)
        sub_labels = labels[:]
        my_tree[best_label][value] = create_tree(sub_dataset, sub_labels)
    return my_tree


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calc_shannon_ent(dataset):
    '''
    计算香农熵
    :param dataset:
    :return:
    '''

    total_num = len(dataset)
    label_dict = {}
    for fect_vec in dataset:
        current_label = fect_vec[-1]
        if current_label not in label_dict.keys():
            label_dict[current_label] = 0
        label_dict[current_label] += 1
    shannon_ent = 0.0
    for key in label_dict.keys():
        prob = float(label_dict[key] / total_num)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def spilt_dataset(dataset, axis, value):
    current_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            current_dataset.append(reduced_feat_vec)
    return current_dataset


def choose_feature_to_split(dataset):
    feature_num = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(feature_num):
        feature_list = [example[i] for example in dataset]
        uniqueValue = set(feature_list)  #剔除重复的
        new_entropy = 0.0
        for value in uniqueValue:
            sub_dataset = spilt_dataset(dataset, i, value)
            prob = (len(sub_dataset) / float(len(dataset)))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    '''
        返回投票最多的类型
    :param class_list:
    :return:
    '''
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), \
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]




if __name__ == '__main__':
    print("init")


