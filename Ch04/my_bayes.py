
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_word2vec(vocab_list, input_set):
    result_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result_vec[vocab_list.index(word)] = 1
        else:
            print("词典里无此单词: %s" % word)
    return result_vec


def bag_word2vec(vocab_list, input_set):
    result_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result_vec[vocab_list.index(word)] += 1
        else:
            print("词典里无此单词: %s" % word)
    return result_vec


def train_nb0(train_matrix, train_category):
    num_of_train_docs = len(train_matrix)
    num_of_words = len(train_matrix[0])
    p_abusive = sum(train_category) / num_of_train_docs
    p0_num = np.ones(shape=num_of_words)
    p1_num = np.ones(shape=num_of_words)
    p0_denom = 2.0; p1_denom = 2.0
    for i in range(num_of_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p0_vect = np.log(p0_num / p0_denom)
    p1_vect = np.log(p1_num / p1_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p0 = np.sum(vec2classify * p0_vec) + np.log(p_class1)
    p1 = np.sum(vec2classify * p1_vec) + np.log(1 - p_class1)
    if p0 > p1:
        return 0
    else:
        return 1


def text_parse(big_string):
    import re
    list_of_tokens = re.split(r'\w*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_text():
    doc_list = []; full_text = []; class_list =[]
    #读取资料
    for i in range(1, 26):
        ham_filename = 'email/ham/%d.txt' % i
        spam_filename = 'email/spam/%d.txt' % i
        with open(ham_filename, 'r', encoding='utf8') as ham_f:
            big_string = ham_f.read()
            list_of_tokens = text_parse(big_string)
            doc_list.append(list_of_tokens)
            full_text.append(list_of_tokens)
            class_list.append(0)
        with open(spam_filename, 'r', encoding='utf8') as spam_f:
            big_string = spam_f.read()
            list_of_tokens = text_parse(big_string)
            doc_list.append(list_of_tokens)
            full_text.append(list_of_tokens)
            class_list.append(1)
    # 制作词字典向量
    vocab_list = create_vocab_list(doc_list)
    #拆分训练集和测试集
    training_set = list(range(50)); test_set = []
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    # 训练
    train_mat = []; train_class = []
    for doc_index in training_set:
        train_mat.append(set_word2vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0v, p1v, p_spam = train_nb0(train_mat, train_class)
    # 测试
    error_count = 0
    for index in test_set:
        word_vector = set_word2vec(vocab_list, doc_list[index])
        if classify_nb(word_vector, p0v, p1v, p_spam) != class_list[index]:
            error_count += 1
    print('错误率有', float(error_count) / len(test_set))






def te_nb():
    postingLists, classVec = loadDataSet()
    my_vocab_list = create_vocab_list(postingLists)
    my_train_matrix = []
    for posting_doc in postingLists:
        my_train_matrix.append(set_word2vec(my_vocab_list, posting_doc))
    p0_vect, p1_vect, p_abusive = train_nb0(my_train_matrix, classVec)

    test_entry = ['love', 'my', 'dalmation']
    this_doc = set_word2vec(my_vocab_list, test_entry)

    print('第一类是', classify_nb(this_doc, p0_vect, p1_vect, p_abusive))

    test_entry_2 = ['stupid', 'garbage']
    this_doc_2 = set_word2vec(my_vocab_list, test_entry_2)

    print('第二类是', classify_nb(this_doc_2, p0_vect, p1_vect, p_abusive))
    return

if __name__ == '__main__':
    spam_text()