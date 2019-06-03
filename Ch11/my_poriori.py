

def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(dataset):
    c1 = []
    for line in dataset:
        for item in line:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    # c_out = []
    # for i in range(len(c1)):
    #     c_out.append(frozenset(c1[i]))
    # return c_out
    return fake_map(frozenset, c1)

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


def scan_d(D, ck, min_support):
    as_cnt = {}
    for tid in D:
        for can in ck:
            if can.issubset(tid):
                if can in as_cnt:
                    as_cnt[can] += 1
                else:
                    as_cnt[can] = 1
    num_items = float(len(D))
    ret_list = []
    support_data = {}
    for key in as_cnt.keys():
        support = as_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(Lk, k):
    ret_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])
    return ret_list


def apriori(dataset, min_support=0.5):
    C1 = create_c1(dataset)
    D = fake_map(set, dataset)
    L1, support_data = scan_d(D, C1, min_support)
    L = [L1]
    k = 2
    while (len(L[k - 2])) > 0:
        Ck = apriori_gen(L[k - 2], k)
        Lk, sup_k = scan_d(D, Ck, min_support)
        support_data.update(sup_k)
        L.append(Lk)
        k += 1
    return L, support_data


def generate_rules(L, support_data, min_conf=0.7):
    big_rule_list = []
    for i in range(1, len(L)):
        for freq_set in L[i]:
            H1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                pass
            else:
                calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, H, support_data, br1, min_conf=0.7):
    pruned_h = []
    for conseq in H:
        conf = support_data[freq_set] / support_data[freq_set - conseq]  #  s(P|H)/s(P)
        if conf > min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, H, support_data, br1, min_conf=0.7):
    '''
        进一步合并
    :param freq_set:
    :param H:
    :param support_data:
    :param br1:
    :param min_conf:
    :return:
    '''
    m = len(H[0]) # H中 频繁集大小， 判断是否可以移除大小为m的子集
    if (len(freq_set) > m + 1):
        hmp1 = apriori_gen(H, m + 1)  # 尝试进一步合并
        hmp1 = calc_conf(freq_set, hmp1, support_data, br1, min_conf) # 创建(h * m + 1) 条新候选规则
        if (len(hmp1) > 1):
            rules_from_conseq(freq_set, hmp1, support_data, br1, min_conf)








if __name__ == '__main__':
    data_arr = load_dataset()
    L, support_data = apriori(data_arr)
    rules = generate_rules(L, support_data, min_conf=0.5)
