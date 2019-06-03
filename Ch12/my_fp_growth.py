


class tree_node:

    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}


    def inc(self, num_occur):
        self.count += num_occur


    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(int(ind + 1))


def create_tree(dataset, nim_sup=1):
    header_table = {}  #头指针表
    for trans in dataset:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + dataset[trans]
    for k in list(header_table.keys()):
        # 频数不够 则删除
        if header_table[k] < nim_sup:
            del(header_table[k])
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]
    ret_tree = tree_node('Null set', 1, None)
    for transet, count in dataset.items():
        local_d = {}
        for item in transet:
            if item in freq_item_set:
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p:p[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = tree_node(items[0], count, in_tree)
        if header_table[items[0]][1] == None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    #寻找最后面的node  链表
    while (node_to_test.node_link != None):
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent != None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    cond_pats = {}
    while tree_node != None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats


def mine_tree(in_tree, header_table, min_sup, prefix, freq_item_list):
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    for base_pat in bigL:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(new_freq_set, header_table[base_pat][1])
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_sup)
        if my_cond_tree != None:
            my_cond_tree.disp()
        if my_head != None:
            mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


if __name__ == '__main__':
    simp_dat = loadSimpDat()
    init_dat = createInitSet(simp_dat)
    tree, header_table = create_tree(init_dat, 3)
    freqItems = []
    mine_tree(tree, header_table, 3, set([]), freqItems)
