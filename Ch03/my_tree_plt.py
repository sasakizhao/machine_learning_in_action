import matplotlib.pyplot as plt


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')




def plot_node(node_txt, center_pt, parent_pt, node_type):
    createPlot.ax1.annotate(node_txt, xy=parent_pt, \
                            xycoords='axes fraction', xytext=center_pt,\
                            textcoords='axes fraction', va='center',\
                            ha='center', bbox=node_type, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plot_node('decision_node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('leaf_node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0] #root
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ =='dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_num_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0] #root
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_num_depth(second_dict[key])
        else:
            this_depth = 1
        if max_depth < this_depth:
            max_depth = this_depth
    return max_depth


def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


if __name__ == '__main__':
    createPlot()
    print(createPlot.ax1)