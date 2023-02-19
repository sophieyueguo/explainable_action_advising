'''toy example to check with the subtree contrcution'''
from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt

from policy_extraction import tree_helper



class Subtree:
    def __init__(self):
        self.n_nodes = 0
        self.node_depth = {}
        self.children_left = {}
        self.children_right = {}
        self.feature = {}
        self.threshold = {}
        self.value = {}
        self.is_leaf = {}


    def predict(self, x):
        curr = 0
        y = 'undecided'
        while not self.is_leaf[str(curr)]:
            if x[self.feature[str(curr)]] <= self.threshold[str(curr)]:
                if str(curr) in self.children_left:
                    curr =  self.children_left[str(curr)]
                else:
                    return y
            else:
                if str(curr) in self.children_right:
                    curr =  self.children_right[str(curr)]
                else:
                    return y
        assert self.is_leaf[str(curr)]
        return self.value[str(curr)]


if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier()
    X = [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 1]]
    Y = [0, 0, 1, 1, 2, 3, 2, 3]

    clf = clf.fit(X, Y)


    tree_helper.basic_info(clf)

    for i in range(len(X)):
        print ('node i', i)
        path = tree_helper.retrieve_path(clf, np.array(X), i, Y)
        for node in path:
            if not node['is_leaf']:
                print(
                    "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                    "{inequality} {threshold})".format(
                        node=node['node'],
                        sample=node['sample'],
                        feature=node['feature'],
                        value=node['value'],
                        inequality=node['inequality'],
                        threshold=node['threshold'],
                    )
                )
            else:
                print ("leaf node {node} : Y[{sample}] representing value {value}".format(
                            node=node['node'],
                            sample=node['sample'],
                            value=node['value']
                    )
                )

        print ()
    leaf_id = clf.apply(np.array(X))
    print ('leaf_id', leaf_id)



    # build a subtree
    print ('--------------------')
    print ('building a subtree')
    build_ind = [0, 2]

    new_clf = Subtree()
    # node_depth, children_left, feature, threshold, children_right = {}, {}, {}, {}
    node_dict = {}
    paths = []

    X_test = [[0, 0, 0],
             [0, 1, 0]]

    for i in build_ind:
        print ('node i', i)
        path = tree_helper.retrieve_path(clf, np.array(X), i, Y)
        paths.append(path)
        for node in path:
            node_dict[node['node']] = node
            if not node['is_leaf']:
                print(
                    "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                    "{inequality} {threshold})".format(
                        node=node['node'],
                        sample=node['sample'],
                        feature=node['feature'],
                        value=node['value'],
                        inequality=node['inequality'],
                        threshold=node['threshold'],
                    )

                )
            else:
                print ("leaf node {node} : Y[{sample}] representing value {value}".format(
                            node=node['node'],
                            sample=node['sample'],
                            value=node['value']
                    )
                )

        print ()
    #
    # self.n_nodes = 0
    # self.node_depth = {}
    # self.children_left = {}  only keep children when children exist in the path
    # self.children_right = {} same as above
    # self.feature = {}
    # self.threshold = {}
    # self.is_leaf = {} according to the path




    new_clf.n_nodes = len(list(node_dict))

    for node_id in node_dict:
        n = node_dict[node_id]
        if not n['is_leaf']:
            new_clf.feature[str(node_id)] = n['feature']
            new_clf.threshold[str(node_id)] = n['threshold']
        new_clf.value[str(node_id)] = n['value']
        new_clf.is_leaf[str(node_id)] = n['is_leaf']


    print ('new_clf.n_nodes', new_clf.n_nodes)
    print ('new_clf.feature', new_clf.feature)
    print ('new_clf.threshold', new_clf.threshold)
    print ('new_clf.value', new_clf.value)
    print ('new_clf.is_leaf', new_clf.is_leaf)

    for path in paths:
        for i in range(len(path)):
            new_clf.node_depth[str(path[i]['node'])] = i
            if not new_clf.is_leaf[str(path[i]['node'])]:
                if path[i]['inequality'] == '<=': # next node in the path is left child
                    new_clf.children_left[str(path[i]['node'])] = path[i+1]['node']
                elif path[i]['inequality'] == '>': # next node in the path is right child
                    new_clf.children_right[str(path[i]['node'])] = path[i+1]['node']
    print ('new_clf.node_depth', new_clf.node_depth)
    print ('new_clf.children_left', new_clf.children_left)
    print ('new_clf.children_right', new_clf.children_right)


    for i in range(len(X)):
        print ('node i', i)
        print (new_clf.predict(X[i]))

    # examine the prediction on this small subtree:
    # the naive tree can accurately predict what has been seen
