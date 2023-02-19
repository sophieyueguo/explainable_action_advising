from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt


'''tutorial https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-download-auto-examples-tree-plot-unveil-tree-structure-py'''

def vis_decision_tree(clf):
    tree.plot_tree(clf)
    plt.show()

def basic_info(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )



def retrieve_path(clf, X_test, sample_id, Y, printline=False):
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    if printline:
        print("Rules used to predict sample {id}:\n".format(id=sample_id))
    path = []
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        if printline:
            print(
                "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                "{inequality} {threshold})".format(
                    node=node_id,
                    sample=sample_id,
                    feature=feature[node_id],
                    value=X_test[sample_id, feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id],
                )
            )
        path.append({'node': node_id,
                     'sample': sample_id,
                     'feature': feature[node_id],
                     'value': X_test[sample_id, feature[node_id]],
                     'inequality': threshold_sign,
                     'threshold': threshold[node_id],
                     'is_leaf': False})

    end = leaf_id[sample_id]
    path.append({'node': end,
                 'sample': sample_id,
                 'value': Y[sample_id],
                 'is_leaf': True})
    return path






def calc_tree_coverage(clf, X_test, sample_ids):
    n_nodes = clf.tree_.node_count
    node_indicator = clf.decision_path(X_test)


    # boolean array indicating the nodes both samples go through
    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    # obtain node ids using position in array
    common_node_id = np.arange(n_nodes)[common_nodes]

    print(
        "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
            samples=sample_ids, nodes=common_node_id
        )
    )
    print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))






###############################################################################

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
        if self.n_nodes == 0:
            return y
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



def collect_clf_paths(clf, build_ind, X, Y, printline=False):
    node_dict = {}
    paths = []
    build_ind = list(set(build_ind))

    for i in build_ind:
        if printline:
            print ('node i', i)
        path = retrieve_path(clf, np.array(X), i, Y)
        paths.append(path)
        for node in path:
            node_dict[node['node']] = node
            if printline:
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

        if printline:
            print ()

    return node_dict, paths


# given the paths, build a small subtree accordingly
def build_sub_tree(node_dict, paths, printline=False):

    new_clf = Subtree()
    new_clf.n_nodes = len(list(node_dict))

    for node_id in node_dict:
        n = node_dict[node_id]
        if not n['is_leaf']:
            new_clf.feature[str(node_id)] = n['feature']
            new_clf.threshold[str(node_id)] = n['threshold']
        new_clf.value[str(node_id)] = n['value']
        new_clf.is_leaf[str(node_id)] = n['is_leaf']

    if printline:
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

    if printline:
        print ('new_clf.node_depth', new_clf.node_depth)
        print ('new_clf.children_left', new_clf.children_left)
        print ('new_clf.children_right', new_clf.children_right)

    if printline:
        for i in range(len(X)):
            print ('node i', i)
            print (new_clf.predict(X[i]))
    return new_clf
