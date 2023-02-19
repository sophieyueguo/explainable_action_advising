

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
        y = None

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

    def add_explanation(self, path):
        node_dict = self.collect_clf_paths(path)

        for node_id in node_dict:
            n = node_dict[node_id]
            if not n['is_leaf']:
                self.feature[str(node_id)] = n['feature']
                self.threshold[str(node_id)] = n['threshold']
            self.value[str(node_id)] = n['value']
            self.is_leaf[str(node_id)] = n['is_leaf']

        for i in range(len(path)):
            self.node_depth[str(path[i]['node'])] = i
            if not self.is_leaf[str(path[i]['node'])]:
                if path[i]['inequality'] == '<=': # next node in the path is left child
                    self.children_left[str(path[i]['node'])] = path[i+1]['node']
                elif path[i]['inequality'] == '>': # next node in the path is right child
                    self.children_right[str(path[i]['node'])] = path[i+1]['node']

        self.n_nodes = len(self.value)


    def collect_clf_paths(self, path, printline=False):
        node_dict = {}

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

        return node_dict
