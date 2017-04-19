from Lab5.search import Search


class DepthFirstSearch(Search):

    def __init__(self):
        super().__init__()
        self.set_queue([])

    def add_node(self, node):
        if node not in self.get_queue():
            self.get_queue().append(node)

    def get_next_node(self):
        return self.get_queue().pop()
