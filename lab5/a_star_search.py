import sys
sys.path.append('.')

from queue import PriorityQueue
from Lab5.search import Search


class AStarSearch(Search):

    def __init__(self):
        super().__init__()
        self.set_queue(PriorityQueue())

    def add_node(self, node):
        self.get_queue().put(node)

    def get_next_node(self):
        node = self.get_queue().get()
        print(node)
        return node

    def has_nodes(self):
        return not self.get_queue().empty()
