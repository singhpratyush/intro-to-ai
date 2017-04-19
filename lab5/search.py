class Search:
    def __init__(self):
        self._visited = set()
        self._queue = None

    def set_queue(self, queue):
        self._queue = queue

    def get_queue(self):
        return self._queue

    def traverse(self):
        touches = 0
        while self.has_nodes():
            curr_node = self.get_next_node()
            self.mark_visited(curr_node)
            touches += 1

            if self.is_final(curr_node):
                return curr_node, self._visited, touches

            next_nodes = self.get_adjacent(curr_node)
            for node in next_nodes:
                if not self.is_visited(node):
                    self.add_node(node)

        return None, self._visited, touches

    def has_nodes(self):
        return len(self._queue) > 0

    def get_next_node(self):
        raise NotImplementedError('Method to get next node not defined')

    def mark_visited(self, node):
        self._visited.add(node)

    def add_node(self, node):
        raise NotImplementedError('Method to add node not defined')

    def is_final(self, node):
        raise NotImplementedError('Method to check final node not defined')

    def get_adjacent(self, node):
        raise NotImplementedError('Method to get adjacent nodes not defined')

    def is_visited(self, node):
        return node in self._visited
