import numpy
import sys

sys.path.append('.')

from Lab5.dfs import DepthFirstSearch


class EightNumberDFS(DepthFirstSearch):

    def is_final(self, node):
        return node == (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def get_adjacent(self, node):
        node = numpy.array(node).reshape([3, 3])
        curr_pos = numpy.where(node == 0)
        curr_pos = curr_pos[0][0], curr_pos[1][0]
        states = []
        # Left
        if curr_pos[0] - 1 >= 0:
            left = curr_pos[0] - 1, curr_pos[1]
            temp = node.copy()
            temp[curr_pos], temp[left] = node[left], node[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Right
        if curr_pos[0] + 1 <= 2:
            right = curr_pos[0] + 1, curr_pos[1]
            temp = node.copy()
            temp[curr_pos], temp[right] = node[right], node[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Top
        if curr_pos[1] - 1 >= 0:
            top = curr_pos[0], curr_pos[1] - 1
            temp = node.copy()
            temp[curr_pos], temp[top] = node[top], node[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Bottom
        if curr_pos[1] + 1 <= 2:
            bottom = curr_pos[0], curr_pos[1] + 1
            temp = node.copy()
            temp[curr_pos], temp[bottom] = node[bottom], node[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        return states


def main():
    enp = EightNumberDFS()
    enp.add_node((1, 2, 3, 4, 0, 6, 8, 7, 5))
    result = enp.traverse()
    print(result[0])
    print(result[2])


if __name__ == '__main__':
    main()