import numpy
import sys
sys.path.append('.')

from Lab5.a_star_search import AStarSearch


class EightNumberAStar(AStarSearch):

    @staticmethod
    def get_score(node):
        if len(node) == 2:
            node = node[1]
        score = 0
        node = node
        for i in range(8):
            if node[i] != i + 1:
                score += 1
        if node[8] != 0:
            score += 1
        return score

    def is_final(self, node):
        return EightNumberAStar.get_score(node) == 0

    def get_adjacent(self, node):
        node = numpy.array(node[1]).reshape([3, 3])
        curr_pos = numpy.where(node == 0)
        curr_pos = curr_pos[0][0], curr_pos[1][0]
        states = []
        # Left
        if curr_pos[0] - 1 >= 0:
            left = curr_pos[0] - 1, curr_pos[1]
            temp = node.copy()
            temp[curr_pos], temp[left] = node[left], node[curr_pos]
            state = tuple(temp.reshape([9]).tolist())
            score = EightNumberAStar.get_score(state)
            states.append((score, state))
        # Right
        if curr_pos[0] + 1 <= 2:
            right = curr_pos[0] + 1, curr_pos[1]
            temp = node.copy()
            temp[curr_pos], temp[right] = node[right], node[curr_pos]
            state = tuple(temp.reshape([9]).tolist())
            score = EightNumberAStar.get_score(state)
            states.append((score, state))
        # Top
        if curr_pos[1] - 1 >= 0:
            top = curr_pos[0], curr_pos[1] - 1
            temp = node.copy()
            temp[curr_pos], temp[top] = node[top], node[curr_pos]
            state = tuple(temp.reshape([9]).tolist())
            score = EightNumberAStar.get_score(state)
            states.append((score, state))
        # Bottom
        if curr_pos[1] + 1 <= 2:
            bottom = curr_pos[0], curr_pos[1] + 1
            temp = node.copy()
            temp[curr_pos], temp[bottom] = node[bottom], node[curr_pos]
            state = tuple(temp.reshape([9]).tolist())
            score = EightNumberAStar.get_score(state)
            states.append((score, state))
        return states


def main():
    enp = EightNumberAStar()
    root = (1, 2, 3, 4, 0, 6, 8, 7, 5)
    score = EightNumberAStar.get_score(root)
    enp.add_node((score, root))
    result = enp.traverse()
    print(result[0])
    print(result[2])


if __name__ == '__main__':
    main()
