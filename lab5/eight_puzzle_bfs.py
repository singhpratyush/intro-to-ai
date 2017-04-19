import random
import numpy

from bfs import BFS


class EightPuzzleBFS(BFS):
    def __init__(self):
        super().__init__()
        initial_state = list(range(9))
        random.shuffle(initial_state)
        print(initial_state)
        self.add_state(tuple(initial_state))

    def is_final(self, state):
        number = 1
        state = numpy.array(state).reshape([3, 3])
        for i in range(3):
            for j in range(3):
                if state[i, j] != number:
                    return False
                number = (number + 1) % 9
        return True

    def generate_states(self, state):
        state = numpy.array(state).reshape([3, 3])
        curr_pos = numpy.where(state == 0)
        curr_pos = curr_pos[0][0], curr_pos[1][0]
        states = []
        # Left
        if curr_pos[0] - 1 >= 0:
            left = curr_pos[0] - 1, curr_pos[1]
            temp = state.copy()
            temp[curr_pos], temp[left] = state[left], state[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Right
        if curr_pos[0] + 1 <= 2:
            right = curr_pos[0] + 1, curr_pos[1]
            temp = state.copy()
            temp[curr_pos], temp[right] = state[right], state[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Top
        if curr_pos[1] - 1 >= 0:
            top = curr_pos[0], curr_pos[1] - 1
            temp = state.copy()
            temp[curr_pos], temp[top] = state[top], state[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        # Bottom
        if curr_pos[1] + 1 <= 2:
            bottom = curr_pos[0], curr_pos[1] + 1
            temp = state.copy()
            temp[curr_pos], temp[bottom] = state[bottom], state[curr_pos]
            states.append(tuple(temp.reshape([9]).tolist()))
        return states

    @staticmethod
    def is_done(state, list_of_objs):
        for s in list_of_objs:
            if numpy.array_equal(s, state):
                return True
        return False


eight_puzz_bfs = EightPuzzleBFS()
ans = eight_puzz_bfs.traverse()
print(ans[0])
print(ans[2])

# [1, 5, 8, 0, 4, 3, 7, 6, 2] - 9722
