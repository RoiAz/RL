MAX_SUM = 21
LOSE = 22
DRAW = 23
WIN = 24
NUM_OF_ITER = 20

#
# def hit_prob_21(old_state):
#     if old_state < 10:
#         return 0
#     elif old_state == 11:
#         return 4.0 / 13.0
#     elif 10 < old_state < 21:
#         return 1.0 / 13.0
#     else:
#         raise Exception("Illegal old state: " + str(old_state) + " in hit_prob_21")
#

def hit_prob_smaller_eq_max(new_state, old_state):
    diff = new_state - old_state
    if


def hit_prob_lose(old_state):
    if old_state == 11:
        return 1.0 / 13.0
    elif 11 < old_state < 21:
        return 1 - ((20.0 - old_state) / 13.0)
    elif old_state:
        return 1
    else:
        raise Exception("Illegal old state: " + str(old_state) + " in hit_prob_lose")


def hit_prob(new_state, old_state):
    if new_state  <= old_state +1:
        return 0
    elif new_state == LOSE:
        return hit_prob_lose(old_state)
    elif 2 < new_state <= MAX_SUM:
        return hit_prob_smaller_eq_max(new_state, old_state)
    elif





def value_iteration(dealer_card):
    curr_values






def meny():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    meny()
