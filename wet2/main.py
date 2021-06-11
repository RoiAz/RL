MAX_SUM = 21
LOSE = 22
WIN = 23
NATURAL_STATE = 1
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
    if diff == 10:
        return 4.0 / 13.0
    elif diff <= 1 or diff > 11:
        raise Exception("Illegal diff: " + str(old_state) + " in hit_prob_smaller_eq_max")
    else:
        return 1.0 / 13.0


def hit_prob_lose(old_state):
    if old_state == 11:  # only another ace
        return 1.0 / 13.0
    elif 11 < old_state < 21:
        return 1 - ((20.0 - old_state) / 13.0)
    elif old_state == 21:
        return 1
    else:
        raise Exception("Illegal old state: " + str(old_state) + " in hit_prob_lose")


def hit_prob(old_state, new_state):
    if new_state <= old_state + 1 or old_state > MAX_SUM or old_state < 2:
        return 0
    elif new_state == LOSE:
        return hit_prob_lose(old_state)
    elif 2 < new_state <= MAX_SUM and new_state > old_state + 1:
        return hit_prob_smaller_eq_max(new_state, old_state)
    else:
        raise Exception("Illegal old state: " + str(old_state) + " in hit_prob")


card_to_val = {i: i for i in range(2, 11)}
card_to_val[11] = 10
card_to_val[12] = 10
card_to_val[13] = 10


def accumulate_prob(k):
    if k == 0:
        return 1
    elif k <= 1:
        return 0
    elif 1 < k <= MAX_SUM:
        val = 0
        for i in range(2, 14):
            val += (1.0 / 13.0) * accumulate_prob(k - card_to_val[i])
        return val
    else:
        raise Exception("Illegal k: " + str(k) + " in accumulate_prob")


def sticks_prob(player_state, dealer_card, new_state):
    if 0: #add protections
        return 0


    for i in range(2, 14):
        if card_to_val[dealer_card] + card_to_val[i] >= 17:
            break




def get_reward(curr_state):
    if curr_state == WIN:
        return 1
    elif curr_state == LOSE:
        return -1
    else:
        return 0


def value_iteration(dealer_card):
    policy = {}
    states_value = {i: 0 for i in range(2, MAX_SUM + 3)}
    for i in range(NUM_OF_ITER):
        temp_values = {i: 0 for i in range(2, MAX_SUM + 3)}
        for curr_state in range(2, MAX_SUM + 4):
            hits_val = get_reward(curr_state)
            sticks_val = hits_val
            for new_state in range(2, MAX_SUM + 4):
                hits_val += hit_prob(curr_state, new_state) * states_value[new_state]
                sticks_val += sticks_prob(curr_state, new_state) * states_value[new_state]
            if curr_state <= MAX_SUM:
                policy[curr_state] = "hits" if hits_val > sticks_val else "sticks"
            temp_values[curr_state] = max(hits_val, sticks_val)
        states_value = temp_values.copy()
    return states_value, policy


def meny():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    meny()
