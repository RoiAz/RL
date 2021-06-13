import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

MAX_BJ_SUM = 21
MAX_DEALER_SUM = 27
HIGHEST_CARD = 14
LOWEST_CARD = 2
LOSE = 22
WIN = 23
DRAW = 24
NUM_OF_ITER = 20
HITS = 0
STICKS = 1


def hit_prob_smaller_eq_max(old_state, new_state):
    diff = new_state - old_state
    if diff == 10:
        return 4.0 / 13.0
    elif diff <= 1 or diff > 11:
        return 0
    else:
        return 1.0 / 13.0


def hit_prob_lose(old_state):
    if old_state == 11:  # only another ace
        return 1.0 / 13.0
    elif 11 < old_state < 21:
        return 1 - ((20.0 - old_state) / 13.0)
    elif old_state < 11:
        return 0
    elif old_state == 21:
        return 1
    return 0


def hit_prob(old_state, new_state):
    if new_state <= old_state + 1 or old_state > MAX_BJ_SUM or old_state < LOWEST_CARD:
        return 0
    elif new_state == LOSE:
        return hit_prob_lose(old_state)
    elif new_state == WIN or new_state == DRAW:
        return 0
    elif LOWEST_CARD < new_state <= MAX_BJ_SUM and new_state > old_state + 1:
        return hit_prob_smaller_eq_max(old_state, new_state)
    return 0


card_to_val = {i: i for i in range(LOWEST_CARD, HIGHEST_CARD)}
card_to_val[11] = 10
card_to_val[12] = 10
card_to_val[13] = 10
card_to_val[14] = 11


def accumulate_prob(curr_dealer_val, k):
    if k == 0:
        return 1
    elif curr_dealer_val >= 17:
        return 0
    elif k <= 1:
        return 0
    elif 1 < k <= MAX_DEALER_SUM:
        val = 0
        for i in range(LOWEST_CARD, HIGHEST_CARD + 1):
            val += (1.0 / 13.0) * accumulate_prob(curr_dealer_val + card_to_val[i], k - card_to_val[i])
        return val
    return 0


def sticks_prob(player_state, dealer_card, new_state):
    if new_state <= MAX_BJ_SUM:
        return 0
    prob_val = 0
    for i in range(LOWEST_CARD, HIGHEST_CARD + 1):  # pull second card
        dealer_new_val = card_to_val[dealer_card] + card_to_val[i]

        if dealer_new_val > MAX_BJ_SUM and new_state == WIN:
            prob_val += 1.0 / 13.0
            continue

        if dealer_new_val >= 17:
            if new_state == WIN and (dealer_new_val < player_state):
                prob_val += 1.0 / 13.0
            elif new_state == DRAW and (dealer_new_val == player_state):
                prob_val += 1.0 / 13.0
            elif new_state == LOSE and (dealer_new_val > player_state):
                prob_val += 1.0 / 13.0
            continue

        for k in range(LOWEST_CARD, MAX_DEALER_SUM + 1 - dealer_new_val):  # max that he can exceeds
            if dealer_new_val + k > MAX_BJ_SUM and new_state == WIN:
                prob_val += (1.0 / 13.0) * accumulate_prob(dealer_new_val, k)
            elif dealer_new_val + k < MAX_BJ_SUM and dealer_new_val + k < player_state and new_state == WIN:
                prob_val += (1.0 / 13.0) * accumulate_prob(dealer_new_val, k)
            elif new_state == LOSE and dealer_new_val + k <= MAX_BJ_SUM and (dealer_new_val + k > player_state):
                prob_val += (1.0 / 13.0) * accumulate_prob(dealer_new_val, k)
            elif new_state == DRAW and dealer_new_val + k <= MAX_BJ_SUM and dealer_new_val + k == player_state:
                prob_val += (1.0 / 13.0) * accumulate_prob(dealer_new_val, k)
    return prob_val


def get_reward(curr_state):
    if curr_state == WIN:
        return 1
    elif curr_state == LOSE:
        return -1
    else:
        return 0


def value_iteration(dealer_card):
    policy = {}
    last_iter = NUM_OF_ITER - 1
    states_value = {i: 0 for i in range(LOWEST_CARD, MAX_BJ_SUM + 4)}
    for i in range(NUM_OF_ITER):
        temp_values = {i: 0 for i in range(LOWEST_CARD, MAX_BJ_SUM + 4)}
        for curr_state in range(LOWEST_CARD, MAX_BJ_SUM + 4):
            hits_val = get_reward(curr_state)
            sticks_val = hits_val
            if curr_state <= MAX_BJ_SUM:
                for new_state in range(LOWEST_CARD, MAX_BJ_SUM + 4):
                    #     print(hit_prob(curr_state, new_state))
                    # print(sticks_prob(curr_state, dealer_card, new_state))
                    hits_val += hit_prob(curr_state, new_state) * states_value[new_state]
                    sticks_val += sticks_prob(curr_state, dealer_card, new_state) * states_value[new_state]
                if i == last_iter:
                    policy[curr_state] = HITS if hits_val > sticks_val else STICKS
            temp_values[curr_state] = max(hits_val, sticks_val)
        # print (states_value)

        states_value = temp_values.copy()
    return states_value, policy


def meny():
    y_x_mat = np.zeros((HIGHEST_CARD - LOWEST_CARD + 1, MAX_BJ_SUM - 1), dtype=float)
    min_sticks_state = np.ones(HIGHEST_CARD - 2 + 1) * np.inf

    for dealer_card in range(LOWEST_CARD, HIGHEST_CARD + 1):  # all possible cards
        states_value, policy = value_iteration(dealer_card)
        for state in range(LOWEST_CARD, MAX_BJ_SUM + 1):
            y_x_mat[dealer_card - LOWEST_CARD][state - LOWEST_CARD] = states_value[state]
        for player_sum, action in policy.items():
            if action == STICKS:
                min_sticks_state[dealer_card-2] = min(player_sum, min_sticks_state[dealer_card-2]) #if we have smaller sum that should  we sticks
                break

    y = np.array(list(range(LOWEST_CARD, HIGHEST_CARD + 1)))
    x = np.array(list(range(LOWEST_CARD, MAX_BJ_SUM + 1)))

    # Plot value function:
    X, Y = np.meshgrid(x, y)
    plt.figure()
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X, Y, y_x_mat, cmap=cm.get_cmap("plasma"), rstride=1)
    plt.colorbar(surf)
    plt.title("Value star")
    plt.xlabel("Player sum")
    plt.ylabel("Dealer showing")
    plt.show()

    plt.figure()
    plt.fill_between(y, min_sticks_state, MAX_BJ_SUM, label="Stick")
    plt.fill_between(y, 0, min_sticks_state, label="Hit")
    plt.title("Min sticks per dealer card")
    plt.xlabel("Dealer Showing")
    plt.ylabel("Player Sum")
    plt.ylim(2, 21)
    plt.text(2, 18, "Sticks", size=25)
    plt.text(10, 5, "Hits", size=25)
    plt.yticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.xticks(np.arange(min(y), max(y) + 1, 1.0))
    plt.grid()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    meny()
