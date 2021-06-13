import matplotlib.pyplot as plt
import numpy as np
import random

NUM_OF_JOBS = 5
NUM_OF_STATES = 2 ** NUM_OF_JOBS
GAMMA = 0.999
finish_prob = [0.6, 0.5, 0.3, 0.7, 0.1]
cost = [1, 4, 6, 2, 9]
THRESHOLD = 0.000000000000001


def state_to_jobs(state):
    arr = [int(x) for x in '{:05b}'.format(state)]
    return arr[::-1]


def state_cost():
    c = np.zeros(NUM_OF_STATES)
    for i in range(NUM_OF_STATES):
        bin_state = state_to_jobs(i)
        for j in range(NUM_OF_JOBS):
            c[i] += cost[j] * bin_state[j]
    return c


def policy_to_trans(pol):
    trans = np.zeros((NUM_OF_STATES, NUM_OF_STATES))
    for i in range(1, NUM_OF_STATES):
        if state_to_jobs(i)[int(pol[i] - 1)] == 0:
            print("ERROR")
        else:
            trans[i][i] = 1 - finish_prob[int(pol[i] - 1)]
            trans[i][int(i - 2 ** (pol[i] - 1))] = finish_prob[int(pol[i] - 1)]
    return trans


def value_func_from_policy(pol):
    p = policy_to_trans(pol)
    r = state_cost()
    unit_mat = np.identity(NUM_OF_STATES)
    return np.linalg.inv(unit_mat - GAMMA * p).dot(r)


def policy_iteration(init_pol, iter_num):
    pol = init_pol.copy()
    curr_values = value_func_from_policy(pol)
    temp_values = np.zeros(NUM_OF_JOBS)
    init_state_values = [curr_values[-1]]
    for i in range(iter_num):
        for j in range(1, NUM_OF_STATES):
            for k in range(NUM_OF_JOBS):
                if state_to_jobs(j)[k] == 1:
                    temp_values[k] = np.sum(np.array(cost) * np.array(state_to_jobs(j))) + GAMMA * (
                            finish_prob[k] * curr_values[j - 2 ** k]
                            + (1 - finish_prob[k]) * curr_values[j])
                else:
                    temp_values[k] = np.inf
            pol[j] = np.argmin(temp_values) + 1
        next_values = value_func_from_policy(pol)
        init_state_values.append(next_values[-1])
        if np.sum(np.abs(next_values - curr_values)) < THRESHOLD:
            return pol, init_state_values
        curr_values = next_values.copy()
    return pol, init_state_values


def next_state_func(state, act):
    return state - (2 ** (act - 1))


def simulator(state, act):
    if state_to_jobs(state)[int(act - 1)] == 0:
        print("simulator error - invalid act")
    c = np.sum(np.array(cost) * np.array(state_to_jobs(state)))
    if np.random.rand() > finish_prob[int(act - 1)]:  # np.random.binomial(1, finish_prob[int(act-1)], size=None):
        next_state = state
    else:
        next_state = next_state_func(state, act)
    return c, next_state


################ Q Learning####################


def Qlearn(alpha_func, vstar):
    q_table = np.zeros((NUM_OF_STATES, NUM_OF_JOBS))
    for s in NUM_OF_STATES:  # put inf in illegal action per state
        np.where(not state_to_jobs(s), q_table[s], np.inf)

    epsilon = 0.1
    num_of_episodes = 1000
    num_of_vists = np.zeros(NUM_OF_STATES)
    max_vec_diffs = []
    s0_vec_diffs = []

    for episode in range(num_of_episodes):
        curr_state = 31
        while curr_state:
            if random.uniform(0, 1) < epsilon:
                action = np.random.choice(np.where(state_to_jobs(curr_state))) # explore
            else:
                action = np.argmin(q_table[curr_state])  # exploit
            print(curr_state)
            print(action + 1)
            c, next_state = simulator(curr_state, action + 1)
            old_val = q_table[curr_state, action]
            next_min = np.min(q_table[next_state])
            num_of_vists[curr_state] += 1
            alpha = alpha_func(num_of_vists[curr_state])

            new_val = old_val + alpha * (c + GAMMA * next_min - old_val)
            q_table[curr_state, action] = new_val

            curr_state = next_state

        greedy_policy = np.argmin(q_table, axis=1)
        values_gpol = value_func_from_policy(greedy_policy)
        max_diff = np.linalg.norm(vstar - values_gpol, norm="inf")
        s0_diff = np.abs(vstar - values_gpol)[NUM_OF_STATES - 1]
        max_vec_diffs.append(max_diff)
        s0_vec_diffs.append(s0_diff)
    return max_vec_diffs, s0_vec_diffs


def a1(num_of_visits):
    return 1.0 / float(num_of_visits)


def a2(num_of_visits):
    return 0.01


def a3(num_of_visits):
    return 10.0 / (100.0 + float(num_of_visits))


if __name__ == '__main__':
    pol_ui_ci = np.zeros(NUM_OF_STATES)
    for i in range(1, NUM_OF_STATES):
        pol_ui_ci[i] = np.argmax(np.array(cost) * np.array(state_to_jobs(i)) * np.array(finish_prob)) + 1
    value_func_ui_ci = value_func_from_policy(pol_ui_ci)

    a_func_arr = [a1, a2, a3]
    a_title_arr = ["a1", "a2", "a3"]

    title = "Max Cost Policy"

    for ai, a in enumerate(a_func_arr):
        max_vec_diffs, s0_vec_diffs = Qlearn(a, value_func_ui_ci)
        plt.figure()
        plt.xlabel("State")
        plt.ylabel("Value")
        for vi, v in enumerate(max_vec_diffs):
            if vi % 100 == 0:
                plt.plot(v, "r-")

        plt.title(a_title_arr[ai])
        plt.grid()
        plt.show()

    # # deter_pol = [0, 1, 2, 1, 3, 1, 3, 1, 4, 1, 2, 1, 3, 1, 3, 1, 5, 1, 2, 1, 3, 1, 2, 1, 5, 1, 5, 1, 5, 1, 5, 4]
    # pol_cost_max = np.zeros(NUM_OF_STATES)
    # for i in range(1, NUM_OF_STATES):
    #     pol_cost_max[i] = np.argmax(np.array(cost) * np.array(state_to_jobs(i))) + 1
    # value_func_max = value_func_from_policy(pol_cost_max)
    # Q = "2.e"
    # if Q == "2.c":
    #     plt.figure()
    #     title = "Max Cost Policy"
    #     plt.xlabel("State")
    #     plt.ylabel("Value")
    #     for v in [value_func_max]:
    #         plt.plot(v, "r-")
    #     plt.title(title)
    #     plt.grid()
    #     plt.show()
    #
    # pol, init_state_values = policy_iteration(pol_cost_max, 10000)
    # if Q == "2.d":
    #     plt.figure()
    #     title = "Policy Iteration"
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Value")
    #     for v in [init_state_values]:
    #         plt.plot(v, "r-")
    #     plt.title(title)
    #     plt.grid()
    #     plt.show()
    #

    # if Q == "2.e":
    #     plt.figure()
    #     title = "Policy compare"
    #     plt.xlabel("State")
    #     plt.ylabel("Action")
    #     for v in [pol]:
    #         plt.plot(v, "r-", label="Optimal policy")
    #         plt.legend()
    #     for v in [pol_ui_ci]:
    #         plt.plot(v, "b-", label="ci_ui policy")
    #         plt.legend()
    #     plt.title(title)
    #     plt.grid()
    #     plt.show()
    #
    #     plt.figure()
    #     title = "Policy compare - value"
    #     plt.xlabel("State")
    #     plt.ylabel("Value")
    #     for v in [value_func_max]:
    #         plt.plot(v, "r-", label="Max policy value")
    #         plt.legend()
    #     for v in [value_func_ui_ci]:
    #         plt.plot(v, "b-", label="ci_ui policy value")
    #         plt.legend()
    #     plt.title(title)
    #     plt.grid()
    #     plt.show()
