from mdp import MDP

from typing import Tuple, Dict, Set

import random


def value_iteration(mdp: MDP, gamma: float, epsilon: float) -> Tuple[Dict, Dict]:
    V = {s: (None, 0.) for s in mdp.S}
    while True:
        delta = 0.
        for s in mdp.S:
            V_old = V[s][1]
            V_new = {a: mdp.R(s, a) + gamma * sum([mdp.P(s_prime, s, a) * V[s_prime][1] for s_prime in mdp.S]) for a in mdp.A}
            V[s] = (max(V_new, key=V_new.get), max(V_new.values()))
            delta = max(delta, abs(V[s][1] - V_old))
        if delta <= epsilon:
            break
    pi = {}
    for s in mdp.S:
        pi[s] = V[s][0]
        V[s] = V[s][1]
    return pi, V


def async_value_iteration(mdp: MDP, gamma: float, num_iterations: int = 1000) -> Tuple[Dict, Dict]:
    Q = {(s, a): 0. for a in mdp.A for s in mdp.S}
    for i in range(num_iterations):
        s = random.sample(mdp.S, 1)[0]
        a = random.sample(mdp.A, 1)[0]
        Q[(s, a)] = mdp.R(s, a) + gamma * sum([mdp.P(s_prime, s, a) * max([Q[(s_prime, a_prime)] for a_prime in mdp.A]) for s_prime in mdp.S])
    pi = {}
    for s in mdp.S:
        values = {a: Q[(s, a)] for a in mdp.A}
        pi[s] = max(values, key=values.get)
    return pi, Q


def modified_policy_iteration(mdp: MDP, gamma: float, epsilon: float, k: int = 5) -> Tuple[Dict, Dict]:
    random_a = random.sample(mdp.A, 1)[0]
    pi = {s: random_a for s in mdp.S}
    V = {s: 0. for s in mdp.S}
    while True:
        for i in range(k):
            for s in mdp.S:
                V[s] = mdp.R(s, pi[s]) + gamma * sum([mdp.P(s_prime, s, pi[s]) * V[s_prime] for s_prime in mdp.S])
        delta = 0.
        for s in mdp.S:
            V_old = V[s]
            V_new = {a: mdp.R(s, a) + gamma * sum([mdp.P(s_prime, s, a) * V[s_prime] for s_prime in mdp.S]) for a in mdp.A}
            pi[s] = max(V_new, key=V_new.get)
            V[s] = max(V_new.values())
            delta = max(delta, abs(V[s] - V_old))
        if delta <= epsilon:
            break
    return pi, V


if __name__ == '__main__':
    '''
    Create Grid MDP as follows (values are rewards):
    x (horizontal), y (vertical)
        0  1  2   3
        -----------
    0 | 0  0  0  +1
    1 | 0  -  0  -1
    2 | 0  0  0   0
    '''

    # NOTE: While (3,0) and (3,1) are states, we remove them from S as they are `terminal states`.
    S = set([f'({x},{y})' for x in range(4) for y in range(3)]) - set(['(1,1)', '(3,0)', '(3,1)'])
    A = set(['u', 'l', 'd', 'r'])
    transitions = {
        ('(0,0)', 'u'): '(0,0)', ('(0,0)', 'l'): '(0,0)', ('(0,0)', 'd'): '(0,1)', ('(0,0)', 'r'): '(1,0)',
        ('(0,1)', 'u'): '(0,0)', ('(0,1)', 'l'): '(0,1)', ('(0,1)', 'd'): '(0,2)', ('(0,1)', 'r'): '(0,1)',
        ('(0,2)', 'u'): '(0,1)', ('(0,2)', 'l'): '(0,2)', ('(0,2)', 'd'): '(0,2)', ('(0,2)', 'r'): '(1,2)',

        ('(1,0)', 'u'): '(1,0)', ('(1,0)', 'l'): '(0,0)', ('(1,0)', 'd'): '(1,0)', ('(1,0)', 'r'): '(2,0)',
        ('(1,2)', 'u'): '(1,2)', ('(1,2)', 'l'): '(0,2)', ('(1,2)', 'd'): '(1,2)', ('(1,2)', 'r'): '(2,2)',

        ('(2,0)', 'u'): '(2,0)', ('(2,0)', 'l'): '(1,0)', ('(2,0)', 'd'): '(2,1)', ('(2,0)', 'r'): '(3,0)',
        ('(2,1)', 'u'): '(2,0)', ('(2,1)', 'l'): '(2,1)', ('(2,1)', 'd'): '(2,2)', ('(2,1)', 'r'): '(3,1)',
        ('(2,2)', 'u'): '(2,1)', ('(2,2)', 'l'): '(1,2)', ('(2,2)', 'd'): '(2,2)', ('(2,2)', 'r'): '(3,2)',

        ('(3,2)', 'u'): '(3,1)', ('(3,2)', 'l'): '(2,2)', ('(3,2)', 'd'): '(3,2)', ('(3,2)', 'r'): '(3,2)',
    }
    rewards = {
        '(0,0)': 0, '(0,1)': 0, '(0,2)': 0, '(1,0)': 0, '(1,2)': 0, '(2,0)': 0, '(2,1)': 0, '(2,2)': 0, '(3,0)': 1, '(3,1)': -1, '(3,2)': 0,
    }
    P = (lambda s_prime, s, a: 1 if transitions[(s, a)] == s_prime else 0)
    R = (lambda s, a: rewards[transitions[(s, a)]])
    grid_mdp = MDP(S, A, P, R)

    gamma = 0.9
    epsilon = 0.001

    pi, V = value_iteration(grid_mdp, gamma, epsilon)
    print(f'Policy:\n{pi}\n')
    print(f'V-Funtion:\n{V}\n')

    pi, Q = async_value_iteration(grid_mdp, gamma)
    print(f'Policy:\n{pi}\n')
    print(f'Q-Function:\n{Q}\n')

    pi, V = modified_policy_iteration(grid_mdp, gamma, epsilon)
    print(f'Policy:\n{pi}\n')
    print(f'V-Funtion:\n{V}\n')
