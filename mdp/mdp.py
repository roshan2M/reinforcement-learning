from typing import Callable, Set

class MDP:
    def __init__(self, S: Set[str], A: Set[str], P: Callable[[str, str, str], float], R: Callable[[str, str], float]):
        '''
        A Markov Decision Process (MDP) is a 4-tuple (S, A, P, R):
        - S is a set of all states
        - A is a set of all actions
        - P is a transition function specifying P(s'|s,a)
        - R is a reward function specifying R(s,a)
        '''
        self.S: Set[str] = S
        self.A: Set[str] = A
        self.P: Callable[[str, str, str], float] = P
        self.R: Callable[[str, str], float] = R

    def P(self, s_prime: str, s: str, a: str) -> float:
        '''
        P for an MDP.
        @param s_prime: next state
        @param s: current state
        @param a: action taken in state s
        @return: probability of going to s_prime given s, a
        '''
        if (s_prime not in self.S) or (s not in self.S) or (a not in self.A):
            raise Exception('ERROR. Invalid arguments for transition function.')
        return self.P(s_prime, s, a)

    def R(self, s: str, a: str) -> float:
        '''
        R for an MDP.
        @param s: current state
        @param a: action taken in state s
        @return: reward received when action a is taken in state s
        '''
        if (s not in self.S) or (a not in self.A):
            raise Exception('ERROR. Invalid arguments for reward function.')
        return self.R(s, a)
