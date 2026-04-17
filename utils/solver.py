import numpy as np
from scipy.optimize import minimize

class Solver():
    def __init__(self, d):
        self.d = d

        self.n = 0
        self.neighbors = np.array([])
        self.reversed_neighbor_idxs = np.array([])
        self.states = []
        self.operators = []
        self.bcs = []
        self.alpha = 0

        self.reset_dm()

    def reset_dm(self):
        raise NotImplementedError
        
    def update_dm(self, element):
        raise NotImplementedError

    def add_system(self, neighbors, states, operators, bcs, alpha, random_state = False, seed = None):
        n, d = neighbors.shape
        d -= 1
        assert self.d == d

        reversed_neighbor_idxs = self.check_neighbors(neighbors)
        self.check_states_operators_bcs(states, operators, bcs, neighbors, reversed_neighbor_idxs)

        self.n = n
        self.neighbors = neighbors
        self.reversed_neighbor_idxs = reversed_neighbor_idxs

        if random_state:
            np.random.seed(seed = seed)
            self.states = [np.random.uniform(low = -1, high = 1, size = state.shape) for state in states]
        else:
            self.states = states
        self.operators = operators
        self.bcs = bcs
        self.alpha = alpha

        self.reset_dm()

    def check_neighbors(self, neighbors):
        reversed_neighbor_idxs = - np.ones(neighbors.shape)
        for element, element_neighbors in enumerate(neighbors):
            for i, element_neighbor in enumerate(element_neighbors):
                if element_neighbor != -1:
                    found = False
                    for reversed_neighbor_idx, reversed_neighbor in enumerate(neighbors[element_neighbor]):
                        if element == reversed_neighbor:
                            found = True
                            reversed_neighbor_idxs[element, i] = reversed_neighbor_idx
                            break
                    if not found:
                        raise ValueError(f'not matching neighbor for element {element}')

        return reversed_neighbor_idxs

    def check_states_operators_bcs(self, states, operators, bcs, neighbors, reversed_neighbor_idxs):
        for state, operator, state_bcs, element_neighbors, element_reversed_neighbor_idxs in zip(states, operators, bcs, neighbors, reversed_neighbor_idxs):
            assert list(state.shape[1:]) == [states[element_neighbor].shape[1:][reversed_neighbor_idx] for element_neighbor, reversed_neighbor_idx in zip(element_neighbors, element_reversed_neighbor_idxs)]
            assert list(operator.shape[1:-1]) == [operators[element_neighbor].shape[1:-1][reversed_neighbor_idx] for element_neighbor, reversed_neighbor_idx in zip(element_neighbors, element_reversed_neighbor_idxs)]
            assert state.shape[0] == operator.shape[0]
            assert state.shape[0] == operator.shape[-1]
            for state_bc, element_neighbor in zip(state_bcs, element_neighbors):              
                if element_neighbor == -1:
                    assert list(state_bc.shape) == 2 * [state.shape[0]]
                else:
                    assert list(state_bc.shape) == 2 * [state.shape[0], states[element_neighbor].shape[0]]

    def get_ansatz(self, element):
        raise NotImplementedError
    
    def get_bcs_regularizer(self, element):
        raise NotImplementedError

    def get_optimization_function(self, element):
        shape = self.states[element].shape
        return lambda state: self.get_ansatz(element)(state.reshape(shape)) + self.alpha * self.get_bcs_regularizer(element)(state.reshape(shape))
    
    def step(self, element):
        shape = self.states[element].shape
        fun = self.get_optimization_function(element)
        result = minimize(fun, self.states[element].reshape(-1))
        if not result.success:
            print(result.message)
            raise RuntimeError()
        self.states[element] = result.x.reshape(shape)
        self.update_dm(element)
    
    def solve(self, rounds, starting_element = 0, starting_direction = 0):
        step = 0
        element = starting_element
        starting_direction = starting_direction
        while step < 2 * rounds * self.n:
            self.step(element)

            step += 1
            next_element = self.neighbors[element][starting_direction]
            if next_element == -1:
                starting_direction = 1 - starting_direction
                next_element = element
            element = next_element

class DMRG(Solver):
    def __init__(self, *_):
        super().__init__(1)

    def reset_dm(self):
        self.dm = {True: dict(), 
                   False: dict()}
        
    def update_dm(self, element):
        for norm in [True, False]:
            for neighbor_idx in range(self.d + 1):
                if neighbor_idx in self.dm[norm]:
                    if element in self.dm[norm][neighbor_idx]:
                        neighbor = self.neighbors[element][neighbor_idx]
                        neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)
                        self.dm[norm][neighbor_idx][element] = self.contract2neighbor_contraction(self.states[element], self.operators[element], neighbor_idx, neighbor_contraction, norm = norm)

    @staticmethod
    def contract2neighbor_contraction(state, operator, neighbor_idx, neighbor_contraction, norm = False):
        if not norm:
            state__neighbor_contraction = np.einsum('qnk,ijk->ijqn', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contraction)
            operator__state__neighbor_contraction = np.einsum('pmjq,ijqn->ipmn', np.moveaxis(operator, neighbor_idx + 1, -2), state__neighbor_contraction)
            return np.einsum('pli,ipmn>lmn', np.moveaxis(state, neighbor_idx + 1, -1), operator__state__neighbor_contraction)
        else:
            state__neighbor_contraction = np.einsum('pnk,ik->ipn', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contraction)
            return np.einsum('pli,ipn>ln', np.moveaxis(state, neighbor_idx + 1, -1), state__neighbor_contraction)


    def get_contraction(self, element, neighbor_idx, norm = False):
        if neighbor_idx not in self.dm[norm]:
            self.dm[norm][neighbor_idx] = {-1: np.ones((1, 1)) if norm else np.ones((1, 1, 1))}

        contration = self.dm[norm][neighbor_idx].get(element, None)
        if contration is None:
            neighbor = self.neighbors[element][neighbor_idx]
            neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)

            contration = self.contract2neighbor_contraction(self.states[element], self.operators[element], neighbor_idx, neighbor_contraction, norm = norm)
            self.dm[norm][neighbor_idx][element] = contration
        return contration


    def get_ansatz(self, element):
        neighbor_contractions = [self.get_contraction(neighbor, neighbor_idx, norm = False) for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        
        def ansatz(state):
            return np.einsum('ijk,ijk->', 
                             neighbor_contractions[1], 
                             self.contract2neighbor_contraction(state, self.operators[element], 0, neighbor_contractions[0], norm = False))
        
        return ansatz
    
    def get_bcs_regularizer(self, element):
        neighbor_states = [self.states[neighbor] if neighbor != -1 else None for neighbor in self.neighbors[element]]
        neighbor_contractions = [self.get_contraction(neighbor, neighbor_idx, norm = True) for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        next_neighbor_contractions = [self.get_contraction(self.neighbors[neighbor][neighbor_idx], neighbor_idx, norm = True) 
                                      if neighbor != -1 else None
                                      for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        bcs = self.bcs[element]

        def bcs_regularizer(state):
            out = 0
            for neighbor_idx, (neighbor_state, neighbor_contraction, next_neighbor_contraction, bc) in enumerate(zip(neighbor_states, neighbor_contractions, next_neighbor_contractions, bcs)):
                if neighbor_state is None:
                    state__neighbor_contraction = np.einsum('qnk,ik->iqn', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contraction)
                    bc__state__neighbor_contraction = np.einsum('pq,iqn->ipn', bc, state__neighbor_contraction)
                    bc_contraction = np.einsum('pli,ipn->ln', np.moveaxis(state, neighbor_idx + 1, -1), bc__state__neighbor_contraction)
                else:
                    neighbor_state__next_neighbor_contraction = np.einsum('swk,ik->isw', np.moveaxis(neighbor_state, neighbor_idx + 1, -1), next_neighbor_contraction)
                    state__next_neighbor_contraction = np.einsum('qnw,isw,->iqsn', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_state__next_neighbor_contraction)
                    bc__state__next_neighbor_contraction = np.einsum('prqs,iqsn->iprn', bc, state__next_neighbor_contraction)
                    neighbor_state__bc__state__next_neighbor_contraction = np.einsum('rui,iprn->upn', np.moveaxis(neighbor_state, neighbor_idx + 1, -1), bc__state__next_neighbor_contraction)
                    bc_contraction = np.einsum('plu,upn->ln', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_state__bc__state__next_neighbor_contraction)
                out += np.einsum('ln,ln->', neighbor_contractions[1 - neighbor_idx], bc_contraction)
            return out
        return bcs_regularizer
