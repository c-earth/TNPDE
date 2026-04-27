import numpy as np
from scipy.optimize import minimize

import string


class Solver():
    def __init__(self, d):
        self.d = d
        self.n = None
        self.bond_order = None
        self.neighbors = None
        self.states = None
        self.operators = None
        self.con_bc_operators = None
        self.env_bc_operators = None
        self.alpha = None
        self.reset_dm()


    def reset_dm(self):
        raise NotImplementedError


    def update_dm(self):
        raise NotImplementedError


    def add_system(self, bond_order, neighbors, states, operators, con_bc_operators, env_bc_operators, alpha):
        n, d = neighbors.shape
        d -= 1
        assert self.d == d

        self.n = n
        self.bond_order = bond_order
        self.neighbors = neighbors
        self.states = states
        self.operators = operators
        self.con_bc_operators = con_bc_operators
        self.env_bc_operators = env_bc_operators
        self.alpha = alpha
        self.reset_dm()


    def get_ansatz(self):
        raise NotImplementedError


    def get_bcs_regularizer(self):
        raise NotImplementedError


    def get_optimization_function(self, element, env = True, pde = (False, [])):
        shape = list(self.states.shape[1:])
        shape[0] -= 1
        def fun(flat_state):
            state = self.states.tensor[element].copy()
            state[...,:-1,:, :] = flat_state.reshape(shape)
            if pde[0]:
                power_states = [np.ones((1,) + state.shape[1:]), state]
                ansatz_power_states = []
                for bot_rank in pde[1]:
                    while bot_rank >= len(power_states):
                        f_script = string.ascii_letters[:len(power_states[-1].shape)]
                        s_script = string.ascii_letters[len(power_states[-1].shape)] + f_script[len(power_states) - 1:]
                        o_script = string.ascii_letters[len(power_states[-1].shape)] + f_script
                        power_states.append(np.einsum(f'{f_script},{s_script}->{o_script}', power_states[-1], state))
                    ansatz_power_states.append(power_states[bot_rank].reshape((-1,) + state.shape[1:]))
                ansatz_state = np.concatenate(ansatz_power_states, axis = 0)
            else:
                ansatz_state = state
            ansatz_out = self.get_ansatz(element)(ansatz_state)
            bc_out = self.get_bcs_regularizer(element, env = env)(state)
            return ansatz_out + self.alpha * bc_out
        return fun


    def step(self, element, env = True, pde = (False, [])):
        shape = list(self.states.shape[1:])
        shape[0] -= 1
        fun = self.get_optimization_function(element, env = env, pde = pde)
        result = minimize(fun, self.states.tensor[element][:-1].reshape(-1), options = {'maxiter': 1})
        self.states.tensor[element][...,:-1,:, :] = result.x.reshape(shape)
        self.update_dm(element)
        return fun(result.x)


    def next(self):
        raise NotImplementedError


    def solve(self, rounds, starting_element = 0, starting_direction = 0, env = True, pde = (False, [])):
        step = 0
        r = 0
        element = starting_element
        direction = starting_direction
        while step < 2 * rounds * self.n:
            self.step(element, env = env, pde = pde)
            step += 1
            if step % (2 * self.n) == 0:
                r += 1
            element, direction = self.next(element, direction)
        return self.states


class DMRG(Solver):
    def __init__(self, *_):
        super().__init__(d = 1)


    def reset_dm(self):
        self.dm = {True: dict(), 
                   False: dict()}


    def update_dm(self, element):
        for norm in [True, False]:
            for neighbor_idx in range(self.d + 1):
                if neighbor_idx in self.dm[norm] and element in self.dm[norm][neighbor_idx]:
                    neighbor = self.neighbors[element][neighbor_idx]
                    neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)
                    self.dm[norm][neighbor_idx][element] = self.contract2neighbor_contraction(self.states.tensor[element], element, neighbor_idx, neighbor_contraction, norm = norm)


    def contract2neighbor_contraction(self, state, element, neighbor_idx, neighbor_contraction, norm = False):
        operator = self.operators.tensor[element]

        print(state.shape, element, neighbor_idx, neighbor_contraction.shape)

        tmp = np.einsum('qnk,i...k->qi...n', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contraction)
        if not norm:
            if self.operators.lat_ranks == []:
                tmp = np.einsum('pq,qin->pin', operator, tmp)
            else:
                tmp = np.einsum('pmjq,qijn->pimn', np.moveaxis(operator, neighbor_idx + 1, -2), tmp)
        return np.einsum('pli,pi...n->l...n', np.moveaxis(state, neighbor_idx + 1, -1), tmp)


    def get_contraction(self, element, neighbor_idx, norm = False):
        if neighbor_idx not in self.dm[norm]:
            cap_no_mid = np.ones(self.bond_order ** 2).reshape((self.bond_order, self.bond_order))
            if self.operators.lat_ranks != []:
                cap = np.ones(self.bond_order ** 2 * self.operators.shape[-2]).reshape((self.bond_order, self.operators.shape[-2], self.bond_order))
            else:
                cap = cap_no_mid
            self.dm[norm][neighbor_idx] = {-1: cap_no_mid if norm else cap}

        contration = self.dm[norm][neighbor_idx].get(element, None)
        if contration is None:
            neighbor = self.neighbors[element][neighbor_idx]
            neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)

            contration = self.contract2neighbor_contraction(self.states.tensor[element], element, neighbor_idx, neighbor_contraction, norm = norm)
            self.dm[norm][neighbor_idx][element] = contration
        return contration


    def get_ansatz(self, element):
        neighbor_contractions = [self.get_contraction(neighbor, neighbor_idx, norm = False) for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        def ansatz(state):
            return np.tensordot(neighbor_contractions[1], 
                                self.contract2neighbor_contraction(state, element, 0, neighbor_contractions[0], norm = False),
                                axes = len(neighbor_contractions[1].shape))
        return ansatz


    def get_bcs_regularizer(self, element, env = True):
        neighbor_states = [self.states.tensor[neighbor] if neighbor != -1 else None for neighbor in self.neighbors[element]]
        neighbor_contractions = [self.get_contraction(neighbor, neighbor_idx, norm = True) for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        next_neighbor_contractions = [self.get_contraction(self.neighbors[neighbor][neighbor_idx], neighbor_idx, norm = True) 
                                      if neighbor != -1 else None
                                      for neighbor_idx, neighbor in enumerate(self.neighbors[element])]
        bcs = []
        for neighbor_idx, neighbor in enumerate(self.neighbors[element]):
            if neighbor != -1:
                bcs.append(self.con_bc_operators[element, neighbor_idx])
            elif not env:
                bcs.append(None)
            else:
                element_bc_operatos = self.env_bc_operators.get(element, dict())
                bcs.append(element_bc_operatos.get(neighbor_idx, None))

        def bcs_regularizer(state):
            out = 0
            for neighbor_idx, (neighbor_state, bc) in enumerate(zip(neighbor_states, bcs)):
                if bc is None:
                    continue
                elif neighbor_state is None:
                    tmp = np.einsum('qnk,ik->qin', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contractions[neighbor_idx])
                    tmp = np.einsum('pq,qin->pin', bc, tmp)
                    tmp = np.einsum('pli,pin->ln', np.moveaxis(state, neighbor_idx + 1, -1), tmp)
                else:
                    tmp = np.einsum('swk,ik->siw', np.moveaxis(neighbor_state, neighbor_idx + 1, -1), next_neighbor_contractions[neighbor_idx])
                    tmp = np.einsum('qnw,siw->qsin', np.moveaxis(state, neighbor_idx + 1, -1), tmp)
                    tmp = np.einsum('prqs,qsin->prin', bc, tmp)
                    tmp = np.einsum('rui,prin->pun', np.moveaxis(neighbor_state, neighbor_idx + 1, -1), tmp)
                    tmp = np.einsum('plu,pun->ln', np.moveaxis(state, neighbor_idx + 1, -1), tmp)
                out += np.einsum('ln,ln->', neighbor_contractions[1 - neighbor_idx], tmp)
            return out
        return bcs_regularizer


    def next(self, element, direction):
        next_element, next_direction = element, direction
        if self.neighbors[element][direction] == -1:
            next_direction = 1 - direction
        else:
            next_element = self.neighbors[element][direction]
        return next_element, next_direction
