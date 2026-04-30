import numpy as np
from scipy.optimize import minimize

from utils.tensor_network import TensorUnit, TensorComplex


class LocalSolver():
    def __init__(self, d):
        self.d = d
        self.n = None
        self.bond_order = None
        self.neighbors = None
        self.states = None
        self.operators = None
        self.pde_power = None
        self.pde_states = None
        self.con_bc_operators = None
        self.env_bc_operators = None
        self.bc_power = None
        self.bc_states = None
        self.alpha = None
        self.reset_dm()


    def reset_dm(self):
        raise NotImplementedError


    def update_dm(self):
        raise NotImplementedError


    def add_system(self, bond_order, neighbors, states, operators, pde_power, con_bc_operators, env_bc_operators, bc_power, alpha):
        n, d = neighbors.shape
        d -= 1
        assert self.d == d

        self.n = n
        self.bond_order = bond_order
        self.neighbors = neighbors
        self.states = states
        self.operators = operators
        self.pde_power = pde_power
        self.con_bc_operators = con_bc_operators
        self.env_bc_operators = env_bc_operators
        self.bc_power = bc_power
        self.alpha = alpha
        
        self.update_states()
        self.reset_dm()


    def update_states(self, element = None):
        s = slice(None) if element is None else slice(element, element + 1)
        power = 1

        shape = list(self.states.shape)
        shape[self.states.top_rank] = 1
        extended_states_tensor = np.concatenate([np.ones(shape), self.states.tensor], axis = self.states.top_rank)

        tmp_states = TensorUnit(extended_states_tensor[s], self.states.top_rank, self.states.lat_ranks, self.states.bot_rank)
        states = tmp_states.copy()
        while power <= self.pde_power or power <= self.bc_power:
            if power == self.pde_power:
                if element is None:
                    self.pde_states = TensorComplex.wrap_lat(states).copy()
                else:
                    self.pde_states.tensor[element] = TensorComplex.wrap_lat(states).tensor[0].copy()
            
            if power == self.bc_power:
                if element is None:
                    self.bc_states = TensorComplex.wrap_lat(states).copy()
                else:
                    self.bc_states.tensor[element] = TensorComplex.wrap_lat(states).tensor[0].copy()

            states = TensorComplex.tpm(states, tmp_states)
            states = TensorUnit(states.tensor.reshape(states.shape[:1] + (-1,) + states.shape[states.top_rank + 1:]), 1, states.lat_ranks, states.bot_rank)
            power += 1


    def get_ansatz(self):
        raise NotImplementedError()


    def get_bcs_regularizer(self):
        raise NotImplementedError()


    def get_optimization_function(self, element, env = True):
        
        def fun(state_tensor):
            shape = [1] + list(self.states.shape[1:])
            shape[self.states.top_rank] = 1
            tmp_state_tensor = np.concatenate([np.ones(shape), state_tensor.reshape((1,) + self.states.shape[1:])], axis = self.states.top_rank)

            power = 1
            tmp_states = TensorUnit(tmp_state_tensor, self.states.top_rank, self.states.lat_ranks, self.states.bot_rank)
            states = tmp_states.copy()
            while power <= self.pde_power or power <= self.bc_power:
                if power == self.pde_power:
                    pde_state_tensor = TensorComplex.wrap_lat(states).tensor[0].copy()
                
                if power == self.bc_power:
                    bc_state_tensor = TensorComplex.wrap_lat(states).tensor[0].copy()

                states = TensorComplex.tpm(states, tmp_states)
                states = TensorUnit(states.tensor.reshape(states.shape[:1] + (-1,) + states.shape[states.top_rank + 1:]), 1, states.lat_ranks, states.bot_rank)
                power += 1

            ansatz_out = self.get_ansatz(element)(pde_state_tensor)
            bc_out = self.get_bcs_regularizer(element, env = env)(bc_state_tensor)
            return ansatz_out + self.alpha * bc_out
        return fun


    def step(self, element, env = True):
        shape = self.states.shape[1:]
        fun = self.get_optimization_function(element, env = env)
        result = minimize(fun, self.states.tensor[element].reshape(-1))#, options = {'maxiter': 1})
        self.states.tensor[element] = result.x.reshape(shape)
        self.update_states(element)
        self.update_dm(element)
        return fun(result.x)


    def next(self):
        raise NotImplementedError


    def solve(self, rounds, starting_element = 0, starting_direction = 0, env = True):
        step = 0
        element = starting_element
        direction = starting_direction
        while step < 2 * rounds * self.n:
            self.step(element, env = env)
            step += 1
            element, direction = self.next(element, direction)
            # break
        return self.states


class DMRG(LocalSolver):
    def __init__(self, *_):
        super().__init__(d = 1)


    def reset_dm(self):
        self.dm = {True: dict(), 
                   False: dict()}


    def update_dm(self, element): # state fix
        for norm in [True, False]:
            for neighbor_idx in range(self.d + 1):
                if neighbor_idx in self.dm[norm] and element in self.dm[norm][neighbor_idx]:
                    neighbor = self.neighbors[element][neighbor_idx]
                    neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)
                    states = self.bc_states if norm else self.pde_states
                    self.dm[norm][neighbor_idx][element] = self.contract2neighbor_contraction(states.tensor[element], element, neighbor_idx, neighbor_contraction, norm = norm)


    def contract2neighbor_contraction(self, state, element, neighbor_idx, neighbor_contraction, norm = False):
        operator = self.operators.tensor[element]
        tmp = np.einsum('qnk,i...k->qi...n', np.moveaxis(state, neighbor_idx + 1, -1), neighbor_contraction)
        if not norm:
            if self.operators.lat_ranks == []:
                tmp = np.einsum('pq,qin->pin', operator, tmp)
            else:
                tmp = np.einsum('pmjq,qijn->pimn', np.moveaxis(operator, neighbor_idx + 1, -2), tmp)
        return np.einsum('pli,pi...n->l...n', np.moveaxis(state, neighbor_idx + 1, -1), tmp)


    def get_contraction(self, element, neighbor_idx, norm = False):
        if neighbor_idx not in self.dm[norm]:
            if norm:
                total_bond_order = self.bond_order ** self.bc_power
                cap = np.ones(total_bond_order ** 2).reshape((total_bond_order, total_bond_order))
            else:
                total_bond_order = self.bond_order ** self.pde_power
                if self.operators.lat_ranks == []:
                    cap = np.ones(total_bond_order ** 2).reshape((total_bond_order, total_bond_order))
                else:
                    cap = np.ones(total_bond_order ** 2 * self.operators.shape[-2]).reshape((total_bond_order, self.operators.shape[-2], total_bond_order))
            self.dm[norm][neighbor_idx] = {-1: cap}

        contration = self.dm[norm][neighbor_idx].get(element, None)
        if contration is None:
            neighbor = self.neighbors[element][neighbor_idx]
            neighbor_contraction = self.get_contraction(neighbor, neighbor_idx, norm = norm)

            states = self.bc_states if norm else self.pde_states
            contration = self.contract2neighbor_contraction(states.tensor[element], element, neighbor_idx, neighbor_contraction, norm = norm)
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
        neighbor_states = [self.bc_states.tensor[neighbor] if neighbor != -1 else None for neighbor in self.neighbors[element]]
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
