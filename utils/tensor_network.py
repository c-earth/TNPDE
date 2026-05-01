import numpy as np
import string


class TensorUnit():
    def __init__(self, tensor, top_rank, lat_ranks, bot_rank):
        super().__init__()
        self.tensor = tensor
        self.top_rank = top_rank
        self.lat_ranks = lat_ranks
        self.bot_rank = bot_rank

        assert self.rank == self.top_rank + sum(self.lat_ranks) + self.bot_rank

    def __repr__(self):
        return f'{self.__class__.__name__}({self.shape}|{self.top_rank} {self.lat_ranks} {self.bot_rank})'

    @property
    def rank(self):
        return len(self.shape) - 1

    @property
    def shape(self):
        return self.tensor.shape
    

    @property
    def top_shape(self):
        return self.shape[1:self.top_rank + 1]
    
    @property
    def bot_shape(self):
        return self.shape[self.rank - self.bot_rank + 1:]
    
    def copy(self):
        return self.__class__(self.tensor.copy(), self.top_rank, self.lat_ranks, self.bot_rank)
    

    def __mul__(self, other):
        assert type(other) == float
        tensor = self.tensor.copy()
        tensor[*([slice(None)] * self.top_rank + [slice(1, None)])] *= other
        return self.__class__(tensor, self.top_rank, self.lat_ranks, self.bot_rank)


    def __rmul__(self, other):
        return self.__mul__(other)


class TensorComplex():
    def __init__(self, domain_derivatives_list, mul_reduce):
        r = mul_reduce.shape[0]

        self.domain_derivatives_list = []
        for domain_derivatives in domain_derivatives_list:
            extended_domain_derivatives = np.zeros(domain_derivatives.shape[:-2] + (r + 1, r + 1))
            extended_domain_derivatives[..., 1:, 1:] = domain_derivatives
            extended_domain_derivatives[..., 0, 0] = 1
            self.domain_derivatives_list.append(extended_domain_derivatives)
        
        self.mul_reduce = np.zeros([r + 1] * 3)
        self.mul_reduce[1:, 1:, 1:] = mul_reduce
        self.mul_reduce[0, 0, 0] = 1

        self.add_reduce = np.zeros([r + 1] * 3)
        for i in range(r):
            self.add_reduce[i + 1, i + 1, 0] = 1
            self.add_reduce[i + 1, 0, i + 1] = 1
        self.add_reduce[0, 0, 0] = 1

    def dif(self, tensor_unit, axes):
        s = [slice(None) if axis is None else axis for axis in axes]
        domain_derivatives = self.domain_derivatives_list[len(axes)][:, *s]

        added_top_rank = len(domain_derivatives.shape) - 3
        f_indices = string.ascii_letters[:len(domain_derivatives.shape)]
        s_indices = f_indices[0] + string.ascii_letters[len(domain_derivatives.shape):][:tensor_unit.top_rank - 1] + f_indices[-1]
        o_indices = f_indices[:-2] + s_indices[1:-1] + f_indices[-2]
        dif_tensor = np.einsum(f'{f_indices},{s_indices}...->{o_indices}...',
                             domain_derivatives,
                             tensor_unit.tensor)
        return TensorUnit(dif_tensor, tensor_unit.top_rank + added_top_rank, tensor_unit.lat_ranks, tensor_unit.bot_rank)


    @staticmethod
    def tpm(tensor_unit_1, tensor_unit_2):
        f_indices = string.ascii_letters[:tensor_unit_1.rank + 1]
        tpm_tensor = np.einsum(f'{f_indices},{f_indices[0]}...->{f_indices}...',
                               tensor_unit_1.tensor,
                               tensor_unit_2.tensor)
        
        axes = list(range(tensor_unit_1.top_rank + 1))
        c_1 = tensor_unit_1.top_rank + 1
        c_2 = tensor_unit_1.rank + 1

        for _ in range(tensor_unit_2.top_rank):
            axes.append(c_2)
            c_2 += 1

        for lat_rank_1, lat_rank_2 in zip(tensor_unit_1.lat_ranks, tensor_unit_2.lat_ranks):
            for _ in range(lat_rank_1):
                axes.append(c_1)
                c_1 += 1

            for _ in range(lat_rank_2):
                axes.append(c_2)
                c_2 += 1

        for _ in range(tensor_unit_1.bot_rank):
            axes.append(c_1)
            c_1 += 1

        for _ in range(tensor_unit_2.bot_rank):
            axes.append(c_2)
            c_2 += 1

        assert len(axes) == len(tpm_tensor.shape)
        assert set(axes) == set(range(len(axes)))
        tpm_tensor = np.permute_dims(tpm_tensor, axes)

        return TensorUnit(tpm_tensor, 
                          tensor_unit_1.top_rank + tensor_unit_2.top_rank, 
                          [lat_rank_1 + lat_rank_2 for lat_rank_1, lat_rank_2 in zip(tensor_unit_1.lat_ranks, tensor_unit_2.lat_ranks)],
                          tensor_unit_1.bot_rank + tensor_unit_2.bot_rank)

    
    def mul(self, tensor_unit_1, tensor_unit_2):
        if type(tensor_unit_1) == float or type(tensor_unit_2) == float:
            return tensor_unit_1 * tensor_unit_2
        else:
            assert len(tensor_unit_1.lat_ranks) == len(tensor_unit_2.lat_ranks)

            tp_unit = self.tpm(tensor_unit_1, tensor_unit_2)
            mul_tensor = np.einsum('ijk,jk...->i...',
                                self.mul_reduce,
                                np.moveaxis(tp_unit.tensor, [tensor_unit_1.top_rank, tp_unit.top_rank], [0, 1]))
            mul_tensor = np.moveaxis(mul_tensor, 0, tp_unit.top_rank - 1)
            return TensorUnit(mul_tensor, tp_unit.top_rank - 1, tp_unit.lat_ranks, tp_unit.bot_rank)

    def add(self, tensor_unit_1, tensor_unit_2):
        assert tensor_unit_1.top_shape == tensor_unit_2.top_shape
        assert len(tensor_unit_1.lat_ranks) == len(tensor_unit_2.lat_ranks)

        tp_unit = self.tpm(tensor_unit_1, tensor_unit_2)
        add_tensor = np.einsum('ijk,jk...->i...',
                               self.add_reduce,
                               np.moveaxis(tp_unit.tensor, [tensor_unit_1.top_rank, tp_unit.top_rank], [0, 1]))
        add_tensor = np.moveaxis(add_tensor, 0, tp_unit.top_rank - 1)
        return TensorUnit(add_tensor, tp_unit.top_rank - 1, tp_unit.lat_ranks, tp_unit.bot_rank)
        

    @staticmethod
    def wrap_lat(tensor_unit):
        lat_shape = []
        s = tensor_unit.top_rank + 1
        for lat_rank in tensor_unit.lat_ranks:
            new_s = s + lat_rank
            lat_shape.append(np.prod(tensor_unit.shape[s:new_s]))
            s = new_s

        shape = list(tensor_unit.shape[:tensor_unit.top_rank + 1]) + lat_shape + list(tensor_unit.shape[tensor_unit.rank - tensor_unit.bot_rank + 1:])
        wrap_lat_tensor = tensor_unit.tensor.reshape(shape)
        return TensorUnit(wrap_lat_tensor, tensor_unit.top_rank, [1] * len(tensor_unit.lat_ranks), tensor_unit.bot_rank)
    
class TensorNetwork():
    def __init__(self, basis_rank, neighbors, domain_derivatives_list, tp_reduce):

        self.states = None
        self._shape = None
        self.operators = None
        self.solver = None
        self.bond_order = None
        self.basis_rank = basis_rank
        self.neighbors = neighbors
        self.tensor_complex = TensorComplex(domain_derivatives_list, tp_reduce)
        self.h_tensor_units = None
        self.u_shape = None
        self.dummy_u = None
        self.p = None
        self.g = None

    def set_bond_order(self, bond_order):
        assert type(bond_order) == int and bond_order > 0
        self.bond_order = bond_order

    def set_u_shape(self, u_shape):
        for s in u_shape:
            assert type(s) == int and s > 0
        self.u_shape = u_shape
        dummy_u_tensor = np.eye(np.prod(self.u_shape + (self.basis_rank + 1,))).reshape(self.u_shape + (self.basis_rank + 1, -1))
        dummy_u_tensor = dummy_u_tensor[..., *([None] * len(self.neighbors[0])), :]

        self.dummy_u = TensorUnit(np.broadcast_to(dummy_u_tensor, (len(self.neighbors),) + dummy_u_tensor.shape), len(self.u_shape) + 1, [1] * len(self.neighbors[0]), 1)

    def set_h_tensor_units(self, h_tensor_units):
        self.h_tensor_units = []
        for h_tensor_unit in h_tensor_units:
            assert type(h_tensor_unit) == TensorUnit
            shape = list(h_tensor_unit.shape)
            shape[h_tensor_unit.top_rank] = 1
            tmp_h_tensor = np.concatenate([np.ones(shape), h_tensor_unit.tensor], axis = h_tensor_unit.top_rank)
            self.h_tensor_units.append(TensorUnit(tmp_h_tensor, h_tensor_unit.top_rank, h_tensor_unit.lat_ranks, h_tensor_unit.bot_rank))


    def get_operators_from_basis_interpolation(self, basis_overlap, domain_basis_fun_overlap):
        r = domain_basis_fun_overlap.shape[-1]
        subtraction_tensor = np.zeros((r, r + 1, r + 1))
        for i in range(r):
            subtraction_tensor[i, i + 1, 0] = 1
            subtraction_tensor[i, 0, i + 1] = -1

        extend_basis_overlap = np.zeros((r + 1, r + 1))
        extend_basis_overlap[1:, 1:] = basis_overlap
        extend_basis_overlap[0, 0] = 1
        extend_domain_basis_fun_overlap = np.concatenate([np.ones(domain_basis_fun_overlap.shape[:-1] + (1,)), domain_basis_fun_overlap], axis = -1)

        interpolation_tensor = np.einsum('ijk,jl,nk->nil', subtraction_tensor, extend_basis_overlap, extend_domain_basis_fun_overlap)
        interpolation_operators = np.einsum('nil,nim->nlm', interpolation_tensor, interpolation_tensor)
        return TensorUnit(interpolation_operators, 1, [], 1)


    def get_operators_from_pde(self, pde, delta, previous_states):
        shape = list(previous_states.shape)
        shape[previous_states.top_rank] = 1
        tmp_p_tensor = np.concatenate([np.ones(shape), previous_states.tensor], axis = previous_states.top_rank)
        self.p = TensorUnit(tmp_p_tensor, previous_states.top_rank, previous_states.lat_ranks, previous_states.bot_rank)

        self.g = 1/delta
        pde_tensor = self.get_pde_tensor(pde)
        # return pde_tensor, pde_tensor.bot_rank
        pde_tensor_flat_bot = pde_tensor.tensor.reshape(pde_tensor.shape[:pde_tensor.rank - pde_tensor.bot_rank + 1] + (-1,))
        return self.tensor_complex.wrap_lat(TensorUnit(np.einsum('nijkl,nipqr->nljpkqr', pde_tensor_flat_bot, pde_tensor_flat_bot), 1, [2, 2], 1)), pde_tensor.bot_rank


    def get_pde_tensor(self, pde):
        if type(pde) == float:
            return pde
        elif type(pde) == list:
            if pde[0] == 'h':
                return self.h_tensor_units[int(pde[1])]
            elif pde[0] == 'D':
                if len(pde) == 2:
                    return self.tensor_complex.dif(self.dummy_u, [None] * int(pde[1]))
                elif len(pde) > 2:
                    assert pde[1] == len(pde[2:])
                    return self.tensor_complex.dif(self.dummy_u, [None if axis == -1 else int(axis) for axis in pde[2:]])
                else:
                    raise ValueError()
            elif pde[0] == '*':
                terms = [self.get_pde_tensor(term) for term in pde[1:]]
                product = terms[0]
                for term in terms[1:]:
                    product = self.tensor_complex.mul(product, term)
                    product = self.tensor_complex.wrap_lat(product)
                return product
            elif pde[0] == '+':
                terms = [self.get_pde_tensor(term) for term in pde[1:]]
                total = None
                for term in terms:
                    assert type(term) == TensorUnit
                    if total is None:
                        total = term
                    else:
                        total = self.tensor_complex.add(total, term)
                        total = self.tensor_complex.wrap_lat(total)
                return total
        elif pde == 'u':
            return self.dummy_u
        elif pde == 'p':
            return self.p
        elif pde == 'g':
            return self.g
        else:
            raise TypeError()

    def set_operators(self, operators, pde_power):
        assert type(operators) == TensorUnit
        self.operators = operators
        self.pde_power = pde_power

    def set_states(self, states, bond_order):
        assert type(states) == TensorUnit
        self.set_bond_order(bond_order)
        self._shape = states.shape
        self.states = TensorUnit(states.tensor.reshape((states.shape[0], -1) + (bond_order,) * len(states.lat_ranks)), 1, states.lat_ranks, 0)

    def set_solver(self, solver):
        self.solver = solver

    def set_bcs(self, con_bc_operators, env_bc_operators, bc_power):
        self.con_bc_operators = con_bc_operators
        self.env_bc_operators = env_bc_operators
        self.bc_power = bc_power

    def solve(self, rounds, alpha , env):
        self.solver.add_system(self.bond_order, self.neighbors, self.states, self.operators, self.pde_power, self.con_bc_operators, self.env_bc_operators, self.bc_power, alpha)
        states_opt = self.solver.solve(rounds, starting_element = 0, starting_direction = 0, env = env)
        return TensorUnit(states_opt.tensor.reshape(self._shape), len(self._shape) - len(states_opt.lat_ranks) - 1, states_opt.lat_ranks, 0)
    