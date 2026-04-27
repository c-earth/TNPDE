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
    

    def __mul__(self, other):
        assert type(other) == float
        return self.__class__(self.tensor * other, self.top_rank, self.lat_ranks, self.bot_rank)


    def __rmul__(self, other):
        return self.__mul__(other)


class TensorComplex():
    def __init__(self, domain_derivatives_list, tp_reduce):
        self.domain_derivatives_list = []
        for domain_derivatives in domain_derivatives_list:
            shape = list(domain_derivatives.shape)
            shape[-1] += 1
            shape[-2] += 1
            tmp = np.zeros(shape)
            tmp[..., :-1, :-1] = domain_derivatives
            tmp[..., -1, -1] = 1
            self.domain_derivatives_list.append(tmp)

        self.tp_reduce = np.zeros([s + 1 for s in tp_reduce.shape])
        self.tp_reduce[:-1, :-1, :-1] = tp_reduce


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


    def tpm(self, tensor_unit_1, tensor_unit_2):
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
        assert len(tensor_unit_1.lat_ranks) == len(tensor_unit_2.lat_ranks)

        tp_unit = self.tpm(tensor_unit_1, tensor_unit_2)
        mul_tensor = np.einsum('ijk,jk...->i...',
                               self.tp_reduce,
                               np.moveaxis(tp_unit.tensor, [tensor_unit_1.top_rank, tp_unit.top_rank], [0, 1]))
        mul_tensor = np.moveaxis(mul_tensor, 0, tp_unit.top_rank - 1)
        return TensorUnit(mul_tensor, tp_unit.top_rank - 1, tp_unit.lat_ranks, tp_unit.bot_rank)
    

    def add(self, tensor_units):
        # assume all bot_rank contract to a power of state self tensor product
        tensor_units_dict = dict()
        for tensor_unit in tensor_units:
            assert tensor_unit.top_rank == tensor_units[0].top_rank
            assert len(tensor_unit.lat_ranks) == len(tensor_units[0].lat_ranks)
            assert sum(tensor_unit.lat_ranks) == sum(tensor_units[0].lat_ranks)
            tensor_units_dict[tensor_unit.bot_rank] = tensor_units_dict.get(tensor_unit.bot_rank, []) + [tensor_unit]

        add_tensor_powers = dict()
        for bot_rank, tensor_power_units in tensor_units_dict.items():
        
            shape = list(tensor_power_units[0].shape[:tensor_power_units[0].top_rank + 1])
            slices = [[slice(None)] * (tensor_power_units[0].top_rank + 1) for _ in tensor_power_units]
            for r in range(tensor_power_units[0].top_rank + 1, tensor_power_units[0].rank - tensor_power_units[0].bot_rank + 1):
                s = 0
                for j, tensor_power_unit in enumerate(tensor_power_units):
                    new_s = s + tensor_power_unit.shape[r]
                    slices[j].append(slice(s, new_s))
                    s = new_s
                shape.append(s)

            shape += list(tensor_power_units[0].shape[tensor_power_units[0].rank - tensor_power_units[0].bot_rank + 1:])
            slices = [s + [slice(None)] * tensor_power_units[0].bot_rank for s in slices]
            add_tensor_power = np.zeros(shape)
            
            for j, tensor_power_unit in enumerate(tensor_power_units):
                add_tensor_power[*slices[j]] = tensor_power_unit.tensor
            
            add_tensor_powers[bot_rank] = add_tensor_power

        bot_ranks = sorted(list(add_tensor_powers.keys()))
        shape = list(add_tensor_powers[bot_ranks[0]].shape[:tensor_power_units[0].top_rank + 1])
        slices = [[slice(None)] * (tensor_power_units[0].top_rank + 1) for _ in add_tensor_powers]
        
        s_bot = 0
        s_lats = [0] * len(tensor_units[0].lat_ranks)

        bot_size = add_tensor_powers[bot_ranks[-1]].shape[-1] # always assume to have at least a `u` in the pde
        for j, bot_rank in enumerate(bot_ranks):
            new_s_bot = s_bot + bot_size ** bot_rank
            new_s_lats = [s_lat + t_lat for s_lat, t_lat in zip(s_lats, add_tensor_powers[bot_rank].shape[tensor_units[0].top_rank + 1:])]
            slices[j] += [slice(s_lat, new_s_lat) for s_lat, new_s_lat in zip(s_lats, new_s_lats)] + [slice(s_bot, new_s_bot)]
            s_bot = new_s_bot
            s_lats = new_s_lats

        add_tensor = np.zeros(shape + s_lats + [s_bot])
        for j, bot_rank in enumerate(bot_ranks):
            add_tensor[*slices[j]] = add_tensor_powers[bot_rank].reshape(add_tensor_powers[bot_rank].shape[:tensor_units[0].rank - tensor_units[0].bot_rank + 1] + (-1,))

        return TensorUnit(add_tensor, tensor_power_units[0].top_rank, tensor_units[0].lat_ranks, 1)


    def wrap_lat(self, tensor_unit):
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
        for h_tensor_unit in h_tensor_units:
            assert type(h_tensor_unit) == TensorUnit
        self.h_tensor_units = h_tensor_units


    def get_operators_from_basis_interpolation(self, basis_overlap, domain_basis_fun_overlap):
        r = domain_basis_fun_overlap.shape[-1]
        subtraction_tensor = np.zeros((r + 1, r + 1, r + 1))
        for i in range(r):
            subtraction_tensor[i, i, r] = 1
            subtraction_tensor[i, r, i] = -1
        subtraction_tensor[r, r, r] = 1

        extend_basis_overlap = np.zeros((r + 1, r + 1))
        extend_basis_overlap[:r, :r] = basis_overlap
        extend_domain_basis_fun_overlap = np.concatenate([domain_basis_fun_overlap, np.ones(domain_basis_fun_overlap.shape[:-1] + (1,))], axis = -1)

        interpolation_tensor = np.einsum('ijk,jl,nk->nil', subtraction_tensor, extend_basis_overlap, extend_domain_basis_fun_overlap)
        interpolation_operators = np.einsum('nil,nim->nlm', interpolation_tensor, interpolation_tensor)
        return TensorUnit(interpolation_operators, 1, [], 1)


    def get_operators_from_pde(self, pde, delta, previous_states):
        self.p = previous_states
        if type(pde) == list and pde[0] == '+':
            completed_pde = pde + [['*', -1/delta, 'u'], ['*', 1/delta, 'p']]
        else:
            completed_pde = ['+', pde, ['*', -1/delta, 'u'], ['*', 1/delta, 'p']]
        pde_tensor = self.get_pde_tensor(completed_pde)
        return self.tensor_complex.wrap_lat(TensorUnit(np.einsum('nijkl,nipqr->nljpkqr', pde_tensor.tensor, pde_tensor.tensor), 1, [2, 2], 1))


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
                    if type(product) == float or type(term) == float:
                        product = product * term
                    else:
                        product = self.tensor_complex.mul(product, term)
                        product = self.tensor_complex.wrap_lat(product)
                return product
            elif pde[0] == '+':
                terms = [self.get_pde_tensor(term) for term in pde[1:]]
                for term in terms:
                    assert type(term) == TensorUnit
                return self.tensor_complex.add(terms)
        elif pde == 'u':
            return self.dummy_u
        elif pde == 'p':
            return self.p
        else:
            raise TypeError()

    def set_operators(self, operators):
        assert type(operators) == TensorUnit
        self.operators = operators

    def set_states(self, states, bond_order):
        self.set_bond_order(bond_order)
        self.states = TensorUnit(states, 1, self.neighbors.shape[-1] * [1], 0)

    def set_solver(self, solver):
        self.solver = solver

    def set_bcs(self, con_bc_operators, env_bc_operators):
        self.con_bc_operators = con_bc_operators
        self.env_bc_operators = env_bc_operators

    def solve(self, rounds, alpha , env, pde = (False, [])):
        self.solver.add_system(self.bond_order, self.neighbors, self.states, self.operators, self.con_bc_operators, self.env_bc_operators, alpha)
        return self.solver.solve(rounds, starting_element = 1, starting_direction = 0, env = env, pde = pde)
    