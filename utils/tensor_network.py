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

    @property
    def rank(self):
        return len(self.shape) - 1

    @property
    def shape(self):
        return self.tensor.shape

class TensorComplex():
    def __init__(self, domain_derivatives_list, tp_reduce):
        self.domain_derivatives_list = domain_derivatives_list
        self.tp_reduce = tp_reduce

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
        assert len(tensor_unit_1.lat_ranks) == len(tensor_unit_2.lat_ranks)

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

            shape += [tensor_power_units[0].shape[tensor_power_units[0].rank - tensor_power_units[0].bot_rank + 1:]]
            slices = [s + [slice(None)] * tensor_power_units[0].bot_rank for s in slices]
            
            add_tensor_power = np.zeros(shape)
            for j, tensor_power_unit in enumerate(tensor_power_units):
                add_tensor_power[slices[j]] = tensor_power_unit.tensor
            
            add_tensor_powers[bot_rank] = add_tensor_power

        shape = list(add_tensor_powers[0].shape[:tensor_power_units[0].rank - tensor_power_units[0].bot_rank + 1])
        slices = [[slice(None)] * (tensor_power_units[0].rank - tensor_power_units[0].bot_rank + 1) for _ in add_tensor_powers]
        bot_ranks = sorted(list(add_tensor_powers.keys()))
        s = 0
        bot_size = add_tensor_powers[bot_ranks[-1]].shape[-1]
        for j, bot_rank in enumerate(bot_ranks):
            new_s = s + bot_size ** bot_rank
            slices[j].append(slice(s, new_s))
            s = new_s

        add_tensor = np.zeros(shape + [s])
        for j, bot_rank in enumerate(bot_ranks):
            add_tensor[slices[j]] = add_tensor_powers[bot_rank].reshape(shape + [-1])

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
    def __init__(self, bond_order, top_rank, operators, neighbors):

        self.states = np.empty((len(neighbors), top_rank) + (bond_order,) * len(neighbors[0]))
        self.operators = operators
        self.neighbors = neighbors

    @classmethod
    def get_operators_from_pde(cls, pde, n, tensor_complex, hs):
        if type(pde) == float:
            return pde
        elif type(pde) == list:
            pass
        else:
            raise TypeError()
        
        

