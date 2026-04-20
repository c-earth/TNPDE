import numpy as np
import string


class TensorUnit_0():
    def __init__(self, tensor, top_rank, bot_rank):
        super().__init__()
        self.tensor = tensor
        self.shape = self.tensor.shape

        self.top_rank = top_rank
        self.bot_rank = bot_rank
        
        assert self.rank <= self.top_rank + self.bot_rank


    @property
    def rank(self):
        return len(self.shape)
    
    @property
    def top_shape(self):
        return self.shape[:self.top_rank]

    @property
    def lat_shape(self):
        return self.shape[self.top_rank: self.rank - self.bot_rank]

    @property
    def bot_shape(self):
        return self.shape[self.rank - self.bot_rank:]
    

    @staticmethod
    def gen_mul_einsum_script(rank_1, top_rank_1, bot_rank_1, rank_2, top_rank_2, bot_rank_2, rank_tp_reduce):
        assert rank_1 + rank_2 + rank_tp_reduce - top_rank_1 - top_rank_2 < 52 # limited by string.ascii_letters characters
        idxs_1 = string.ascii_letters[:rank_1]
        idxs_2 = string.ascii_letters[rank_1: rank_1 + rank_2]
        split_1 = [idxs_1[:top_rank_1], idxs_1[top_rank_1: rank_1 - bot_rank_1], idxs_1[rank_1 - bot_rank_1:]]
        split_2 = [idxs_2[:top_rank_2], idxs_2[top_rank_2: rank_2 - bot_rank_2], idxs_2[rank_2 - bot_rank_2:]]
        split_tp_reduce = [string.ascii_letters[rank_1 + rank_2: rank_1 + rank_2 + rank_tp_reduce - top_rank_1 - top_rank_2], split_1[0], split_2[0]]
        idxs_tp_reduce = sum(split_tp_reduce, start = '')
        idxs_out = split_tp_reduce[0] + ''.join(sum(zip(split_1[1], split_2[1]), start = tuple())) + split_1[2] + split_2[2]
        return f'{idxs_tp_reduce},{idxs_1},{idxs_2}->{idxs_out}'


    def mul(self, other, tp_reduce):
        assert self.top_rank + other.top_rank == tp_reduce.bot_rank
        assert self.rank - self.top_rank - self.bot_rank == other.rank - other.top_rank - other.bot_rank

        mul_shape = tp_reduce.top_shape + tuple(map(np.prod, zip(self.lat_shape, other.lat_shape))) + self.bot_shape + other.bot_shape

        mul_tensor = np.einsum(self.gen_mul_einsum_script(self.rank, self.top_rank, self.bot_rank, 
                                                          other.rank, other.top_rank, other.bot_rank, 
                                                          tp_reduce.rank), 
                               tp_reduce.tensor, self.tensor, other.tensor).reshape(mul_shape)
        
        return self.__class__(mul_tensor, tp_reduce.top_rank, self.bot_rank + other.bot_rank)
    
    @staticmethod
    def gen_mul_hang_einsum_script(rank_1, top_rank_1, bot_rank_1, rank_tp_reduce, top_rank_tp_reduce):
        assert rank_1 + rank_tp_reduce - top_rank_1 < 52 # limited by string.ascii_letters characters
        idxs_1 = string.ascii_letters[:rank_1]
        split_1 = [idxs_1[:top_rank_1], idxs_1[top_rank_1: rank_1 - bot_rank_1], idxs_1[rank_1 - bot_rank_1:]]
        split_tp_reduce = [string.ascii_letters[rank_1: rank_1 + top_rank_tp_reduce], 
                           split_1[0], 
                           string.ascii_letters[rank_1 + top_rank_tp_reduce: rank_1 + rank_tp_reduce - top_rank_1]]
        
        idxs_tp_reduce = sum(split_tp_reduce, start = '')

        idxs_out = split_tp_reduce[0] + split_1[1] + split_1[2] + split_tp_reduce[2]

        return f'{idxs_tp_reduce},{idxs_1}->{idxs_out}'

    def mul_hang(self, tp_reduce):
        assert self.top_rank <= tp_reduce.bot_rank
        mul_hang_tensor = np.einsum(self.gen_mul_hang_einsum_script(self.rank, self.top_rank, self.bot_rank, 
                                                                    tp_reduce.rank, tp_reduce.top_rank), 
                                    tp_reduce.tensor, 
                                    self.tensor)

        return self.__class__(mul_hang_tensor, tp_reduce.top_rank, self.bot_rank + tp_reduce.bot_rank - self.top_rank)
    
    def cap(self, other):
        assert self.top_rank >= other.bot_rank
        idxs_cap = string.ascii_letters[:other.rank]
        cap_tensor = np.einsum(f'{idxs_cap},{idxs_cap[other.rank - other.bot_rank:]}...->{idxs_cap[:other.rank - other.bot_rank]}...', 
                               other.tensor, 
                               self.tensor)
        

        return self.__class__(cap_tensor, self.top_rank - other.bot_rank + other.top_rank, self.bot_rank)
    
    def dup(self, n):
        assert self.top_rank == 1

        dup_shape = n * self.shape[:1] + self.shape[1:]
        dup_tensor = np.zeros(dup_shape)
        for i in range(n):
            dup_tensor[[slice(i, i + 1)] * n] = self.tensor[i: i + 1]

        return self.__class__(dup_tensor, n, self.bot_rank)
    
    @classmethod
    def sum(cls, tensor_units):
        # assume all units with bottom rank 1 contract to the same tensor, i.e., state tensor
        bot_size = None
        top_rank = tensor_units[0].top_rank
        for tensor_unit in tensor_units:
            assert tensor_unit.top_shape == tensor_units[0].top_shape
            assert tensor_unit.rank - tensor_unit.top_rank - tensor_unit.bot_rank == tensor_units[0].rank - top_rank - tensor_units[0].bot_rank
            assert tensor_unit.bot_rank in [0, 1]
            if tensor_unit.bot_rank == 1:
                if bot_size is None:
                    bot_size = tensor_unit.shape[-1]
                else:
                    assert tensor_unit.shape[-1] == bot_size
        
        sum_shape = tensor_units[0].top_shape + tuple(map(sum, zip(*[tensor_unit.lat_shape for tensor_unit in tensor_units]))) + (bot_size + 1,)
        sum_tensor = np.zeros(sum_shape)

        slices = [[slice(None)] * len(sum_shape) for _ in tensor_units]
        for r in range(len(sum_shape)):
            if r < top_rank:
                continue
            else:
                idx = 0
                for s, tensor_unit in zip(slices, tensor_units):
                    if r != len(sum_shape) - 1:
                        if idx == 0:
                            s[top_rank-r] = slice(None, tensor_unit.lat_shape[top_rank-r])
                        else:
                            s[top_rank-r] = slice(idx, idx + tensor_unit.lat_shape[top_rank-r])
                        idx += tensor_unit.lat_shape[top_rank-r]
                    else:
                        if tensor_unit.bot_rank == 0:
                            s[top_rank-r] = slice(None, 1)
                        else:
                            s[top_rank-r] = slice(1, None)
                    
        for s, tensor_unit in zip(slices, tensor_units):
            sum_tensor[s] = tensor_unit.tensor

        return cls(sum_tensor, tensor_units[0].top_rank, 1)  



class TensorUnit():
    def __init__(self, tensor):
        super().__init__()

        self.tensor = tensor

    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def rank(self):
        return self.shape

    def contract(self, other, self_indices, other_indices):
        assert len(self_indices) == len(other_indices)
        c = len(self_indices)

        tmp_self = np.moveaxis(self.tensor, self_indices, list(range(-c, 0)))
        tmp_other = np.moveaxis(other.tensor, other_indices, list(range(c)))

        contracted_tensor = np.tensordot(tmp_self, tmp_other, c)
        return self.__class__(contracted_tensor)
    
    def extend(self, *others, indices):
        for other in others:
            assert self.rank == other.rank

        shape = []
        self_slice = []
        others_slice = [[]] * len(others)
        for i, self_s in enumerate(self.shape):
            if i in indices:
                s = self_s
                self_slice.append(slice(None, self_s))
                for j, other in enumerate(others):
                    new_s = s + other.shape[i]
                    others_slice[j].append(slice(s, new_s))
                    s = new_s
            else:
                shape.append(self_s)
                self_slice.append(slice(None))
                for j, other in enumerate(others):
                    assert self_s == other.shape[i]
                    others_slice[j].append(slice(None))
        
        extended_tensor = np.zeros(shape)
        extended_tensor[self_slice] = self.tensor
        for j, other in enumerate(others):
            extended_tensor[others_slice[j]] = other.tensor

        return self.__class__(extended_tensor)
    
    def merge_indices(self, merge_list):
        permutation = sum(merge_list, start = [])
        assert len(permutation) == self.rank
        assert set(permutation) == set(range(self.rank))

        shape = [np.prod([self.shape[idx] for idx in merge_idxs]) for merge_idxs in merge_list]
        tmp_self = np.moveaxis(self.tensor, permutation, list(range(self.rank)))

        merge_indices_tensor = tmp_self.reshape(shape)

        return self.__class__(merge_indices_tensor)

class TensorPDE(TensorUnit):
    def __init__(self, tensor, top_rank, lat_ranks, bot_rank):
        super().__init__(tensor)

        self.top_rank = top_rank
        self.lat_ranks = lat_ranks
        self.bot_rank = bot_rank

        assert self.rank == self.top_rank + sum(self.lat_ranks) + self.bot_rank

    @classmethod
    def from_pde(cls, pde, domain_derivatives_list, domain_tp_reduce, hs):
        args = [domain_derivatives_list, domain_tp_reduce, hs]
        if type(pde) == float:
            return cls()
        elif type(pde) == str:
            if pde == 'x':
                return slice(None)

        elif type(pde) == list:
            if pde[0] == 'u':
                domain_derivatives = domain_derivatives_list[len(pde) - 1]
                s = [slice(None)] + [slice(None) if x is None else x for x in pde[1:]]
                return domain_derivatives[s]
            elif pde[0] == '*':
                out = None
                for t in pde[1:]:
                    term = cls.from_pde(t, *args)
                    if out is None:
                        out = term
                    elif type(term) is float or type(out) is float:
                        out = out * term
                    else:
                        out = np.einsum('nijk,nj...,nk...->??', domain_tp_reduce, out, term)
                return out
            elif pde[0] == '+':
                out_0 = []
                out = []
                for t in pde[1:]:
                    term = cls.from_pde(t, *args)
                    if term.bot_rank == 0:
                        out_0.append(term)
                    else:
                        out.append(term)
                
                out_0 = cls.extend(out_0)
                out = cls.extend(out)
                return cls.extend([out, out_0])
            elif pde[0] == 'h':
                return hs[pde[1]]

class TensorNetwork():
    def __init__(self, bond_order, top_rank, operators, neighbors):

        self.states = np.empty((len(neighbors), top_rank) + (bond_order,) * len(neighbors[0]))
        self.operators = operators
        self.neighbors = neighbors

    @classmethod
    def from_pde(cls, pde):
        
        return cls()
