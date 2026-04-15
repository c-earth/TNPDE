from math import comb
import numpy as np
from scipy import integrate

from utils.tensor_network import TensorNetwork


class Basis():
    '''
    Base class of element basis in Barycentric coordinate
    '''
    def __init__(self, d, domain_rank):
        super().__init__()

        self.d = d
        self.domain_rank = domain_rank

        self.rank = self.get_rank()
        self.order2bcp = self.get_order2bcp(self.d, self.domain_rank)
        assert self.rank == len(self.order2bcp)

        self.standard_domain = self.get_standard_domain()
        self.ranges = self.get_ranges()

        self.basis_funs = self.get_basis_funs()
        self.basis_ders = self.get_basis_ders()

        self.basis_overlap = self.get_basis_overlap()
        self.inv_basis_overlap = self.get_inv_basis_overlap()
        self.derivatives = self.get_derivatives()
        self.tp_reduce = self.get_tp_reduce()


    def get_rank(self):
        return comb(self.domain_rank + self.d, self.d)
    
    @classmethod
    def get_order2bcp(cls, d, domain_rank):
        order2bcps = []
        if d == 1:
            order2bcps.append(np.array([[r] for r in range(domain_rank + 1)]))
        else:
            for r in range(domain_rank + 1):
                sub_order2bcp = cls.get_order2bcp(d - 1, domain_rank - r)
                order2bcps.append(np.concatenate([r * np.ones((sub_order2bcp.shape[0], 1)), sub_order2bcp], axis = 1))
        return np.concatenate(order2bcps, axis = 0)

    def get_standard_domain(self):
        return np.concatenate([np.eye(self.d), np.zeros((1, self.d))], axis = 0)
    
    def get_ranges(self):
        return [lambda *x: [0, 1 - sum(x)] for _ in range(self.d)]
    
    @staticmethod
    def get_barycentric_transform_parameters(domain):
        v_0 = domain[-1]
        J_x_l = (domain[:-1] - v_0).T
        return J_x_l, v_0
    
    @classmethod
    def transform(cls, T, domain, to_bary = False, is_coordinate = False, is_contravariants = None):
        J, v = cls.get_barycentric_transform_parameters(domain)
        J_inv = np.linalg.inv(J)

        if to_bary:
            v = - np.einsum('ij,j->i', J_inv, v)
            J, J_inv = J_inv, J

        if is_coordinate:
            return np.einsum('ij,...j->...i', J, T) + v
        
        elif is_contravariants is None: # assume no transformation
            return T
        
        else:
            assert len(T.shape) >= len(is_contravariants), 'not enough indices'
            out = T
            for i, is_contravariant in zip(range(-len(is_contravariants), 0), is_contravariants):
                J_axis = J if is_contravariant else J_inv.T
                out = np.swapaxes(np.einsum('ij,...j->...i', J_axis, np.swapaxes(out, i, -1)), i, -1)
            return T
        

    def get_basis_funs(self):
        raise NotImplementedError
    
    def get_basis_ders(self):
        raise NotImplementedError
    
    
    def overlap(self, *funss):
        overlap_tensor = np.empty([len(funs) for funs in funss], dtype = float)
        for idxs in np.ndindex(overlap_tensor.shape):
            overlap_tensor[idxs] = integrate.nquad(lambda *x: np.prod([funss[i][idx](x) for i, idx in enumerate(idxs)]), self.ranges)
        return overlap_tensor
    
    
    def get_basis_overlap(self):
        return self.overlap(self.basis_funs, self.basis_funs)

    def get_inv_basis_overlap(self):
        return np.linalg.inv(self.basis_overlap)
    
    
    def get_basis_der_overlap(self):
        return self.overlap(self.basis_funs, sum(self.basis_ders, start = []))
    
    def get_derivatives(self):
        return np.transpose(np.einsum('ij,jk->ik', self.inv_basis_overlap, self.get_basis_der_overlap()).reshape((self.rank, self.d, self.rank)), axes = [1, 0, 2])
    
    def diff(self, rep, dim):
        return np.einsum('ij,j->i', self.derivatives[dim], rep)

    
    def get_basis_tp_reduce_overlap(self):
        return self.overlap(self.basis_funs, self.basis_funs, self.basis_funs)
    
    def get_tp_reduce(self):
        return np.einsum('ij,jkl->ikl', self.inv_basis_overlap, self.get_basis_tp_reduce_overlap())
    
    def tp(self, rep_1, rep_2):
        return np.einsum('ijk,j,k->i', self.tp_reduce, rep_1, rep_2)
    
    
    def basis_fun_overlap(self, fun):
        return self.overlap(self.basis_funs, fun).flatten()
    
    def element_fun2rep(self, fun):
        return np.einsum('ij,j->i', self.inv_basis_overlap, self.basis_fun_overlap(fun))
    
    def element_rep2fun(self, rep):
        return lambda x: sum([r * basis_fun(x) for r, basis_fun in zip(rep, self.basis_funs)])


class LagrangeBasis(Basis):
    '''
    Lagrange orthogonal polynomial basis
    '''
    def get_basis_fun(self, bcp):
        def fun(x):
            x = self.domain_rank * np.array(x).reshape((-1, self.d))
            x0 = self.domain_rank - x.sum(axis = 1)
            out = np.ones(x.shape[0])
            for i in range(self.d):
                for c in range(bcp[i]):
                    out *= (x[:, i] - c) / (bcp[i] - c)

            b0 = self.domain_rank - bcp.sum()
            for c in range(b0):
                out *= (x0 - c) / (b0 - c)
            return out
        return fun
    
    def get_basis_funs(self):
        return [self.get_basis_fun(bcp) for bcp in self.order2bcp]
    

    def get_basis_der(self, bcp, dim):
        def der(x):
            x = self.domain_rank * np.array(x).reshape((-1, self.d))
            x0 = self.domain_rank - x.sum(axis = 1)
            b0 = self.domain_rank - bcp.sum()

            out = np.ones(x.shape[0])
            for i in range(self.d):
                if i != dim:
                    for c in range(bcp[i]):
                        out *= (x[:, i] - c) / (bcp[i] - c)

            out_dim = np.zeros(x.shape[0])
            for is_b0, c_dim in zip(bcp[dim] * [0] + b0 * [1], list(range(bcp[dim])) + list(range(b0))):
                out_tmp = np.ones(x.shape[0])

                for c in range(bcp[i]):
                    if is_b0 or c != c_dim:
                        out_tmp *= (x[:, i] - c) 
                    out_tmp /= (bcp[i] - c)

                for c in range(b0):
                    if (not is_b0) or c != c_dim:
                        out_tmp *= (x0 - c)
                    else:
                        out_tmp *= -1
                    out_tmp /= (b0 - c)

                out_dim += out_tmp

            return out * out_dim
        return der
    
    def get_basis_ders(self):
        return [[self.get_basis_der(bcp, dim) for bcp in self.order2bcp] for dim in self.d]
        

class FiniteElement():
    '''
    Finite element manager of triangulation domains
    '''
    
    def __init__(self, solver, pde, triangulation, bond_order = 1, domain_rank = 1, basis_class = LagrangeBasis, is_contravariants = None):
        super().__init__()
        
        self.triangulation = triangulation
        self.bond_order = bond_order
        self.domain_rank = domain_rank

        self.basis = basis_class(self.triangulation.d, domain_rank)
        self.is_contravariants = is_contravariants

        self.d = self.triangulation.d
        self.n = len(self.triangulation.simplices)

        self.solver = solver(self.d)
        self.pde = pde

        self.tensor_network = TensorNetwork(self.bond_order, self.pde, self.is_contravariants)

        self.domain_derivatives = self.get_domain_derivatives()

    def get_domain_derivatives(self):
        domain_derivatives = []
        for domain_simplex in self.triangulation.simplices:
            domain = self.triangulation.points[domain_simplex]
            derivatives = self.basis.transform(np.transpose(self.basis.derivatives, axes = [1, 2, 0]), domain, to_bary = False, is_contravariants = [False])
            domain_derivatives.append(np.transpose(derivatives, axes = [2, 0, 1]))
        return domain_derivatives
    

    def diff(self, domain_rep, element, dim):
        return np.einsum('ij,j->i', self.domain_derivatives[element][dim], domain_rep)
       
    def tp(self, domain_rep_1, domain_rep_2):
        return self.basis.tp(domain_rep_1, domain_rep_2)
    

    def fun2rep(self, fun, is_contravariants = None):
        rep = []
        for domain_simplex in self.triangulation.simplices:
            domain = self.triangulation.points[domain_simplex]

            def element_fun(u):
                x = self.basis.transform(u, domain, to_bary = False, is_coordinate = True)
                f = fun(x)
                return self.basis.transform(f, domain, to_bary = True, is_contravariants = is_contravariants)
            
            rep.append(self.basis.element_fun2rep(element_fun))
        return rep


    def rep2fun(self, rep, is_contravariants = None):
        def fun(x):
            domain_idx = self.triangulation.find_simplex(x)
            domain = self.triangulation.points[self.triangulation.simplices[domain_idx]]
            element_fun = self.basis.element_rep2fun(rep[domain_idx])
            u = self.basis.transform(x, domain, to_bary = True, is_coordinate = True)
            return self.basis.transform(element_fun(u), domain, to_bary = False, is_contravariants = is_contravariants)
        return fun

