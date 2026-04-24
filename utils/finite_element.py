from math import comb
import itertools

import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt


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
        self.bcp2order = dict([(tuple(bcp), order) for order, bcp in enumerate(self.order2bcp)])
        self.side_bcps = self.get_side_bcps()
        self.side_orders = self.get_side_orders()
        self.permuted_order_maps = self.get_permuted_order_maps()
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
                order2bcps.append(np.concatenate([r * np.ones((sub_order2bcp.shape[0], 1), dtype = int), sub_order2bcp], axis = 1))
        return np.concatenate(order2bcps, axis = 0)
    
    def get_side_bcps(self):
        side_bcps = [[] for _ in range(self.d + 1)]
        for bcp in self.order2bcp:
            if sum(bcp) == self.domain_rank:
                side_bcps[0].append(bcp)
            for side in range(1, self.d + 1):
                if bcp[side - 1] == 0:
                    side_bcps[side].append(bcp)
        return np.array(side_bcps)
    
    def get_side_orders(self):
        side_orders = []
        for neighbor_idx in range(self.d + 1):
            side_orders.append([self.bcp2order[tuple(bcp)] for bcp in self.side_bcps[neighbor_idx]])
        return np.array(side_orders)
    

    def get_permuted_order_maps(self):
        permuted_order_maps = dict()
        for simplex_permutation in itertools.permutations(list(range(self.d + 1))):
            order2bcp_with_0_points = np.concatenate([self.order2bcp, self.domain_rank - np.sum(self.order2bcp, axis = 1, keepdims = True)], axis = 1)
            
            permuted_order2bcp_with_0_points = order2bcp_with_0_points[:, simplex_permutation[:-1]]
            permuted_order_map = [-1] * self.rank
            for order, bcp in enumerate(permuted_order2bcp_with_0_points):
                permuted_order_map[self.bcp2order[tuple(bcp)]] = order

            permuted_order_maps[tuple(simplex_permutation)] = np.array(permuted_order_map)
        return permuted_order_maps


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
            return out
        

    def get_basis_funs(self):
        raise NotImplementedError
    
    def get_basis_ders(self):
        raise NotImplementedError
    
    
    def overlap(self, *funss):
        overlap_tensor = np.empty([len(funs) for funs in funss], dtype = float)
        for idxs in np.ndindex(overlap_tensor.shape):
            overlap_tensor[idxs] = integrate.nquad(lambda *x: np.prod([funss[i][idx](x) for i, idx in enumerate(idxs)]), self.ranges)[0]
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
        return self.overlap(self.basis_funs, [fun]).flatten()
    
    def element_fun2rep(self, fun):
        return np.einsum('ij,j->i', self.inv_basis_overlap, self.basis_fun_overlap(fun))
    
    def element_rep2fun(self, rep):
        return lambda x: sum([r * basis_fun(x) for r, basis_fun in zip(rep, self.basis_funs)])
    

    def visualize(self):
        if self.d == 1:
            x = np.linspace(0, 1, 20)
            plt.figure()
            for fun in self.basis_funs:
                plt.plot(x, fun(x))
            plt.show()
        elif self.d == 2:
            xs, ys = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
            fig, axs = plt.subplots(self.domain_rank + 1, self.domain_rank + 1, figsize = ((self.domain_rank + 1)*2, (self.domain_rank + 1)*2))
            for ax in axs.ravel():
                ax.set_axis_off()
            for i, fun in enumerate(self.basis_funs):
                ax = axs[*self.order2bcp[i]]

                zs = np.array([[fun(point)[0] if point[0] >= 0 and point[1] >= 0 and np.sum(point) <= 1 else 0 for point in zip(xr, yr)] for xr, yr in zip(xs, ys)])
                zs = zs.reshape((20, 20)).T
                ax.imshow(zs, cmap = 'viridis', interpolation = 'nearest')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
            fig.tight_layout()
            fig.show()
        else:
            raise NotImplementedError()


class LagrangeBasis(Basis):
    '''
    Lagrange orthogonal polynomial basis
    '''
    def __init__(self, d, domain_rank, mock = False):
        if not mock:
            super().__init__(d, domain_rank)
        else:
            self.d = d
            self.domain_rank = domain_rank

            self.rank = self.get_rank()
            self.order2bcp = self.get_order2bcp(self.d, self.domain_rank)
            assert self.rank == len(self.order2bcp)

            self.basis_funs = self.get_basis_funs()


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
                    else:
                        out_tmp *= self.domain_rank
                    out_tmp /= (bcp[i] - c)

                for c in range(b0):
                    if (not is_b0) or c != c_dim:
                        out_tmp *= (x0 - c)
                    else:
                        out_tmp *= -self.domain_rank
                    out_tmp /= (b0 - c)

                out_dim += out_tmp

            return out * out_dim
        return der
    
    def get_basis_ders(self):
        return [[self.get_basis_der(bcp, dim) for bcp in self.order2bcp] for dim in range(self.d)]
        

class FiniteElement():
    '''
    Finite element manager of triangulation domains
    '''
    
    def __init__(self, triangulation, basis):
        super().__init__()
        self.triangulation = triangulation
        self.basis = basis
        self.domain_derivatives_list = [np.broadcast_to(np.eye(self.basis.rank), 
                                                        (self.triangulation.n, self.basis.rank, self.basis.rank)), 
                                        self.get_domain_derivatives()]
        self.neighbor_maps = self.get_neighbor_maps()
        self.order_maps = self.get_order_maps()
        self.u_shape = None
        self.d_con_bc_operators = None
        self.n_con_bc_operators = None
        self.con_bc_operators = None
        self.env_bc_operators = None

    def get_domain_derivatives(self):
        domain_derivatives = []
        for domain_simplex in self.triangulation.simplices:
            domain = self.triangulation.points[domain_simplex]
            derivatives = self.basis.transform(np.transpose(self.basis.derivatives, axes = [1, 2, 0]), domain, to_bary = False, is_contravariants = [False])
            domain_derivatives.append(np.transpose(derivatives, axes = [2, 0, 1]))
        return np.array(domain_derivatives)
    

    def diff(self, domain_rep, element, dim):
        return np.einsum('ij,j->i', self.domain_derivatives_list[1][element][dim], domain_rep)
    
    def all_diff(self, rep, dim):
        return np.array([self.diff(domain_rep, element, dim) for element, domain_rep in enumerate(rep)])
       
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
        return np.array(rep)


    def rep2fun(self, reps, is_contravariants = None):
        element_funs = [self.basis.element_rep2fun(rep) for rep in reps]
        domains = self.triangulation.points[self.triangulation.simplices]
        def fun(x):
            x = np.array([x])
            domain_idx = self.triangulation.find_simplex(x)[0]
            u = self.basis.transform(x, domains[domain_idx], to_bary = True, is_coordinate = True)
            return self.basis.transform(element_funs[domain_idx](u), domains[domain_idx], to_bary = False, is_contravariants = is_contravariants)[0]
        return np.vectorize(fun)
    

    def get_neighbor_maps(self):
        neighbor_maps = []
        for element in range(self.triangulation.n):
            neighbor_maps.append(self.get_element_neighbor_maps(element))

        return np.array(neighbor_maps)
    

    def get_order_maps(self):
        def fun(x):
            x = tuple([int(c) for c in x])
            if -1 in x:
                return self.basis.permuted_order_maps[tuple(range(self.basis.d + 1))]
            else:
                return self.basis.permuted_order_maps[x] 
        return np.apply_along_axis(fun, axis = -1, arr = self.neighbor_maps)
    

    def get_element_neighbor_maps(self, element):
        neighbors = self.triangulation.neighbors[element]

        neighbor_maps = - np.ones((self.basis.d + 1, self.basis.d + 1))
        for neighbor_idx, neighbor in enumerate(neighbors):
            if neighbor != -1:
                element_simplex = self.triangulation.simplices[element]
                neighbor_simplex = self.triangulation.simplices[neighbor]

                for element_i, element_point in enumerate(element_simplex):
                    if element_i == neighbor_idx:
                        continue

                    for neighbor_i, neighbor_point in enumerate(neighbor_simplex):
                        if neighbor_point == element_point:
                            neighbor_maps[neighbor_idx][neighbor_i] = element_i
                
                assert np.sum(neighbor_maps[neighbor_idx] == -1) == 1

                neighbor_maps[neighbor_idx][neighbor_maps[neighbor_idx] == -1] = neighbor_idx
        return neighbor_maps
    

    def calculate_higher_domain_derivatives(self, derivative_order):
        while len(self.domain_derivatives_list) < derivative_order + 1:
            self.domain_derivatives_list.append(np.einsum('ndij,n...jk->nd...ik', 
                                                          self.domain_derivatives_list[1], 
                                                          self.domain_derivatives_list[-1]))


    def set_con_bc_operators(self, con_order):
        d_con_bc_operators = None
        n_con_bc_operators = None

        if con_order is not None:
            d_con_bc_operators = []
            n_con_bc_operators = []
            self.calculate_higher_domain_derivatives(con_order)
            for domain_derivatives in self.domain_derivatives_list:
                domain_side_derivatives = np.moveaxis(domain_derivatives[..., self.basis.side_orders, :], -3, 1)

                neighbor_derivatives_std_order = domain_derivatives[self.triangulation.neighbors]
                tmp = neighbor_derivatives_std_order[np.arange(self.triangulation.n)[:, None, None], 
                                                     np.arange(self.basis.d + 1)[None, :, None], 
                                                     ..., self.order_maps, :]
                
                neighbor_derivatives_dmn_order = np.moveaxis(tmp, 2, -2)
                neighbor_side_derivatives = np.moveaxis(neighbor_derivatives_dmn_order[:, np.arange(self.basis.d + 1)[:, None], ... , self.basis.side_orders, :], [0, 1], [1, -2])

                d_shape = domain_side_derivatives.shape
                d_shape = d_shape[:2] + (-1,) + d_shape[-1:]
                n_shape = neighbor_side_derivatives.shape
                n_shape = n_shape[:2] + (-1,) + n_shape[-1:]

                d_con_bc_operators.append(domain_side_derivatives.reshape(d_shape))
                n_con_bc_operators.append(neighbor_side_derivatives.reshape(n_shape))
        
            d_con_bc_operators = np.concatenate(d_con_bc_operators, axis = 2)
            n_con_bc_operators = np.concatenate(n_con_bc_operators, axis = 2)

        self.d_con_bc_operators = d_con_bc_operators
        self.n_con_bc_operators = n_con_bc_operators

        if self.u_shape is not None:
            self.set_u_shape(self.u_shape)
        else:
            self.u_shape = tuple()
            self.set_con_bc_operators_with_u_shape()


    def set_con_bc_operators_with_u_shape(self):
        d_con_bc_operators = np.kron(np.eye(int(np.prod(self.u_shape))), self.d_con_bc_operators)
        n_con_bc_operators = np.kron(np.eye(int(np.prod(self.u_shape))), self.n_con_bc_operators)

        d_con_bc_operators = scipy.linalg.block_diag(d_con_bc_operators, np.ones(d_con_bc_operators.shape[:-2] + (1, 1)))
        n_con_bc_operators = scipy.linalg.block_diag(n_con_bc_operators, np.ones(n_con_bc_operators.shape[:-2] + (1, 1)))

        assert d_con_bc_operators.shape == n_con_bc_operators.shape

        r = d_con_bc_operators.shape[-2] - 1
        subtraction_tensor = np.zeros((r, r + 1, r + 1))
        for i in range(r):
            subtraction_tensor[i, i, r] = 1
            subtraction_tensor[i, r, i] = -1

        
        con_bc_operators = np.einsum('ijk,ndjp,ndkq->ndipq', subtraction_tensor, d_con_bc_operators, n_con_bc_operators)
        con_bc_operators = np.einsum('ndipq,ndijk->ndpqjk', con_bc_operators, con_bc_operators)

        self.con_bc_operators = con_bc_operators

    def set_u_shape(self, u_shape):
        self.u_shape = u_shape
        self.set_con_bc_operators_with_u_shape()

        
    def set_env_bc_operators(self, env_bcs = None):
        self.env_bc_operators = None

        if env_bcs is not None:
            self.env_bc_operators = dict()
            for element, element_env_bcs in env_bcs.items():
                self.env_bc_operators[element] = self.env_bc_operators.get(element, dict())
                for neighbor_idx, side_env_bcs in element_env_bcs.items():
                    env_bc_matrices, env_bc_vector = side_env_bcs
                    self.calculate_higher_domain_derivatives(len(env_bc_matrices))

                    env_bc_cummulant = None
                    for domain_derivatives, env_bc_matrix in zip(self.domain_derivatives_list, env_bc_matrices):
                        derivatives = domain_derivatives[element]
                        env_bc_derivative_cummulant = np.tensordot(env_bc_matrix, derivatives[..., self.basis.side_orders[neighbor_idx], :], axes = len(env_bc_matrix.shape) - 1)
                        if env_bc_cummulant is None:
                            env_bc_cummulant = env_bc_derivative_cummulant
                        else:
                            env_bc_cummulant += env_bc_derivative_cummulant

                    assert env_bc_cummulant is not None

                    
                    env_bc_operator = np.concatenate([env_bc_cummulant, -env_bc_vector], axis = -1).reshape((env_bc_cummulant.shape[0], -1))

                    self.env_bc_operators[element][neighbor_idx] = np.einsum('mi,mj->ij', env_bc_operator, env_bc_operator)
