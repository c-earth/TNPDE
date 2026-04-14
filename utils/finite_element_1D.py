import numpy as np
from scipy.sparse import bsr_array
from scipy.sparse.linalg import inv
from scipy.special import factorial


class Domains1D():
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__()
        
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

    @property
    def shape(self):
        return self.lower_bounds.shape

    @property
    def size(self):
        return self.lower_bounds.size

    def __repr__(self):
        return f'{self.__class__.__name__}({self.lower_bounds.tolist()}, {self.upper_bounds.tolist()})'
    
    def __getitem__(self, subscript):
        sub_lower_bounds = self.lower_bounds.__getitem__(subscript)
        sub_upper_bounds = self.upper_bounds.__getitem__(subscript)
        return Domains1D(sub_lower_bounds, sub_upper_bounds)
    
    @property
    def periods(self):
        return self.upper_bounds - self.lower_bounds
    
    def include(self, points):
        return (points > self.lower_bounds[..., *([None] * len(points.shape))]) & (points < self.upper_bounds[..., *([None] * len(points.shape))])
    
    @classmethod
    def domain_overlap(cls, self, other):
        return cls(np.maximum(self.lower_bounds[..., *([None]*len(other.shape))], other.lower_bounds[*([None]*len(self.shape)), ...]),\
                   np.minimum(self.upper_bounds[..., *([None]*len(other.shape))], other.upper_bounds[*([None]*len(self.shape)), ...]))


class Basis1D():
    def __init__(self, rank, domains, domain_adjacency, res):
        super().__init__()

        self.rank = rank
        self.domains = domains
        self.domain_adjacency = domain_adjacency
        self.res = res
        self.basis_overlap = self.get_basis_overlap()
        self.basis_der_overlap = self.get_basis_der_overlap()
        self.derivative = self.get_derivative()
        self.basis_tp_overlap = self.get_basis_tp_overlap()
        self.tp_reduce = self.get_tp_reduce()

    def __repr__(self):
        return f'{self.__class__.__name__}(rank: {self.rank}, shape: {self.domains.shape})'
    
    @property
    def dim(self):
        raise NotImplementedError

    @property
    def size(self):
        return self.domains.size * self.dim

    def get_fun(self, *_):
        raise NotImplementedError
    
    def get_der(self, *_):
        raise NotImplementedError

    def get_fun_tp(self, fun_1, fun_2):
        def fun_tp(points):
            return (fun_1(points).reshape(-1, self.dim, 1) * fun_2(points).reshape(-1, 1, self.dim))
        return fun_tp

    def diff(self, rep):
        return self.derivative @ rep

    def tp(self, rep_1, rep_2):
        rep_1 = rep_1.reshape(-1, self.dim)
        rep_2 = rep_2.reshape(-1, self.dim)
        rep_tp = np.einsum('ij,kl->ikjl', rep_1, rep_2).flatten()
        return self.tp_reduce @ rep_tp

    def overlap(self, fun, conj_fun, domains):
        points = np.linspace(domains.lower_bounds, domains.upper_bounds, self.res)
        out = fun(points).reshape(*points.shape, 1, -1)
        conj_out = conj_fun(points).reshape(*points.shape, -1, 1)
        return np.trapezoid(conj_out.conj() * out, points[..., None, None], axis = 0)
    
    def basis_fun_overlap(self, fun):
        return self.overlap(fun, self.get_fun(self.domains), self.domains).flatten()
    
    def get_basis_overlap(self):
        data, indices, indptr = [], [], [0]
        for domain, domain_adjacency_row in zip(self.domains, self.domain_adjacency):
            adjacent_domains = self.domains[domain_adjacency_row]
            domain_overlaps = Domains1D.domain_overlap(domain, adjacent_domains)
            data.append(self.overlap(self.get_fun(adjacent_domains), self.get_fun(domain), domain_overlaps))
            indices += domain_adjacency_row
            indptr.append(len(indices))
        return bsr_array((np.concatenate(data, axis = 0), indices, indptr)).tocsc()
    
    def get_basis_der_overlap(self):
        data, indices, indptr = [], [], [0]
        for domain, domain_adjacency_row in zip(self.domains, self.domain_adjacency):
            adjacent_domains = self.domains[domain_adjacency_row]
            domain_overlaps = Domains1D.domain_overlap(domain, adjacent_domains)
            data.append(self.overlap(self.get_der(adjacent_domains), self.get_fun(domain), domain_overlaps))
            indices += domain_adjacency_row
            indptr.append(len(indices))
        return bsr_array((np.concatenate(data, axis = 0), indices, indptr)).tocsc()
    
    def get_basis_tp_overlap(self):
        data, indices, indptr = [], [], [0]
        for domain, domain_adjacency_row in zip(self.domains, self.domain_adjacency):
            for domain_adjacency_1 in domain_adjacency_row:
                for domain_adjacency_2 in self.domain_adjacency[domain_adjacency_1]:
                    if not domain_adjacency_2 in domain_adjacency_row:
                        continue
                    

                    adjacent_domain_1 = self.domains[domain_adjacency_1]
                    adjacent_domain_2 = self.domains[domain_adjacency_2]

                    domain_overlap = Domains1D.domain_overlap(domain, Domains1D.domain_overlap(adjacent_domain_1, adjacent_domain_2))

                    data.append(self.overlap(self.get_fun_tp(self.get_fun(adjacent_domain_1), self.get_fun(adjacent_domain_2)), self.get_fun(domain), domain_overlap))
                    indices.append(domain_adjacency_1 * self.domains.size + domain_adjacency_2)

            indptr.append(len(indices))

        return bsr_array((data, indices, indptr), shape = (self.size, self.size ** 2)).tocsc()
    
    def get_derivative(self):
        return inv(self.basis_overlap) @ self.basis_der_overlap

    def get_tp_reduce(self):
        return inv(self.basis_overlap) @ self.basis_tp_overlap
    
    def fun2rep(self, fun):
        return inv(self.basis_overlap) @ self.basis_fun_overlap(fun)
    
    def rep2fun(self, rep):
        def fun(points):
            out = np.zeros_like(points)
            for domain, domain_rep in zip(self.domains, rep.reshape(-1, self.dim)):
                include = domain.include(points)
                out[include] += np.sum(self.get_fun(domain)(points[include])*domain_rep, axis = -1)
            return out
        return fun


class QHOBasis1D(Basis1D):
    @property
    def dim(self):
        return self.rank
    
    @property
    def ranks(self):
        return np.arange(self.rank)
    
    @staticmethod
    def norm(n):
        return np.sqrt(2 ** n * factorial(n) * np.sqrt(np.pi))
    
    @staticmethod
    def hermite(n, points):
        return np.polynomial.hermite.Hermite.basis(n)(points)
    
    @classmethod
    def norm_hermite(cls, n, points):
        return cls.hermite(n, points) / cls.norm(n)
    
    @classmethod
    def qho(cls, rel_points, sigmas, rank):
        scale_rel_points = rel_points / sigmas
        return np.stack([np.exp(- scale_rel_points ** 2 / 2) / np.sqrt(sigmas) * cls.norm_hermite(n, scale_rel_points) for n in range(rank)], axis = -1)
    
    @classmethod
    def qho_der(cls, rel_points, sigmas, rank):
        scale_rel_points = rel_points / sigmas
        qho_ders = []
        for n in range(rank):
            hermite_der = np.sqrt(n + 1) * cls.norm_hermite(n + 1, scale_rel_points)
            if n != 0:
                hermite_der -= np.sqrt(n) * cls.norm_hermite(n - 1, scale_rel_points)
            qho_ders.append(- np.exp(- scale_rel_points ** 2 / 2) / np.sqrt(sigmas) * hermite_der / np.sqrt(2) / sigmas)
        return np.stack(qho_ders, axis = -1)
    
    @staticmethod
    def taper(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        return (np.ones(x.shape) * mask)[..., None]
    
    @staticmethod
    def taper_der(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        return (np.zeros(x.shape) * mask)[..., None]
    
    def get_fun(self, domains):
        def fun(points):
            assert domains.shape in [points.shape[-len(domains.shape):], tuple()]
            rel_points = (points - (domains.lower_bounds + domains.upper_bounds) / 2)

            return self.qho(rel_points, domains.periods/16, self.rank) * self.taper(rel_points, domains)
        return fun
    
    def get_der(self, domains):
        def fun(points):
            assert domains.shape in [points.shape[-len(domains.shape):], tuple()]
            rel_points = (points - (domains.lower_bounds + domains.upper_bounds) / 2)

            return self.qho_der(rel_points, domains.periods/16, self.rank) * self.taper(rel_points, domains) + self.qho(rel_points, domains.periods/2, self.rank) * self.taper_der(rel_points, domains)
        return fun
    

class TaperedQHOBasis1D(QHOBasis1D):
    @staticmethod
    def taper(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        return ((1 - np.abs(x)) * mask)[..., None]
    
    @staticmethod
    def taper_der(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        return - ((2 * np.heaviside(x, 0.5) - 1) * mask)[..., None]
    

class SmoothBumpQHOBasis1D(TaperedQHOBasis1D):
    @staticmethod
    def taper(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        out = np.zeros(x.shape)
        out[mask] = np.exp(-1/(1 - x[mask] ** 2))
        return out[..., None]
    
    @staticmethod
    def taper_der(rel_points, domains):
        x = 2 * rel_points / domains.periods
        mask = np.abs(x) < 1
        out = np.zeros(x.shape)
        out[mask] = np.exp(-1/(1 - x[mask] ** 2)) * -2 * x[mask] / (1 - x[mask] ** 2) ** 2
        return out[..., None]


class FiniteElement1D():
    def __init__(self, basis):
        super().__init__()
        self.basis = basis

    def __repr__(self):
        return f'{self.__class__.__name__}({self.basis})'

    @classmethod
    def from_uniform_grid(cls, domain, element_num, basis_class, rank, res):
        step_size = domain.periods / element_num
        lower_bounds = []
        upper_bounds = []
        basis_domain_adjacency = []
        for i in range(element_num + 1):
            lower_bounds.append(domain.lower_bounds + (i - 1) * step_size)
            upper_bounds.append(domain.lower_bounds + (i + 1) * step_size)
            if i == 0:
                basis_domain_adjacency.append([i, i + 1])
            elif i == element_num:
                basis_domain_adjacency.append([i - 1, i])
            else:
                basis_domain_adjacency.append([i - 1, i, i + 1]) 
        basis_domains = Domains1D(np.concatenate(lower_bounds), np.concatenate(upper_bounds))
        return cls(basis_class(rank, basis_domains, basis_domain_adjacency, res))
