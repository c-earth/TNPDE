class TensorNetwork():
    def __init__(self, bond_order, pde, is_contravariants = None):
        self.bond_order = bond_order
        self.pde = pde
        self.is_contravariants = is_contravariants
