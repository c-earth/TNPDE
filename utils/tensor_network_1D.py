class PDF1D():
    def __init__(self, function = 'u-x*u_x', symbol = 'u', parameter = 'x'):
        self.function = function
        self.symbol = symbol
        self.parameter = parameter
        self.pdf = self.parse()


    def parse(self):
        pass

