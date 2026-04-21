class PDE():
    '''
    `pde_txt` example:
    1. 5: `5`
    2. 8x: `8*h_0` where `hs[0] = lambda *x: x[0]`
    3. du/dx: `D[1, 0]` or `D[1]` (for 1D problem)
    5. u+9du/dy: `u+9*D[1, 1]`
    6. 8(u+x-y): `8*(u+h_0+-1*h_1)` where `hs[0] = lambda *x: x[0]`, and `hs[1] = lambda *x: x[1]`
    '''
    operators = set('+*()')

    def __init__(self, pde_txt):
        super().__init__()
        # shape must be tuple/list not array/tensor
        self.pde_txt = pde_txt
        self.pde = self.parse(self.pde_txt)

    @classmethod
    def parse_element(cls, ele_txt):
        try:
            ele = float(ele_txt)
        except ValueError:
            if ele_txt[0:2] == 'D[' and ele_txt[-1] == ']':
                arguments = ele_txt[2:-1].split(',')
                ele = ['D'] + [cls.parse_element(argument) for argument in arguments]
            elif '_' in ele_txt:
                ele = ele_txt.split('_')
                ele[1:] = [int(e) for e in ele[1:]]
            else:
                ele = ele_txt
        return ele
    
    @classmethod
    def clean_parse(cls, priority_out):
        if type(priority_out) == float:
            return priority_out
        
        elif type(priority_out[0]) == float:
            return priority_out[0]
        
        elif priority_out[0] == '+':
            clean_out = ['+']
            c = 0.0
            for p in priority_out[1:]:
                p = cls.clean_parse(p)
                if type(p) == float:
                    c += p
                else:
                    clean_out.append(p)
            if c != 0.0:
                clean_out.append(c)
            
            if len(clean_out) == 1:
                return c
            elif len(clean_out) == 2:
                return clean_out[-1]
            else:
                return clean_out
            
        elif priority_out[0] == '*':
            clean_out = ['*']
            c = 1.0
            for p in priority_out[1:]:
                p = cls.clean_parse(p)
                if type(p) == float:
                    c *= p
                else:
                    clean_out.append(p)
            if c != 1.0:
                clean_out.append(c)

            if len(clean_out) == 1:
                return c
            elif len(clean_out) == 2:
                return clean_out[-1]
            else:
                return clean_out
        else:
            return priority_out

    @classmethod
    def parse(cls, pde_txt):
        pde_txt = pde_txt.replace(' ', '')
        if pde_txt == '':
            return []
        elif set(pde_txt).intersection(cls.operators):
            part = ''
            op = 0
            out = []
            for c in pde_txt:
                if op == 1 and c == ')':
                    op = 0
                    out.append(cls.parse(part))
                    part = ''
                elif op == 0:
                    if c == '(':
                        op += 1
                    elif c == ')':
                        raise ValueError('not matching parentheses')
                    elif c in '+*':
                        if part != '':
                            out.append(cls.parse_element(part))
                            part = ''
                        out.append(c)                        
                    else:
                        part += c
                else:
                    part += c
            if part != '':
                out.append(cls.parse_element(part))

            priority_out = []
            i = 0
            while i < len(out):
                if out[i] == '*':
                    if priority_out[0] not in ['+', '*']:
                        priority_out = ['*'] + priority_out + [out[i+1]]
                    elif priority_out[0] == '+':
                        if type(priority_out[-1]) == list and priority_out[-1][0] == '*':
                            priority_out[-1].append(out[i+1])
                        else:
                            priority_out[-1] = ['*', priority_out[-1], out[i+1]]
                    else:
                        priority_out = ['*'] + priority_out + [out[i+1]]
                    i += 1
                elif out[i] == '+':
                    if priority_out[0] not in ['+', '*']:
                        priority_out = ['+'] + priority_out
                    elif priority_out[0] == '*':
                        priority_out = ['+'] + [priority_out]
                else:
                    priority_out.append(out[i])
                i += 1
            return cls.clean_parse(priority_out)
        else:
            return cls.clean_parse([cls.parse_element(pde_txt)])
