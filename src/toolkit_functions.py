from math import pi, e
from typing import overload

import torch
from torch import Tensor
from torch import empty as empty
from torch import sqrt, prod, sin, cos
from torch import sum as t_sum
from torch.nn import Parameter

torch.manual_seed(69420)

torch.clear_autocast_cache()
torch.cuda.empty_cache()
torch.set_default_device('cuda')




class TestFunction:
    def __init__(self, params: Tensor = None):
        self.params = Parameter(params)


    @overload
    def loss(self) -> Tensor:
        ...


    def loss(self) -> Tensor:
        raise NotImplementedError




class Rastrigin(TestFunction):
    def __init__(self, params: Tensor = None, num_params: int = 2, a: float = 10.0):

        if num_params < 2:
            ValueError('Custom defined num_params must be > 1!')

        if params is not None and params.numel() < 2:
            ValueError('Custom defined params.numel() must > 1!')

        if params is None:
            if num_params == 2:
                params = Parameter(Tensor([-4.0, 4.0]))
            else:
                params = Parameter(empty((num_params,)).fill_(-4.0))

        self.a = a

        super().__init__(params)


    def loss(self) -> Tensor:
        return self.a * self.params.numel() + t_sum(self.params ** 2 - self.a * cos(2 * pi * self.params))




class Ackley(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-4.0, 4.0]))

        if params.numel() != 2:
            raise ValueError("Ackley requires a parameter tensor of length 2!")

        super().__init__(params)


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        term_1 = -20 * torch.exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2)))
        term_2 = -torch.exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
        return term_1 + term_2 + e + 20




class Sphere(TestFunction):
    def __init__(self, params: Tensor = None, num_params: int = 2):

        if params is None and num_params < 2:
            ValueError('Custom defined num_params must be > 1!')

        if params is not None and params.numel() < 2:
            ValueError('Custom defined params.numel() must > 1!')

        if params is None:
            if num_params == 2:
                params = Parameter(Tensor([-2.0, 2.0]))
            else:
                params = Parameter(empty((num_params,)).fill_(-2.0))

        super().__init__(params)


    def loss(self) -> Tensor:
        return t_sum(self.params ** 2)




class Rosenbrock(TestFunction):
    def __init__(self, params: Tensor = None, num_params: int = 2):

        if params is None and num_params < 2:
            ValueError('Custom defined num_params must be > 1!')

        if params is not None and params.numel() < 2:
            ValueError('Custom defined params.numel() must > 1!')

        if params is None:
            if num_params == 2:
                params = Parameter(Tensor([-1.67, 2.5]))
            else:
                params = Parameter(empty((num_params,)).uniform_(-1.67, 2.5))

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return t_sum(100.0 * (y - x ** 2) ** 2 + (1 - x) ** 2)




class Beale(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-2.0, 2.0]))

        if params.numel() != 2:
            raise ValueError("Beale requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return ((1.5 - x + x * y) ** 2 +
                (2.25 - x + x * y ** 2) ** 2 +
                (2.625 - x + x * y ** 3))




class GoldsteinPrice(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([2.0, 1.0]))

        if params.numel() != 2:
            raise ValueError("Goldstein-Price requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        term_1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
        term_2 = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
        return term_1 * term_2




class Booth(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-7.5, -7.5]))

        if params.numel() != 2:
            raise ValueError("Booth requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2




class BukinN6(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-14.0, 4.0]))

        if params.numel() != 2:
            raise ValueError("Bukin N.6 requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return 100 * sqrt(torch.abs(y - 0.01 * x ** 2)) + 0.01 * torch.abs(x + 10)




class Matyas(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-10.0, 2.5]))

        if params.numel() != 2:
            raise ValueError("Matyas requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y




class LeviN13(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-4.0, -4.0]))

        if params.numel() != 2:
            raise ValueError("Levi N.13 requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return (sin(3 * pi * x) ** 2 +
                (x - 1) ** 2 * (1 + sin(3 * pi * y) ** 2) +
                (y - 1) ** 2 * (1 + sin(2 * pi * y) ** 2))




class Griewank(TestFunction):
    def __init__(self, params: Tensor = None, num_params: int = 2):

        if params is None and num_params < 2:
            ValueError('Custom defined num_params must be > 1!')

        if params is not None and params.numel() < 2:
            ValueError('Custom defined params.numel() must > 1!')

        if params is None:
            params = Parameter(empty((num_params,)).fill_(-7.5))


        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        n = self.params.numel()
        sum_term = t_sum(self.params ** 2) / 4000.0
        indices = torch.arange(1, n + 1, dtype=self.params.dtype, device=self.params.device)
        prod_term = prod(cos(self.params / sqrt(indices)))
        return 1 + sum_term - prod_term




class Himmelblau(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([0.0, 0.0]))

        if params.numel() != 2:
            raise ValueError("Himmelblau requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2




class ThreeHumpCamel(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-4.0, -4.0]))

        if params.numel() != 2:
            raise ValueError("Three-Hump Camel requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6) / 6 + x * y + y ** 2




class Easom(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([0.0, 0.0]))

        if params.numel() != 2:
            raise ValueError("Easom requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return -cos(x) * cos(y) * torch.exp(-((x - pi) ** 2 + (y - pi) ** 2))




class CrossInTray(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-5.0, 5.0]))

        if params.numel() != 2:
            raise ValueError("Cross-In-Tray requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        inner = torch.abs(sin(x) * cos(y) *
                          torch.exp(torch.abs(100 - sqrt(x ** 2 + y ** 2) / pi))) + 1
        return -0.0001 * (inner ** 0.1)




class Eggholder(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-500.0, -500.0]))

        if params.numel() != 2:
            raise ValueError("Eggholder requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        term1 = -(y + 47) * sin(sqrt(torch.abs(x / 2 + (y + 47))))
        term2 = -x * sin(sqrt(torch.abs(x - (y + 47))))
        return term1 + term2




class HolderTable(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([0.0, 0.0]))

        if params.numel() != 2:
            raise ValueError("Holder Table requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return -torch.abs(sin(x) * cos(y) *
                          torch.exp(torch.abs(1 - sqrt(x ** 2 + y ** 2) / pi)))




class Mccormick(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([3.5, 3.0]))

        if params.numel() != 2:
            raise ValueError("Mccormick requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        return sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1




class SchafferN2(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-30.0, -30.0]))

        if params.numel() != 2:
            raise ValueError("Schaffer N.2 requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        numerator = sin(x ** 2 - y ** 2) ** 2 - 0.5
        denominator = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
        return 0.5 + numerator / denominator




class SchafferN4(TestFunction):
    def __init__(self, params: Tensor = None):

        if params is None:
            params = Parameter(Tensor([-30.0, -30.0]))

        if params.numel() != 2:
            raise ValueError("Schaffer N.4 requires a parameter tensor of length 2!")

        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        x, y = self.params[0], self.params[1]
        numerator = cos(sin(torch.abs(x ** 2 - y ** 2))) ** 2 - 0.5
        denominator = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
        return 0.5 + numerator / denominator




class StyblinskiTang(TestFunction):
    def __init__(self, params: Tensor = None, num_params: int = 2):

        if params is None and num_params < 2:
            ValueError('Custom defined num_params must be > 1!')

        if params is not None and params.numel() < 2:
            ValueError('Custom defined params.numel() must > 1!')

        if params is None:
            params = Parameter(empty((num_params,)).fill_(0.0))
        self.params = Parameter(params)

        super().__init__()


    def loss(self) -> Tensor:
        # Standard definition multiplies the sum by 0.5
        sum_term = t_sum(self.params ** 4 - 16 * self.params ** 2 + 5 * self.params)
        return sum_term / 2.0




class Shekel(TestFunction):
    def __init__(self, params: Tensor, m: int, b: Tensor, c: list):

        if b.numel() != m:
            raise ValueError('Number of elements in b does not match m!')
        if len(c) != m:
            raise ValueError('The number of rows in c does not match m!')

        self.params = Parameter(params)
        self.m = m
        self.b = b
        self.c = c

        n = params.numel()
        for i in range(m):
            if len(self.c[i]) != n:
                raise ValueError(f'Row {i} of c does not have the required length {n}!')

        super().__init__()


    def loss(self) -> Tensor:
        n = self.params.numel()
        total = 0.0
        for i in range(self.m):
            sum_squares = sum([(self.params[j] - self.c[i][j]) ** 2 for j in range(n)])
            total += 1.0 / (self.b[i] + sum_squares)
        return torch.tensor([total])
