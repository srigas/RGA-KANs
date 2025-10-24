from .rgakan import RGAKAN
from .kan import KAN
from .piratenet import PirateNet

from typing import Union


class BurgersKAN(KAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_layers: int, D: int = 5,
                 init_scheme: Union[dict, None] = "default",
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 seed: int = 42):
        
        self.model = KAN(n_in, n_out, n_hidden, num_layers, D, init_scheme,
                            period_axes, rff_std, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, -1) = u(t, 1) = 0: multiply by (1 - x^2)
        x_coord = original_x[:, 1:2]
        y = (1 - x_coord**2) * y

        return y
        

class BurgersModel(RGAKAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 D: int = 5, init_scheme: Union[dict, None] = None,
                 alpha: float = 0.0, beta: float = 1.0, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 sine_D: Union[None, int] = None, seed: int = 42):
        
        self.model = RGAKAN(n_in, n_out, n_hidden, num_blocks, D, init_scheme, alpha, beta, ref,
                            period_axes, rff_std, sine_D, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, -1) = u(t, 1) = 0: multiply by (1 - x^2)
        x_coord = original_x[:, 1:2]
        y = (1 - x_coord**2) * y

        return y


class BurgersPirate(PirateNet):
    
    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 alpha: float, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):

        self.model = PirateNet(n_in, n_out, n_hidden, num_blocks, alpha, ref, period_axes,
                               rff_std, RWF, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, -1) = u(t, 1) = 0: multiply by (1 - x^2)
        x_coord = original_x[:, 1:2]
        y = (1 - x_coord**2) * y

        return y


class SGKAN(KAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_layers: int, D: int = 5,
                 init_scheme: Union[dict, None] = "default",
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 seed: int = 42):
        
        self.model = KAN(n_in, n_out, n_hidden, num_layers, D, init_scheme,
                            period_axes, rff_std, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, 0) = u(t, 1) = 0: multiply by x(1 - x)
        x_coord = original_x[:, 1:2]
        y = x_coord * (1 - x_coord) * y

        return y
        

class SGModel(RGAKAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 D: int = 5, init_scheme: Union[dict, None] = None,
                 alpha: float = 0.0, beta: float = 1.0, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 sine_D: Union[None, int] = None, seed: int = 42):
        
        self.model = RGAKAN(n_in, n_out, n_hidden, num_blocks, D, init_scheme, alpha, beta, ref,
                            period_axes, rff_std, sine_D, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, 0) = u(t, 1) = 0: multiply by x(1 - x)
        x_coord = original_x[:, 1:2]
        y = x_coord * (1 - x_coord) * y

        return y


class SGPirate(PirateNet):
    
    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 alpha: float, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):

        self.model = PirateNet(n_in, n_out, n_hidden, num_blocks, alpha, ref, period_axes,
                               rff_std, RWF, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(t, 0) = u(t, 1) = 0: multiply by x(1 - x)
        x_coord = original_x[:, 1:2]
        y = x_coord * (1 - x_coord) * y

        return y


class PoissonKAN(KAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_layers: int, D: int = 5,
                 init_scheme: Union[dict, None] = "default",
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 seed: int = 42):
        
        self.model = KAN(n_in, n_out, n_hidden, num_layers, D, init_scheme,
                            period_axes, rff_std, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(-1, y) = u(1, y) = u(x, -1) = u(x, 1) = 0: multiply by (1 - x^2)(1 - y^2)
        x_coord = original_x[:, 0:1]
        y_coord = original_x[:, 1:2]
        y = (1 - x_coord**2)  * (1 - y_coord**2) * y

        return y
        

class PoissonModel(RGAKAN):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 D: int = 5, init_scheme: Union[dict, None] = None,
                 alpha: float = 0.0, beta: float = 1.0, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 sine_D: Union[None, int] = None, seed: int = 42):
        
        self.model = RGAKAN(n_in, n_out, n_hidden, num_blocks, D, init_scheme, alpha, beta, ref,
                            period_axes, rff_std, sine_D, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(-1, y) = u(1, y) = u(x, -1) = u(x, 1) = 0: multiply by (1 - x^2)(1 - y^2)
        x_coord = original_x[:, 0:1]
        y_coord = original_x[:, 1:2]
        y = (1 - x_coord**2)  * (1 - y_coord**2) * y

        return y


class PoissonPirate(PirateNet):
    
    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 alpha: float, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):

        self.model = PirateNet(n_in, n_out, n_hidden, num_blocks, alpha, ref, period_axes,
                               rff_std, RWF, seed)

    
    def __call__(self, x):
        
        original_x = x

        y = self.model(x)

        # Impose BC u(-1, y) = u(1, y) = u(x, -1) = u(x, 1) = 0: multiply by (1 - x^2)(1 - y^2)
        x_coord = original_x[:, 0:1]
        y_coord = original_x[:, 1:2]
        y = (1 - x_coord**2)  * (1 - y_coord**2) * y

        return y