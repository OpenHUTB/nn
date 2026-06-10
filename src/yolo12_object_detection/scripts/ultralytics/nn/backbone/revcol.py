import torch
import torch.nn as nn
import torch.distributed as dist
from ..modules.conv import Conv
from ..modules.block import C2f, C3, C3Ghost, C3k2
from ..extra_modules import *

__all__ = 'RevCol',

def get_gpu_states(fwd_gpu_devices):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states

def get_gpu_device(*args):

    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices

def set_device_states(fwd_cpu_state, devices, states) -> None:
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_and_grad(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)

class ReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0*alpha0
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3*alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1 ,c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse
            g2_up = g2_right+ c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
        
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()
        
        self.level = level
        self.first_col = first_col
        self.down = Conv(channels[level-1], channels[level], k=2, s=2, p=0, act=False) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = nn.Sequential(Conv(channels[level+1], channels[level]), nn.Upsample(scale_factor=2, mode='nearest')) if level in [0, 1, 2] else nn.Identity()            

    def forward(self, *args):

        c_down, c_up = args
        
        if self.first_col:
            x = self.down(c_down)
            return x
        
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel, first_col) -> None:
        super().__init__()
        self.fusion = Fusion(level, channels, first_col)
        modules = [eval(f'{kernel}')(channels[level], channels[level]) for i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)
    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x

class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel, first_col, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 

        self.level0 = Level(0, channels, layers, kernel, first_col)

        self.level1 = Level(1, channels, layers, kernel, first_col)

        self.level2 = Level(2, channels, layers, kernel, first_col)

        self.level3 = Level(3, channels, layers, kernel, first_col)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3= args

        c0 = (self.alpha0)*c0 + self.level0(x, c1)
        c1 = (self.alpha1)*c1 + self.level1(c0, c2)
        c2 = (self.alpha2)*c2 + self.level2(c1, c3)
        c3 = (self.alpha3)*c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class RevCol(nn.Module):
    def __init__(self, kernel='C2f', channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5, save_memory=True) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.channels = channels
        self.layers = layers

        self.stem = Conv(3, channels[0], k=4, s=4, p=0)

        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(channels, layers, kernel, first_col, save_memory=save_memory))
        
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        c0, c1, c2, c3 = 0, 0, 0, 0
        x = self.stem(x)        
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)       
        return [c0, c1, c2, c3]