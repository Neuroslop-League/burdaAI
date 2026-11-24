"""FP32 CPU AdamW with synchronous gradient and parameter transfers."""
import torch


class HostAdamW:
    def __init__(self, parameters, **options):
        self.device_params = list(parameters)
        if any(p.dtype != torch.float32 for p in self.device_params):
            raise ValueError("HostAdamW requires FP32 model parameters")
        self.host_params = [torch.nn.Parameter(p.detach().to("cpu", copy=True))
                            for p in self.device_params]
        self.optimizer = torch.optim.AdamW(self.host_params, foreach=False, fused=False, **options)

    def zero_grad(self, set_to_none=True):
        for p in self.device_params:
            p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def stage_gradients(self):
        for device, host in zip(self.device_params, self.host_params):
            host.grad = None if device.grad is None else device.grad.detach().to("cpu", copy=True)
            device.grad = None

    @torch.no_grad()
    def step(self):
        self.stage_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for device, host in zip(self.device_params, self.host_params):
            device.copy_(host)

    @property
    def state(self):
        return self.optimizer.state

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def load_state_dict(self, state):
        # Model weights have just been restored by the caller.
        for device, host in zip(self.device_params, self.host_params):
            host.copy_(device)
        self.optimizer.load_state_dict(state)
