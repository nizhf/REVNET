import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from .build import LR_SCHEDULERS, WEIGHT_SCHEDULERS

LR_SCHEDULERS.register_module(name="StepLR", module=optim.lr_scheduler.StepLR)
LR_SCHEDULERS.register_module(name="CosineAnnealingLR", module=optim.lr_scheduler.CosineAnnealingLR)


@LR_SCHEDULERS.register_module()
class ExponentialLinearWarmupLR(optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer: optim.Optimizer, warmup, warmup_init_decay, decay_step, lr_decay, lowest_decay, last_epoch=-1
    ):
        self.warmup = warmup
        self.warmup_init_decay = warmup_init_decay
        self.decay_step = decay_step
        self.lr_decay = lr_decay
        self.lowest_decay = lowest_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch >= self.warmup:
            lr_ratio = max(self.lr_decay ** ((epoch - self.warmup) / self.decay_step), self.lowest_decay)
        else:
            lr_ratio = self.warmup_init_decay + epoch / self.warmup * (1 - self.warmup_init_decay)
        return [lr_ratio * base_lr for base_lr in self.base_lrs]


@LR_SCHEDULERS.register_module()
class CosineAnnealingWarmupLR(optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer: optim.Optimizer, epochs, warmup, warmup_init_decay, niter_per_ep, lowest_decay, last_epoch=-1
    ):
        self.warmup_iters = warmup * niter_per_ep
        self.iters = epochs * niter_per_ep - self.warmup_iters
        self.warmup_init_decay = warmup_init_decay
        self.lowest_decay = lowest_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # treats last_epoch as last step
        if self.last_epoch < self.warmup_iters:
            lr_ratio = self.warmup_init_decay + self.last_epoch / self.warmup_iters * (1 - self.warmup_init_decay)
            return [lr_ratio * base_lr for base_lr in self.base_lrs]
        else:
            lr_ratio = self.lowest_decay + 0.5 * (1 - self.lowest_decay) * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / self.iters)
            )
            return [lr_ratio * base_lr for base_lr in self.base_lrs]


@LR_SCHEDULERS.register_module()
class CosineWarmupLR(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup, max_epoch, lr_min, lr_max, last_epoch=-1):
        self.warmup = warmup
        self.max_epoch = max_epoch
        self.lr_min = lr_min
        self.lr_max = lr_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup:
            lr_ratio = (epoch + 1) / self.warmup
        else:
            lr_ratio = (
                self.lr_min
                + 0.5
                * (self.lr_max - self.lr_min)
                * (1.0 + math.cos((epoch - self.warmup) / (self.max_epoch - self.warmup) * math.pi))
            ) / self.lr_max
        return [lr_ratio * base_lr for base_lr in self.base_lrs]


def dino_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


@LR_SCHEDULERS.register_module()
class TransformerLR(optim.lr_scheduler.LRScheduler):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"

    Args:
        optimizer (Optimizer): Optimizer.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        final_lr_scale (float): Final learning rate scale
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        decay_steps (int): Steps in decay stages
        last_epoch (int): Start epoch. Default to -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        peak_lr: float,
        final_lr: float,
        final_lr_scale: float,
        warmup_steps: int,
        decay_steps: int,
        last_epoch=-1,
    ):
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            lr = self.last_epoch * self.warmup_rate
        elif stage == 1:
            lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            lr = self.final_lr
        else:
            raise ValueError(f"Undefined stage {stage}.")
        return [lr for _ in self.optimizer.param_groups]

    def _decide_stage(self):
        if self.last_epoch < self.warmup_steps:
            return 0, self.last_epoch

        if self.warmup_steps <= self.last_epoch < self.warmup_steps + self.decay_steps:
            return 1, self.last_epoch - self.warmup_steps

        return 2, None


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler:
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


class WeightScheduler:
    def __init__(self, last_step=-1):
        self.last_step = last_step
        self._current_weight = None

    def get(self):
        if self._current_weight is None:
            raise ValueError("Weight scheduler called before initial step")
        return self._current_weight

    def step(self):
        raise NotImplementedError()


@WEIGHT_SCHEDULERS.register_module()
class ConstantWeight(WeightScheduler):
    r"""
    Change weight to a constant value at given steps
    """

    def __init__(self, values, stages=[0], last_step=-1):
        super().__init__(last_step)
        if not isinstance(values, list):
            self.values = [values]
        else:
            self.values = values
        self.stages = stages

    def step(self):
        self.last_step += 1
        current_stage = self.determine_stage()
        if current_stage == -1:
            self._current_weight = 0
        else:
            self._current_weight = self.values[current_stage]
        return self._current_weight

    def determine_stage(self):
        current_stage = -1
        for s in self.stages:
            if self.last_step >= s:
                current_stage += 1
            else:
                break
        return current_stage


@WEIGHT_SCHEDULERS.register_module()
class StepWeight(WeightScheduler):
    r"""
    Weighting scheduler, decay gamma after given step_size
    """

    def __init__(self, start_step, start_value, step_size, gamma, last_step=-1):
        super().__init__(last_step)
        self.start_step = start_step
        self.start_value = start_value
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.last_step += 1
        if self.last_step >= self.start_step:
            weight = self.start_value * self.gamma ** ((self.last_step - self.start_step) // self.step_size)
            self._current_weight = weight
        else:
            self._current_weight = 0
        return self._current_weight


@WEIGHT_SCHEDULERS.register_module()
class CyclicalAnnealingKLWeight(WeightScheduler):
    r"""
    Weighting scheduler for Variational Autoencoder with KL loss.
    Proposed in "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing" (NAACL 2019).
    """

    def __init__(
        self,
        start_step=0,
        min_weight=0.0,
        max_weight=1.0,
        period=2000,
        increase_ratio=0.5,
        mode="linear",
        last_step=-1,
    ):
        super().__init__(last_step)
        self.start_step = start_step
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.period = period
        self.step_size = 1 / (self.period * increase_ratio)
        self.mode = mode

    def step(self):
        self.last_step += 1
        if self.last_step >= self.start_step:
            self._current_weight = self.get_weight(self.last_step)
        else:
            self._current_weight = 0
        return self._current_weight

    def get_weight(self, step):
        step_in_cycle = (step - self.start_step) % self.period
        if self.mode == "linear":
            t = step_in_cycle * self.step_size
            ratio = min(self.min_weight + t * (self.max_weight - self.min_weight), self.max_weight)
        elif self.mode == "sigmoid":
            # scale step to [-5, 5]
            t = step_in_cycle * self.step_size * 10 - 5
            if t >= 5:
                ratio = self.max_weight
            else:
                ratio = min(
                    self.min_weight + (1 / (1 + np.exp(-t))) * (self.max_weight - self.min_weight), self.max_weight
                )
        elif self.mode == "cosine":
            # scale step to [0, pi]
            t = step_in_cycle * self.step_size * np.pi
            if t >= np.pi:
                ratio = self.max_weight
            else:
                ratio = min(
                    self.min_weight + (0.5 - 0.5 * np.cos(t)) * (self.max_weight - self.min_weight), self.max_weight
                )
        else:
            raise NotImplementedError(f"{self.mode} unknown.")
        return ratio


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # scheduler = CyclicalAnnealingKLWeightScheduler(150, 0.000001, 0.0001, 2000, 0.95, "linear", -1)
    # ratios = []
    # for step in range(10000):
    #     ratios.append(scheduler.step())
    # ratios = np.array(ratios)
    # plt.plot(ratios)
    # plt.show()
    from easydict import EasyDict

    configs = {
        "loss_weight_scheduler": [
            {"name": "ConstantWeightScheduler", "kwargs": {"values": 1}},
            {"name": "ConstantWeightScheduler", "kwargs": {"values": [1, 2, 3], "stages": [2, 4, 6]}},
            {
                "name": "StepWeightScheduler",
                "kwargs": {"start_step": 1, "start_value": 2, "step_size": 2, "gamma": 0.5},
            },
            {
                "name": "CyclicalAnnealingKLWeightScheduler",
                "kwargs": {"start_step": 0, "min_weight": 0.0, "max_weight": 1.0, "period": 4, "increase_ratio": 0.5},
            },
        ]
    }
    configs = EasyDict(configs)
    weight_schedulers = WeightScheduler(configs.loss_weight_scheduler)
    for i in range(20):
        print(weight_schedulers.step())
