# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of fm_solvers.py - Flow Matching DPM Solver
import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import mlx.core as mx


@dataclass
class SchedulerOutput:
    prev_sample: mx.array


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))
    return sigma


class FlowDPMSolverMultistepScheduler:
    """MLX port of FlowDPMSolverMultistepScheduler.

    Implements DPM-Solver++ for flow matching diffusion models.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        invert_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.euler_at_final = euler_at_final
        self.final_sigmas_type = final_sigmas_type
        self.lambda_min_clipped = lambda_min_clipped

        # Compute sigmas
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = mx.array(sigmas.astype(np.float32))
        self.timesteps = self.sigmas * num_train_timesteps

        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps=None,
        sigmas=None,
        mu=None,
        shift=None,
        **kwargs,
    ):
        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.use_dynamic_shifting and mu is not None:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            sigma_last = sigmas[-1]

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(timesteps.astype(np.int64))
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(self, model_output, sample=None):
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
            return x0_pred
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
            return epsilon

    def dpm_solver_first_order_update(self, model_output, sample=None, noise=None):
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s = mx.log(alpha_s) - mx.log(sigma_s)

        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (mx.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (mx.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            x_t = ((sigma_t / sigma_s * mx.exp(-h)) * sample +
                   (alpha_t * (1 - mx.exp(-2.0 * h))) * model_output +
                   sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise)
        elif self.algorithm_type == "sde-dpmsolver":
            x_t = ((alpha_t / alpha_s) * sample - 2.0 *
                   (sigma_t * (mx.exp(h) - 1.0)) * model_output +
                   sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise)
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_output_list, sample=None, noise=None):
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s0 = self.sigmas[self.step_index]
        sigma_s1 = self.sigmas[self.step_index - 1]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        if self.algorithm_type == "dpmsolver++":
            if self.solver_type == "midpoint":
                x_t = ((sigma_t / sigma_s0) * sample -
                       (alpha_t * (mx.exp(-h) - 1.0)) * D0 - 0.5 *
                       (alpha_t * (mx.exp(-h) - 1.0)) * D1)
            elif self.solver_type == "heun":
                x_t = ((sigma_t / sigma_s0) * sample -
                       (alpha_t * (mx.exp(-h) - 1.0)) * D0 +
                       (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1)
        elif self.algorithm_type == "dpmsolver":
            if self.solver_type == "midpoint":
                x_t = ((alpha_t / alpha_s0) * sample -
                       (sigma_t * (mx.exp(h) - 1.0)) * D0 - 0.5 *
                       (sigma_t * (mx.exp(h) - 1.0)) * D1)
            elif self.solver_type == "heun":
                x_t = ((alpha_t / alpha_s0) * sample -
                       (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                       (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1)
        return x_t

    def multistep_dpm_solver_third_order_update(self, model_output_list, sample=None):
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s0 = self.sigmas[self.step_index]
        sigma_s1 = self.sigmas[self.step_index - 1]
        sigma_s2 = self.sigmas[self.step_index - 2]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)
        lambda_s2 = mx.log(alpha_s2) - mx.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        h_1 = lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0 = (1.0 / r0) * (m0 - m1)
        D1_1 = (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        if self.algorithm_type == "dpmsolver++":
            x_t = ((sigma_t / sigma_s0) * sample -
                   (alpha_t * (mx.exp(-h) - 1.0)) * D0 +
                   (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1 -
                   (alpha_t * ((mx.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * D2)
        elif self.algorithm_type == "dpmsolver":
            x_t = ((alpha_t / alpha_s0) * sample -
                   (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                   (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1 -
                   (sigma_t * ((mx.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * D2)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = mx.argwhere(schedule_timesteps == timestep)
        if len(indices) > 1:
            return int(indices[1])
        return int(indices[0])

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output, timestep, sample, return_dict=True):
        if self.num_inference_steps is None:
            raise ValueError("Need to run set_timesteps first")

        if self.step_index is None:
            self._init_step_index(timestep)

        lower_order_final = (
            self.step_index == len(self.timesteps) - 1
        ) and (
            self.euler_at_final or
            (self.lower_order_final and len(self.timesteps) < 15) or
            self.final_sigmas_type == "zero"
        )
        lower_order_second = (
            self.step_index == len(self.timesteps) - 2
        ) and self.lower_order_final and len(self.timesteps) < 15

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        sample = sample.astype(mx.float32)
        noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        prev_sample = prev_sample.astype(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(self, original_samples, noise, timesteps):
        sigmas = self.sigmas
        step_indices = []
        for t in timesteps:
            step_indices.append(self.index_for_timestep(t))

        sigma = sigmas[mx.array(step_indices)]
        while len(sigma.shape) < len(original_samples.shape):
            sigma = mx.expand_dims(sigma, axis=-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
