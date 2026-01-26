# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 NVIDIA Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

import torch
import torch.distributed
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.megatron_utils import (
    copy_megatron_model_to_cpu,
    restore_megatron_model_from_cpu,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

from .checkpoint_engine import CheckpointEngine

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def get_inference_model(rollout, debug=True):
    """
    get models according to different types of inference_engine
    Args:
        rollout: rollout object
        debug: whether to print debug information
    Returns:
        model: model object
    """
    inference_engine = rollout.inference_engine

    if debug:
        print(f"[DEBUG get_inference_model] inference_engine type: {type(inference_engine)}")
        print(f"[DEBUG get_inference_model] has llm_engine: {hasattr(inference_engine, 'llm_engine')}")
        print(f"[DEBUG get_inference_model] has worker: {hasattr(inference_engine, 'worker')}")

    if hasattr(inference_engine, "llm_engine"):
        if debug:
            print("[DEBUG get_inference_model] Using llm_engine path")
        inference_model = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    elif hasattr(inference_engine, "worker"):
        if debug:
            worker = inference_engine.worker
            print(f"[DEBUG get_inference_model] Using worker path")
            print(f"[DEBUG get_inference_model] worker type: {type(worker)}")
            print(f"[DEBUG get_inference_model] has model_runner: {hasattr(worker, 'model_runner')}")
            if hasattr(worker, "model_runner"):
                model_runner = worker.model_runner
                print(f"[DEBUG get_inference_model] model_runner type: {type(model_runner)}")
                print(f"[DEBUG get_inference_model] has model: {hasattr(model_runner, 'model')}")
        inference_model = inference_engine.worker.model_runner.model
    else:
        raise AttributeError(
            f"Unsupported inference_engine type: {type(inference_engine)}. "
            f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
        )

    if debug:
        print(f"[DEBUG get_inference_model] inference_model type: {type(inference_model)}")
        print(f"[DEBUG get_inference_model] model param count: {sum(p.numel() for p in inference_model.parameters())}")
        # Print a sample of parameter names
        param_names = [name for name, _ in inference_model.named_parameters()]
        print(f"[DEBUG get_inference_model] first 5 param names: {param_names[:5]}")
        print(f"[DEBUG get_inference_model] last 5 param names: {param_names[-5:]}")

    return inference_model


class DetachNcclSync(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_checkpoint_engine(self, rank_offset: int, actor_num: int, rollout_num: int):
        current_rank = torch.distributed.get_rank() + rank_offset
        actor_ranks = list(range(actor_num))
        rollout_ranks = [rank + actor_num for rank in range(rollout_num)]
        assert rank_offset == 0 or rank_offset == actor_num

        self.checkpoint_engine = CheckpointEngine(
            current_rank, actor_ranks, rollout_ranks, self.config.checkpoint_engine.device_buffer_size_M
        )

    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        local_rank = torch.distributed.get_rank()
        print(f"[DEBUG sync_rollout_weights] rank={local_rank}, is_actor={self._is_actor}, is_rollout={self._is_rollout}")
        print(f"[DEBUG sync_rollout_weights] weights_info count: {len(self._weights_info)}")

        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        params_generator = self._get_actor_params_generator() if self._is_actor else None

        inference_model = None
        model_param_names = set()
        sample_param_name = None
        sample_param_before = None

        if self._is_rollout:
            inference_model = get_inference_model(self.rollout, debug=True)
            model_param_names = set(name for name, _ in inference_model.named_parameters())
            print(f"[DEBUG sync_rollout_weights] model has {len(model_param_names)} parameters")

            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)

            # Sample a parameter before weight sync for comparison
            for name, param in inference_model.named_parameters():
                if "embed" in name.lower() or "layer" in name.lower():
                    sample_param_name = name
                    sample_param_before = (param.mean().item(), param.std().item(), param.abs().max().item())
                    print(f"[DEBUG sync_rollout_weights] BEFORE sync - {name}: mean={sample_param_before[0]:.6f}, std={sample_param_before[1]:.6f}, max={sample_param_before[2]:.6f}")
                    break

        # Track key matching statistics
        keys_matched = 0
        keys_not_found = []
        weights_loaded = 0

        from ray.util.collective import collective

        for idx, (key, shape, dtype) in enumerate(self._weights_info):
            if self._is_actor:
                weight_key, weight = next(params_generator)
                assert key == weight_key
                assert shape == weight.size()
                assert dtype == weight.dtype

            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor and torch.distributed.get_rank() == 0:
                tensor.copy_(weight)
                origin_tensor = tensor.clone()
                print(f"[DEBUG sync_rollout_weights] Actor weight {key}: shape={shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")

            # Synchronize GPU operations before the Ray collective broadcast
            get_torch_device().synchronize()

            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)

            if self._is_actor and torch.distributed.get_rank() == 0:
                # Verify that the broadcasted tensor matches the original
                if not torch.allclose(tensor, origin_tensor):
                    print(f"[DEBUG sync_rollout_weights] ERROR: Broadcasted tensor for key {key} does not match original!")

            if self._is_rollout:
                # Check if key exists in model
                if key in model_param_names:
                    keys_matched += 1
                else:
                    keys_not_found.append(key)
                    if len(keys_not_found) <= 5:
                        print(f"[DEBUG sync_rollout_weights] WARNING: Key '{key}' not found in model parameters!")

                # Check for NaN/Inf in weights
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"[DEBUG sync_rollout_weights] Loading weight {key}: shape={shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
                    print(f"[DEBUG sync_rollout_weights] ERROR: Weight {key} contains NaN or Inf!")

                inference_model.load_weights([(key, tensor)])
                weights_loaded += 1

        if self._is_rollout:
            print(f"[DEBUG sync_rollout_weights] Weight sync summary:")
            print(f"[DEBUG sync_rollout_weights]   Total weights in info: {len(self._weights_info)}")
            print(f"[DEBUG sync_rollout_weights]   Weights loaded: {weights_loaded}")
            print(f"[DEBUG sync_rollout_weights]   Keys matched in model: {keys_matched}")
            print(f"[DEBUG sync_rollout_weights]   Keys not found: {len(keys_not_found)}")
            if keys_not_found:
                print(f"[DEBUG sync_rollout_weights]   First 10 missing keys: {keys_not_found[:10]}")

            # Check sample parameter after sync
            if sample_param_name:
                for name, param in inference_model.named_parameters():
                    if name == sample_param_name:
                        after_stats = (param.mean().item(), param.std().item(), param.abs().max().item())
                        print(f"[DEBUG sync_rollout_weights] AFTER sync - {name}: mean={after_stats[0]:.6f}, std={after_stats[1]:.6f}, max={after_stats[2]:.6f}")
                        if sample_param_before:
                            changed = abs(after_stats[0] - sample_param_before[0]) > 1e-6 or abs(after_stats[1] - sample_param_before[1]) > 1e-6
                            print(f"[DEBUG sync_rollout_weights] Parameter changed after sync: {changed}")
                        break

        if self._is_actor and self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}
        self.cpu_saved_models[n] = copy_megatron_model_to_cpu(self.actor.actor_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        if n in self.cpu_saved_models:
            restore_megatron_model_from_cpu(self.actor.actor_module, self.cpu_saved_models[n])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]

    def cache_actor_weights_to_cpu(self):
        self.cpu_named_params = {}
        if self._is_actor:
            params_generator = self._get_actor_params_generator()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            print(f"cache_actor_weights_to_cpu, local_rank:{local_rank}, world_size:{world_size}")
            for tensor_idx, (key, tensor) in enumerate(params_generator):
                if tensor_idx % world_size == local_rank:
                    self.cpu_named_params[key] = tensor.to("cpu", non_blocking=True)
            get_torch_device().synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        local_rank = torch.distributed.get_rank()
        print(f"[DEBUG sync_rollout_weights_by_checkpoint] rank={local_rank}, is_actor={self._is_actor}, is_rollout={self._is_rollout}")

        # Load model to GPU
        load_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        load_duration = time.time() - load_start_time

        from ray.util.collective import collective

        # Cache actor weights to CPU and measure the time taken
        cache_start_time = time.time()
        self.cache_actor_weights_to_cpu()
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time

        # Register the cached weights into the checkpoint engine
        self.checkpoint_engine.register_checkpoint(self._weights_info, self.cpu_named_params)
        register_end_time = time.time()
        register_duration = register_end_time - cache_end_time
        self.cpu_named_params = {}

        collective.barrier(group_name=sync_group_name)
        update_start_time = time.time()

        inference_model = None
        sample_param_name = None
        sample_param_before = None
        if self._is_rollout:
            inference_model = get_inference_model(self.rollout, debug=True)
            model_param_names = set(name for name, _ in inference_model.named_parameters())
            print(f"[DEBUG sync_rollout_weights_by_checkpoint] model has {len(model_param_names)} parameters")

            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)

            # Sample a parameter before weight sync for comparison
            for name, param in inference_model.named_parameters():
                if "embed" in name.lower() or "layer" in name.lower():
                    sample_param_name = name
                    sample_param_before = (param.mean().item(), param.std().item(), param.abs().max().item())
                    print(f"[DEBUG sync_rollout_weights_by_checkpoint] BEFORE sync - {name}: mean={sample_param_before[0]:.6f}, std={sample_param_before[1]:.6f}, max={sample_param_before[2]:.6f}")
                    break

        # Update the checkpoint with the inference model and broadcast weights
        self.checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=sync_group_name,
            overlap_broadcast_and_consume=self.config.checkpoint_engine.overlap_broadcast_and_consume,
        )

        update_end_time = time.time()
        update_duration = update_end_time - update_start_time

        # Check sample parameter after sync
        if self._is_rollout and sample_param_name and inference_model:
            for name, param in inference_model.named_parameters():
                if name == sample_param_name:
                    after_stats = (param.mean().item(), param.std().item(), param.abs().max().item())
                    print(f"[DEBUG sync_rollout_weights_by_checkpoint] AFTER sync - {name}: mean={after_stats[0]:.6f}, std={after_stats[1]:.6f}, max={after_stats[2]:.6f}")
                    if sample_param_before:
                        changed = abs(after_stats[0] - sample_param_before[0]) > 1e-6 or abs(after_stats[1] - sample_param_before[1]) > 1e-6
                        print(f"[DEBUG sync_rollout_weights_by_checkpoint] Parameter changed after sync: {changed}")
                    break

        offload_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        offload_duration = time.time() - offload_start_time

        print(
            f"sync_rollout_weights_by_checkpoint finish!, rank:{torch.distributed.get_rank()},"
            f" is_actor:{self._is_actor}, is_rollout:{self._is_rollout},"
            f" total cost:{update_end_time - cache_start_time} seconds, while cache cost {cache_duration} seconds, "
            f" register cost {register_duration} seconds, update cost {update_duration} seconds"
        )

        if self._is_actor and self._is_offload_param:
            print(
                f"sync_rollout_weights_by_checkpoint load model to gpu cost {load_duration} seconds,"
                f" offload model to cpu cost {offload_duration} seconds"
            )


class DetachActorWorker(DetachNcclSync):
    def _get_actor_params_generator(self):
        assert self._is_actor
        if self.bridge is not None:
            generator = self.bridge.export_weights(self.actor.actor_module)
        else:
            generator = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )

        return generator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        # Here, we only call this function at the beginning,
        # and immediately afterwards we call sync_rollout_weights.
        # So we no longer call offload in this.
        return ret


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        ActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
