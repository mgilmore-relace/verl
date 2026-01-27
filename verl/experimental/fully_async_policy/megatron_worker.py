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
from numpy import mod
from omegaconf import DictConfig
from pandas.core.algorithms import mode

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
from verl.utils.tensordict_utils import get
from verl.utils.vllm.vllm_fp8_utils import is_fp8_model, load_quanted_weights
from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

from .checkpoint_engine import CheckpointEngine

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def get_inference_model_and_runner(rollout):
    """
    get models according to different types of inference_engine
    Args:
        rollout: rollout object
        debug: whether to print debug information
    Returns:
        model: model object
        runner: runner object
    """
    inference_engine = rollout.inference_engine

    if hasattr(inference_engine, "llm_engine"):
        model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        inference_model = model_runner.model
    elif hasattr(inference_engine, "worker"):
        model_runner = inference_engine.worker.model_runner
        inference_model = model_runner.model
    else:
        raise AttributeError(
            f"Unsupported inference_engine type: {type(inference_engine)}. "
            f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
        )

    return inference_model, model_runner

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

    @staticmethod
    def _get_layer_id(key: str) -> str:
        """Extract layer identifier from weight key for grouping."""
        parts = key.split(".")
        # model.layers.X.* -> model.layers.X
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts):
                return ".".join(parts[: idx + 2])
        # embed_tokens, lm_head, norm, etc.
        for special in ["embed_tokens", "lm_head", "norm", "final_layernorm"]:
            if special in key:
                return special
        return parts[0] if parts else key

    def parameter_generator(self, sync_group_name: str = "actor_rollout"):
        """Generate weight chunks grouped by layer for atomic loading.

        Yields lists of (key, tensor) pairs where weights belonging to the
        same transformer layer are grouped together.
        """
        from ray.util.collective import collective

        params_generator = self._get_actor_params_generator() if self._is_actor else None

        current_layer = None
        chunk = []

        for key, shape, dtype in self._weights_info:
            if self._is_actor:
                weight_key, weight = next(params_generator)
                assert key == weight_key and shape == weight.size() and dtype == weight.dtype

            layer_id = self._get_layer_id(key)

            # Yield previous chunk when layer changes
            if layer_id != current_layer and chunk:
                yield chunk
                chunk = []
            current_layer = layer_id

            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor and torch.distributed.get_rank() == 0:
                tensor.copy_(weight)

            get_torch_device().synchronize()
            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)
            get_torch_device().synchronize()

            chunk.append((key, tensor))

        if chunk:
            yield chunk

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

        inference_model = None
        model_runner = None

        if self._is_rollout:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            inference_model, model_runner = get_inference_model_and_runner(self.rollout)

            patch_vllm_moe_model_weight_loader(inference_model)

        # Generator yields chunks of (key, tensor) pairs grouped by layer
        # Both actors and rollouts must iterate to participate in collective broadcast
        weight_chunks = self.parameter_generator(sync_group_name)


        if self._is_rollout and use_fp8:
            use_fp8 = is_fp8_model(model_runner.vllm_config)
            total_loaded = 0
            logger.info("Start loading FP8 weights (async)...")

        for chunk in weight_chunks:
            if self._is_rollout:
                if use_fp8:
                    loaded_params = load_quanted_weights(chunk, model_runner)
                    total_loaded += len(loaded_params)
                else:
                    inference_model.load_weights(chunk)
                    total_loaded += len(chunk)

        if self._is_rollout:
            if use_fp8:
                logger.info(f"FP8 weights loaded (async), loaded_params: {total_loaded}")
            else:
                logger.info(f"Loaded {total_loaded} weights")

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
            inference_model = get_inference_model_and_runner(self.rollout)
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
