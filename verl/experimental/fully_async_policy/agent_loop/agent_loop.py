# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    AsyncLLMServerManager,
    DictConfigWrap,
    TrajectorySegment,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils.rollout_trace import (
    rollout_trace_attr,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)
        # A shared cancellation event for all agent loops running on this worker.
        self.cancellation_event = asyncio.Event()

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)
        try:
            tasks = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                kwargs["output"] = partial_output_list[i]
                tasks.append(
                    asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
                )
            outputs = await asyncio.gather(*tasks)
        except Exception:
            logger.exception("_partial_run_agent_loop failed")
            raise

        is_cancel = any(output.extra_fields.get("is_cancel", False) for output in outputs)
        if not is_cancel:
            output = self._postprocess(outputs)
            output = self._addition_process(output)
            return output, is_cancel
        return outputs, is_cancel

    def _addition_process(self, output: DataProto):
        """collect metirics"""
        metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
        output.non_tensor_batch["processing_times"] = processing_times_list
        output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
        return output

    def _extract_segment_from_agent_data(self, agent_data: Any, mask_before: int = 0) -> TrajectorySegment:
        """Extract a TrajectorySegment from agent_data state.

        Segments store cumulative data - all response_ids, logprobs, and routed_experts
        generated up to this point. The response_mask is modified to mask out tokens
        covered by previous segments.

        Args:
            agent_data: The agent data containing response state
            mask_before: Index before which to zero out response_mask (tokens covered by previous segments)

        Returns:
            TrajectorySegment with cumulative data, response_mask zeroed for [0:mask_before]
        """
        response_mask = list(agent_data.response_mask)

        # Mask out tokens covered by previous segments
        for i in range(min(mask_before, len(response_mask))):
            response_mask[i] = 0

        # Extract full response from prompt_ids (response_ids is only the last chunk)
        full_response = agent_data.prompt_ids[-len(agent_data.response_mask):] if agent_data.response_mask else []

        return TrajectorySegment(
            prompt_ids=agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)],
            response_ids=list(full_response),
            response_mask=response_mask,
            response_logprobs=list(agent_data.response_logprobs) if agent_data.response_logprobs else None,
            routed_experts=agent_data.routed_experts,
        )

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        # Completed, return directly
        if kwargs["output"] is not None and not kwargs["output"].extra_fields.get("is_cancel", False):
            logger.info("In _partial_run_agent_loop, already completed, return derictly!")
            return kwargs["output"]

        # Check for param_version change and handle segment creation
        prev_output = kwargs.get("output")
        accumulated_segments = []
        prev_segment_end = 0  # Track end of previous segment for masking
        current_param_version = kwargs.get("param_version", 0)

        if prev_output is not None and prev_output.extra_fields.get("is_cancel", False):
            # Get accumulated segments from previous runs
            accumulated_segments = prev_output.extra_fields.get("accumulated_segments", [])
            prev_segment_end = prev_output.extra_fields.get("prev_segment_end", 0)
            agent_data = prev_output.extra_fields.get("agent_data")

            if agent_data is not None:
                prev_param_version = agent_data.extra_fields.get("param_version_end", 0)

                # If param_version changed, create segment and reset routed_experts
                if current_param_version != prev_param_version and agent_data.routed_experts is not None:
                    # Create cumulative segment, masking tokens covered by previous segments
                    segment = self._extract_segment_from_agent_data(agent_data, mask_before=prev_segment_end)
                    if len(segment.response_mask) > 0:
                        accumulated_segments = accumulated_segments + [segment]
                        # Update prev_segment_end for next segment
                        prev_segment_end = len(agent_data.response_mask)

                    # Reset routed_experts so the agent loop uses fresh expert selection
                    agent_data.routed_experts = None

                    # Update param_version_end so subsequent resumes with same version don't create spurious segments
                    agent_data.extra_fields["param_version_end"] = current_param_version

                    logger.info(
                        f"[PartialToolAgent] Param version changed {prev_param_version} -> {current_param_version}, "
                        f"created segment and reset routed_experts"
                    )

        try:
            with rollout_trace_attr(
                step=trajectory["step"],
                sample_index=trajectory["sample_index"],
                rollout_n=trajectory["rollout_n"],
                validate=trajectory["validate"],
                name="agent_loop",
            ):
                assert agent_name in _agent_loop_registry, (
                    f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
                )

                agent_loop_config = _agent_loop_registry[agent_name]
                agent_loop = hydra.utils.instantiate(
                    config=agent_loop_config,
                    trainer_config=DictConfigWrap(config=self.config),
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    dataset_cls=self.dataset_cls,
                    dataset_config=self.config.data,
                )
                output: AgentLoopOutput = await agent_loop.run(
                    sampling_params, cancellation_event=self.cancellation_event, **kwargs
                )

                if output.extra_fields.get("is_cancel", False):
                    # Store accumulated segments and prev_segment_end in cancelled output for next resume
                    output.extra_fields["accumulated_segments"] = accumulated_segments
                    output.extra_fields["prev_segment_end"] = prev_segment_end
                else:
                    # Completed - build final cumulative segment from the output fields
                    if output.routed_experts is not None and len(output.response_mask) > 0:
                        # Apply masking for tokens covered by previous segments
                        final_response_mask = list(output.response_mask)
                        for i in range(min(prev_segment_end, len(final_response_mask))):
                            final_response_mask[i] = 0

                        final_segment = TrajectorySegment(
                            prompt_ids=output.prompt_ids,
                            response_ids=list(output.response_ids),
                            response_mask=final_response_mask,
                            response_logprobs=list(output.response_logprobs)
                            if output.response_logprobs else None,
                            routed_experts=output.routed_experts,
                        )
                        accumulated_segments = accumulated_segments + [final_segment]

                    output.segments = accumulated_segments if accumulated_segments else None

                    kwargs.pop("output", None)
                    output = await self._agent_loop_postprocess(output, **kwargs)

                return output
        except Exception:
            logger.exception("Agent_loop run failed")
            raise

    async def pause_agent_loops(self):
        """Set the shared cancellation event and abort all active requests."""
        self.cancellation_event.set()
        return await self.server_manager.sleep()

    async def resume_agent_loops(self):
        """Clear the shared cancellation event and allow new requests."""
        self.cancellation_event.clear()
        await self.server_manager.wake_up()


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        self.agent_loop_train_workers_class = FullyAsyncAgentLoopWorker
        self.agent_loop_valid_workers_class = ray.remote(AgentLoopWorker)
        self.rollout_replica_class = vLLMReplica

        self.rm_resource_pool = rm_resource_pool
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_train_workers = None
        self.agent_loop_valid_workers = None

    @classmethod
    async def create(
        cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ) -> "FullyAsyncAgentLoopManager":
        instance = cls(config, worker_group, rm_resource_pool)
        await instance._async_init()
        return instance

    async def _async_init(self):
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    def _init_agent_loop_workers(self):
        self.agent_loop_train_workers = []
        self.agent_loop_valid_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_train_workers.append(
                self.agent_loop_train_workers_class.options(
                    name=f"agent_loop_train_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )
            self.agent_loop_valid_workers.append(
                self.agent_loop_valid_workers_class.options(
                    name=f"agent_loop_valid_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_valid_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_valid_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def _initialize_llm_servers_async(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            await asyncio.to_thread(
                update_prometheus_config, rollout_config.prometheus, self.server_addresses, rollout_config.name
            )

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """
        Asynchronously process a single sample

        Args:
            sample: Single sample data
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: Processing results
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_train_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_train_workers)
        return worker

    async def pause(self):
        """Cancel all active requests from this manager's workers."""
        worker_pause_tasks = [worker.pause_agent_loops.remote() for worker in self.agent_loop_train_workers]
        await asyncio.gather(*worker_pause_tasks)

    async def resume(self):
        """Resume this manager's workers and allow new requests."""
        worker_resume_tasks = [worker.resume_agent_loops.remote() for worker in self.agent_loop_train_workers]
        await asyncio.gather(*worker_resume_tasks)

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
