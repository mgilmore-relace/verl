# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Memory profiling utilities for checkpoint loading
#
# Usage: Set VERL_PROFILE_CHECKPOINT_MEMORY=1 before running training
# to enable memory profiling during checkpoint loads.

import functools
import gc
import os

import torch


def get_memory_stats() -> dict:
    """Get current memory statistics for both CPU and GPU."""
    import psutil

    stats = {
        "cpu_rss_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
    }

    if torch.cuda.is_available():
        stats.update({
            "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cuda_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        })

    return stats


def format_memory_stats(stats: dict) -> str:
    """Format memory stats as a string."""
    msg = f"CPU RSS: {stats['cpu_rss_mb']:.1f} MB"
    if "cuda_allocated_mb" in stats:
        msg += (
            f", CUDA alloc: {stats['cuda_allocated_mb']:.1f} MB, "
            f"reserved: {stats['cuda_reserved_mb']:.1f} MB"
        )
    return msg


def format_memory_delta(label: str, before: dict, after: dict) -> str:
    """Format the difference in memory between two snapshots."""
    cpu_delta = after["cpu_rss_mb"] - before["cpu_rss_mb"]
    msg = f"[MEM] {label}: CPU delta={cpu_delta:+.1f} MB"

    if "cuda_allocated_mb" in after:
        cuda_alloc_delta = after["cuda_allocated_mb"] - before["cuda_allocated_mb"]
        cuda_reserved_delta = after["cuda_reserved_mb"] - before["cuda_reserved_mb"]
        msg += f", CUDA alloc delta={cuda_alloc_delta:+.1f} MB"
        msg += f", reserved delta={cuda_reserved_delta:+.1f} MB"

    return msg


class MemoryTracker:
    """Track memory across multiple checkpoints."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.enabled = os.getenv("VERL_PROFILE_CHECKPOINT_MEMORY", "0") == "1"

    def _log(self, msg: str):
        if self.rank == 0:
            print(f"[rank {self.rank}] {msg}", flush=True)

    def snapshot(self, label: str) -> dict:
        if not self.enabled:
            return {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats = get_memory_stats()
        self._log(f"[MEM] {label}: {format_memory_stats(stats)}")
        return stats

    def delta(self, label: str, before: dict) -> dict:
        if not self.enabled:
            return {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        after = get_memory_stats()
        self._log(format_memory_delta(label, before, after))
        return after


def patch_load_checkpoint():
    """
    Patch MegatronCheckpointManager.load_checkpoint with memory profiling.

    Call this once at startup, or set VERL_PROFILE_CHECKPOINT_MEMORY=1.
    """
    from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing
    from verl.utils.megatron_utils import get_dist_checkpoint_path

    original_load = MegatronCheckpointManager.load_checkpoint

    @functools.wraps(original_load)
    def profiled_load_checkpoint(
        self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False
    ):
        tracker = MemoryTracker(self.rank)

        if not tracker.enabled:
            return original_load(self, local_path, hdfs_path, del_local_after_load)

        tracker._log("=" * 60)
        tracker._log("CHECKPOINT LOAD MEMORY PROFILE START")
        tracker._log("=" * 60)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        baseline = tracker.snapshot("baseline")

        # Phase 1: Generate state dict
        before = tracker.snapshot("before generate_state_dict")
        sharded_state_dict = self.generate_state_dict(
            self.should_load_model and self.use_dist_checkpointing,
            self.should_load_optimizer,
            self.should_load_extra,
            is_loading=True,
        )
        tracker.delta("generate_state_dict", before)

        # Phase 2: Load from disk
        before = tracker.snapshot("before load_dist_checkpointing")
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )
        tracker.delta("load_dist_checkpointing", before)

        # Phase 3: Load model
        if self.should_load_model and self.use_dist_checkpointing:
            before = tracker.snapshot("before load model state_dict")
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) == 1:
                    model_state_dict = state_dict.pop("model")
                else:
                    model_state_dict = state_dict.pop(f"model{vpp_rank}")
                self.model[vpp_rank].load_state_dict(model_state_dict)
                del model_state_dict
            tracker.delta("load model state_dict", before)

        # Phase 4: Load optimizer
        if self.should_load_optimizer:
            before = tracker.snapshot("before load optimizer state_dict")
            optimizer_state_dict = state_dict.pop("optimizer")
            self.optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict
            tracker.delta("load optimizer state_dict", before)

            if self.use_checkpoint_opt_param_scheduler and self.lr_scheduler is not None:
                lr_scheduler_state_dict = state_dict.pop("lr_scheduler")
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                del lr_scheduler_state_dict

        # Phase 5: Load RNG states
        if self.should_load_extra:
            before = tracker.snapshot("before load rng_state")
            rng_state = state_dict.pop("rng_state")
            self.load_rng_states(rng_state)
            del rng_state
            tracker.delta("load rng_state", before)

        # Phase 6: Cleanup
        before = tracker.snapshot("before cleanup")
        state_dict.clear()
        del state_dict
        del sharded_state_dict
        gc.collect()
        tracker.delta("gc.collect()", before)

        # Phase 7: empty_cache effect
        if torch.cuda.is_available():
            before = tracker.snapshot("before empty_cache")
            torch.cuda.empty_cache()
            after = tracker.delta("empty_cache", before)
            freed = before["cuda_reserved_mb"] - after["cuda_reserved_mb"]
            tracker._log(f"[MEM] empty_cache freed {freed:.1f} MB of cached memory")

        # Final summary
        final = tracker.snapshot("final")
        tracker._log("-" * 60)
        tracker._log("SUMMARY:")
        cpu_delta = final["cpu_rss_mb"] - baseline["cpu_rss_mb"]
        tracker._log(f"  CPU RSS: {baseline['cpu_rss_mb']:.1f} -> {final['cpu_rss_mb']:.1f} MB (delta: {cpu_delta:+.1f} MB)")
        if "cuda_allocated_mb" in final:
            cuda_delta = final["cuda_allocated_mb"] - baseline["cuda_allocated_mb"]
            tracker._log(
                f"  CUDA alloc: {baseline['cuda_allocated_mb']:.1f} -> "
                f"{final['cuda_allocated_mb']:.1f} MB (delta: {cuda_delta:+.1f} MB)"
            )
            tracker._log(f"  CUDA peak: {final['cuda_max_allocated_mb']:.1f} MB")
        tracker._log("=" * 60)

        if del_local_after_load:
            from verl.utils.fs import is_non_local
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception:
                pass

    MegatronCheckpointManager.load_checkpoint = profiled_load_checkpoint
    print("[MEMORY_PROFILER] Patched MegatronCheckpointManager.load_checkpoint")


# Note: To enable profiling, import this module AFTER MegatronCheckpointManager
# is defined, then call patch_load_checkpoint(), or set the env var and import
# this module from megatron_checkpoint_manager.py at the end of the file.
