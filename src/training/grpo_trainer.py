"""Synchronous single-pool Teacher GRPO trainer."""

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class TeacherGRPOTrainer(PPOTrainerSync):
    """Use all assigned GPUs for rollout and the on-policy actor update."""
