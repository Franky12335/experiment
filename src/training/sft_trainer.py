"""Project SFT trainer behavior needed by the preparation src."""

from __future__ import annotations

from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.device import get_device_name


class SeededSFTTrainer(SFTTrainer):
    """Pass the configured seed explicitly because the upstream sampler otherwise uses zero."""

    def _build_dataloader(self) -> None:
        config = self.config
        device_name = get_device_name()
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()
        seed = int(config.trainer.seed)

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=dp_size,
            rank=dp_rank,
            drop_last=True,
            seed=seed,
        )
        self.global_batch_size = config.data.train_batch_size
        if self.global_batch_size % dp_size:
            raise ValueError(
                "SFT train_batch_size must be divisible by the data-parallel size"
            )
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            if len(self.val_dataset) % self.global_batch_size:
                raise ValueError(
                    "SFT validation rows must be a complete global batch; rerun "
                    "src.data_prep.prepare_experiment_data before the training run"
                )
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=True,
                seed=seed,
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None
