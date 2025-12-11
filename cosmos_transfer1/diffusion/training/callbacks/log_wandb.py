# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
from torch import Tensor
import wandb

from cosmos_transfer1.utils.callback import Callback
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.distributed import rank0_only
from cosmos_transfer1.utils.model import Model
from cosmos_transfer1.utils.trainer import Trainer

class WandbCallback(Callback):
    def __init__(self):
        # self.config = config
        self.start_iteration_time = time.time()


    @rank0_only
    def on_train_start(self, model, iteration=0):
        wandb.init(
            project=self.config.job.project,
            name=self.config.job.name,
            config=self.config, 
            resume="auto"
        )

    @rank0_only
    def on_training_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        self.start_iteration_time = time.time()

    @rank0_only
    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        iteration_time = time.time() - self.start_iteration_time
        logs = {
            "train/loss": loss.item(),
            "iteration_time": iteration_time,
        }
        # 核心修正：从 trainer 中获取学习率（优先从scheduler，其次从optimizer）
        try:
            # 方式1：从 trainer 的 scheduler 获取（推荐，适配代码库的调度器逻辑）
            if hasattr(self.trainer, "scheduler") and self.trainer.scheduler is not None:
                lr = self.trainer.scheduler.get_last_lr()[0]
            # 方式2：如果没有scheduler，直接从 optimizer 获取
            elif hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
                lr = self.trainer.optimizer.param_groups[0]["lr"]
            else:
                lr = None  # 无学习率时跳过
        except Exception as e:
            lr = None
            print(f"Warning: 获取学习率失败 - {e}")

        if lr is not None:
            logs["train/lr"] = lr
        wandb.log(logs, step=iteration)

    @rank0_only
    def on_train_end(self, model, iteration=0):
        wandb.finish()