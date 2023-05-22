#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher


class ContrastiveSeqBatchScheduler(SeqBatchScheduler):

    def init_forward(self, input_ids: torch.Tensor, batch_uids: torch.Tensor,
                     config: SearchConfig) -> SeqBatcher:
        pass

    def inference_call(self) -> torch.Tensor:
        pass
