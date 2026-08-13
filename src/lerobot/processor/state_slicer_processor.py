#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from src.lerobot.configs.types import PipelineFeatureType, PolicyFeature
from src.lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from src.lerobot.utils.constants import OBS_STATE


@dataclass
@ProcessorStepRegistry.register(name="state_slicer_processor")
class StateSlicerProcessorStep(ObservationProcessorStep):
    """Truncates ``observation.state`` to the first ``n_dims`` dimensions.

    Used when a dataset's state dimension exceeds a model's ``max_state_dim``
    (e.g. 48-dim dataset vs. 32-dim pi0).  Only the leading ``n_dims``
    values are kept — typically the robot's joint positions.
    """

    n_dims: int = 20

    def observation(self, observation):
        if OBS_STATE in observation:
            state = observation[OBS_STATE]
            observation[OBS_STATE] = state[..., : self.n_dims]
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            feat = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            features = deepcopy(features)
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=feat.type,
                shape=(self.n_dims,),
            )
        return features

    def get_config(self) -> dict[str, Any]:
        return {"n_dims": self.n_dims}


def slice_stats_for_state(
    stats: dict[str, dict[str, Any]], n_dims: int
) -> dict[str, dict[str, Any]]:
    """Truncate per-feature statistics for ``observation.state`` to ``n_dims``.

    Each entry under ``stats[OBS_STATE]`` (``mean``, ``std``, ``min``, ``max``,
    ``q01``, ``q99``, …) is assumed to be a tensor or array whose last
    dimension matches the original state dimension.
    """
    if OBS_STATE not in stats:
        return stats
    result = deepcopy(stats)
    for stat_name, tensor in result[OBS_STATE].items():
        t = torch.as_tensor(tensor)
        result[OBS_STATE][stat_name] = t[..., :n_dims]
    return result
