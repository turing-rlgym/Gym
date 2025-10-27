# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import time
from fastapi import Request
from pydantic import Field

from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.integrations.atropos import (
    AtroposAgentVerifyRequest,
    AtroposNeMoGymResponse,
    AtroposSeedSessionRequest,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class AtroposAgentRunRequest(AtroposSeedSessionRequest):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=NeMoGymResponseCreateParamsNonStreaming
    )


class AtroposAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef


class AtroposAgent(SimpleResponsesAPIAgent):
    config: AtroposAgentConfig

    async def responses(
        self,
        body: AtroposAgentRunRequest,
    ) -> AtroposNeMoGymResponse:
        return await self.run(None, body)

    async def run(
        self,
        request: Request,
        body: AtroposAgentRunRequest,
    ) -> AtroposNeMoGymResponse:
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
        )
        seed_response.raise_for_status()
        seed_data = await seed_response.json()

        trajectory_data = seed_data.get("metadata", {}).get("trajectory_data", {})
        env_id = seed_data.get("env_id")

        dummy_response = AtroposNeMoGymResponse(
            id="atropos-batch",
            created_at=int(time.time()),
            model="atropos",
            object="response",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id=env_id,
            output=[],
        )

        verify_request = AtroposAgentVerifyRequest(
            response=dummy_response,
            responses_create_params=body.responses_create_params,
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        verify_response.raise_for_status()
        verify_data = await verify_response.json()

        await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={"env_id": env_id},
        )

        return AtroposNeMoGymResponse(
            id="atropos-batch",
            created_at=int(time.time()),
            model="atropos",
            object="response",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id=env_id,
            output=[],
            reward=verify_data.get("reward", 0.0),
            trajectory_data=trajectory_data,
        )


if __name__ == "__main__":
    AtroposAgent.run_webserver()
