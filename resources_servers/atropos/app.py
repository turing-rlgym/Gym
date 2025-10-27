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
import asyncio
import importlib
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import Field

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.integrations.atropos import (
    AtroposAgentVerifyRequest,
    AtroposAgentVerifyResponse,
    AtroposCloseRequest,
    AtroposCloseResponse,
    AtroposResourcesServerConfig,
    AtroposSeedSessionRequest,
    AtroposSeedSessionResponse,
    AtroposStepRequest,
    AtroposStepResponse,
)
from nemo_gym.global_config import (
    POLICY_BASE_URL_KEY_NAME,
    POLICY_API_KEY_KEY_NAME,
    POLICY_MODEL_NAME_KEY_NAME,
    get_global_config_dict,
)


class AtroposServerConfig(AtroposResourcesServerConfig):
    atropos_path: str
    environment_module: str  # e.g., "environments.gsm8k_server"
    environment_class: str   # e.g., "GSM8kEnv"
    group_size: int = 8
    env_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system_prompt_attr: Optional[str] = "system_prompt"
    dataset_attr: Optional[str] = "train"


class AtroposResourcesServer(SimpleResourcesServer):
    config: AtroposServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_class: Optional[type] = None
        self._env_instance: Optional[Any] = None
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._startup()
            yield

        app = FastAPI(lifespan=lifespan)
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        app.post("/step")(self.step)
        app.post("/verify")(self.verify)
        app.post("/close")(self.close)
        return app

    async def _startup(self):
        print("=== Initializing Atropos Environment ===")

        atropos_path = Path(self.config.atropos_path).resolve()
        if str(atropos_path) not in sys.path:
            sys.path.insert(0, str(atropos_path))
            print(f"Added {atropos_path} to sys.path")

        try:
            module = importlib.import_module(self.config.environment_module)
            self._env_class = getattr(module, self.config.environment_class)
            print(f"Loaded: {self.config.environment_module}.{self.config.environment_class}")
        except Exception as e:
            print(f"ERROR: Failed to import environment: {e}")
            raise

        global_config = get_global_config_dict()
        policy_base_url = global_config.get(POLICY_BASE_URL_KEY_NAME, "")
        policy_api_key = global_config.get(POLICY_API_KEY_KEY_NAME, "EMPTY")
        policy_model_name = global_config.get(POLICY_MODEL_NAME_KEY_NAME, "policy_model")

        if not policy_base_url:
            raise ValueError(f"Must set {POLICY_BASE_URL_KEY_NAME} in env.yaml")

        print(f"Using vLLM: {policy_base_url}")

        default_env_config, default_server_configs = self._env_class.config_init()

        default_env_config.group_size = self.config.group_size
        for key, value in self.config.env_kwargs.items():
            setattr(default_env_config, key, value)

        if isinstance(default_server_configs, list):
            for server_config in default_server_configs:
                server_config.base_url = policy_base_url
                server_config.api_key = policy_api_key
                server_config.model_name = policy_model_name
        else:
            default_server_configs.base_url = policy_base_url
            default_server_configs.api_key = policy_api_key
            default_server_configs.model_name = policy_model_name

        try:
            self._env_instance = self._env_class(
                config=default_env_config,
                server_configs=default_server_configs if isinstance(default_server_configs, list) else [default_server_configs],
                slurm=False,
                testing=False,
            )

            if hasattr(self._env_instance, 'setup'):
                print("Calling environment setup()...")
                if asyncio.iscoroutinefunction(self._env_instance.setup):
                    await self._env_instance.setup()
                else:
                    self._env_instance.setup()
                print("Setup complete")

            if not hasattr(self._env_instance, self.config.dataset_attr):
                raise AttributeError(
                    f"Environment {self.config.environment_class} has no '{self.config.dataset_attr}' attribute. "
                    f"Available attributes: {[a for a in dir(self._env_instance) if not a.startswith('_')]}"
                )

            print("=== Atropos Environment Ready ===")
        except Exception as e:
            print(f"ERROR: Failed to initialize environment: {e}")
            raise

    async def seed_session(
        self,
        request: Request,
        body: AtroposSeedSessionRequest,
    ) -> AtroposSeedSessionResponse:
        env_id = str(uuid.uuid4())

        dataset = getattr(self._env_instance, self.config.dataset_attr, None)
        if dataset is None:
            attrs = [a for a in dir(self._env_instance) if not a.startswith('_')]
            print(f"ERROR: Environment has no '{self.config.dataset_attr}' attribute")
            print(f"Available attributes: {attrs[:20]}")
            raise ValueError(f"Environment has no '{self.config.dataset_attr}' attribute")

        item = dataset[body.task_idx % len(dataset)]

        print(f"Generating trajectory for task {body.task_idx}...")
        try:
            if asyncio.iscoroutinefunction(self._env_instance.collect_trajectories):
                result = await self._env_instance.collect_trajectories(item)
            else:
                result = self._env_instance.collect_trajectories(item)

            # (scored_data_group, new_items)
            scored_data, _ = result

            if scored_data is None:
                print("WARNING: collect_trajectories returned None")
                scored_data = {"tokens": [], "scores": [0.0], "masks": []}

            scores = scored_data.get("scores", [0.0])
            avg_reward = sum(scores) / len(scores) if scores else 0.0

            print(f"Generated trajectory with reward: {avg_reward}")

            self._sessions[env_id] = {
                "trajectory": scored_data,
                "avg_reward": avg_reward,
                "item": item,
            }

        except Exception as e:
            print(f"ERROR in collect_trajectories: {e}")
            import traceback
            traceback.print_exc()
            self._sessions[env_id] = {
                "trajectory": {"tokens": [], "scores": [0.0], "masks": []},
                "avg_reward": 0.0,
                "item": item,
            }

        system_prompt = getattr(self._env_instance, self.config.system_prompt_attr, None)

        return AtroposSeedSessionResponse(
            env_id=env_id,
            obs=[],
            system_prompt=system_prompt,
            metadata={
                "trajectory_data": self._sessions[env_id]["trajectory"],
                "avg_reward": self._sessions[env_id]["avg_reward"],
            },
        )

    async def step(
        self,
        request: Request,
        body: AtroposStepRequest,
    ) -> AtroposStepResponse:
        """No-op."""
        return AtroposStepResponse(obs=[], reward=0.0, done=True, info={})

    async def verify(
        self,
        request: Request,
        body: AtroposAgentVerifyRequest,
    ) -> AtroposAgentVerifyResponse:
        env_id = body.response.env_id
        session = self._sessions.get(env_id, {})

        return AtroposAgentVerifyResponse(
            **body.model_dump(),
            reward=session.get("avg_reward", 0.0),
            trajectory_data=session.get("trajectory", {}),
        )

    async def close(
        self,
        request: Request,
        body: AtroposCloseRequest,
    ) -> AtroposCloseResponse:
        self._sessions.pop(body.env_id, None)
        return AtroposCloseResponse(message="Session closed", success=True)

if __name__ == "__main__":
    AtroposResourcesServer.run_webserver()