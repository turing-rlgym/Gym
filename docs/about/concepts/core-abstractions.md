(core-abstractions)=

# Core Abstractions

Before diving into code, let's understand the three core abstractions in NeMo Gym.

> If you are new to reinforcement learning for LLMs, we recommend you review **[Key Terminology](./key-terminology)** first.

```{image} ../../_images/product_overview.svg
:alt: NeMo Gym Architecture
:width: 800px
:align: center
```

::::{tab-set}

:::{tab-item} Model

Responses API Model servers are stateless model endpoints that perform single-call text generation without conversation memory or orchestration. During training, you will always have at least one active Responses API Model server, typically called the "policy" model.

**Available Implementations:**
- `openai_model`: Direct integration with OpenAI's Responses API  
- `vllm_model`: Middleware converting local models (via vLLM) to Responses API format

**Configuration:** Models are configured with API endpoints and credentials via YAML files in `responses_api_models/*/configs/`

:::

:::{tab-item} Resources

Resources servers provide tool implementations that can be invoked via tool calling and verification logic that measures task performance. NeMo Gym includes various NVIDIA and community-contributed resources servers for use during training, and provides tutorials for creating your own Resource server.

**Resources Provide**
- **Tools**: Functions agents can call (e.g., `get_weather`, `search_web`)
- **Verification Logic**: Scoring systems that evaluate agent responses for training/evaluation

**Examples:**
- `simple_weather`: Mock weather API for testing and tutorials
- `google_search`: Web search capabilities via Google Search API  
- `math_with_code`: Python code execution environment for mathematical reasoning
- `math_with_judge`: Mathematical problem verification using symbolic computation
- `mcqa`: Multiple choice question answering evaluation
- `instruction_following`: General instruction compliance scoring


**Configuration**: See resource-specific config files in `resources_servers/*/configs/`

:::

:::{tab-item} Agents

Responses API Agent servers orchestrate the interaction between models and resources.

- Route requests to the right model
- Provide tools to the model
- Handle multi-turn conversations
- Format responses consistently

Agents are also called "training environments." NeMo Gym includes several training environment patterns covering multi-step, multi-turn, and user modeling scenarios.

**Examples:**
- `simple_agent`: Basic agent that coordinates model calls with resource tools

**Configuration Pattern**:
```yaml
your_agent_name:                     # server ID
  responses_api_agents:              # server type. corresponds to the folder name in the project root
    your_agent_name:                 # agent type. name of the folder inside the server type folder 
      entrypoint: app.py             # server entrypoint path, relative to the agent type folder 
      resources_server:              # which resource server to use
        name: simple_weather         
      model_server:                  # which model server to use
        name: policy_model           
```

:::
::::
