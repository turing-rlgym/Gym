# Atropos Integration for Nemo Gym

[Atropos](https://github.com/NousResearch/atropos) 


```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/gsm8k.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=gsm8k_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/gsm8k_sample.jsonl \
    +output_jsonl_fpath=results/gsm8k_rollouts.jsonl \
    +limit=5
```
