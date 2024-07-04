# Sound Instruct Llama3

## 1. Training Setup and Data Processing
<!-- readme chưa chấp nhận được
giải thích by default dùng data và model nào ở đâu (instruct và cái data có sẵn mình trên hf)
by default mọi thứ setup rồi ko cần làm gì nhưng nếu bạn muốn coi sound synthetic thì vô data processing -->

Check out [DATA PROCESSING](data/README.md)

## 2. Fire training code

### 2.1 Config accelerator to use multiple GPUS

- Run this script to start configuring

```bash
accelerate config
```

- Sample configuration:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: true
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 2.2 Fire the training

Run

```bash
accelerate launch --config_file {path/to/config/my_config_file.yaml} adam_mini_train.py
```

## 3. TODO

- [ ] Add single inference script
  - [ ] Add faster inference with vllm

## 4. Acknowledgement
Super duper thankful to the team behind [Adam-mini](https://github.com/zyushun/Adam-mini) and [Llama3](https://github.com/meta-llama/llama3) for their superb works.
