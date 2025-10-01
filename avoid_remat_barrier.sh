#!/bin/bash

# Usage: launch with
#  avoid_remat_barrier.sh                           # as usual all inputs of jax.rematted layer go through opt-barrier
#  avoid_remat_barrier.sh avoid_remat_barrier=true  # excludes params from said opt-barrier (but thread params through new opt-barrier between forward and backward)

# Some useful one-liners
#  grep -P 'opt-barrier\(' $(ls -t /tmp/avoid_remat_barrier_xla_out/module_*.jit_train_step.before_optimizations.txt | head -n 1) | wc -l
#  grep -P 'all-gather-start\(' $(ls -t /tmp/avoid_remat_barrier_xla_out/module_*.jit_train_step.sm_9.0a_gpu_after_optimizations.txt | head -n 1) | wc -l

export BASE_XLA_FLAGS=${BASE_XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_all_reduce_combine_threshold_bytes=2147483648
                --xla_gpu_all_gather_combine_threshold_bytes=4294967296
                --xla_gpu_reduce_scatter_combine_threshold_bytes=536870912
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization}

XLA_FLAGS="--xla_dump_to=/tmp/avoid_remat_barrier_xla_out --xla_dump_hlo_module_re='train_step'"
# XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_dot"
export XLA_FLAGS="$BASE_XLA_FLAGS ${XLA_FLAGS:-}"
export JAX_VJP3=1

python -m MaxText.train MaxText/configs/base.yml run_name=demo max_target_length=4096 steps=10 per_device_batch_size=2 model_name=llama3-8b remat_policy=minimal_flash enable_checkpointing=false logits_dot_in_fp32=false use_iota_embed=false tokenizer_path=assets/tokenizer_llama3.tiktoken tokenizer_type=tiktoken dataset_type=synthetic base_output_directory=/tmp/avoid_remat_barrier_train_out enable_goodput_recording=false monitor_goodput=false ici_fsdp_parallelism=-1 scan_layers=false "$@" &> avoid_remat_barrier.log
