#!/usr/bin/bash
PROFILE_CMD=""
PROFILE_CMD="nsys profile --output mmpp_profile.nsys-rep --cpuctxsw=none --trace=cublas,cuda,cudnn,cusolver,nvtx,osrt,python-gil --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --python-sampling=true"

# USE_MMPP=false
USE_MMPP=true

NVTE_FUSED_ATTN=1 $PROFILE_CMD python3 -m MaxText.train MaxText/configs/base.yml \
    run_name=logdir \
    model_name=llama2-7b \
    steps=10 \
    per_device_batch_size=2 \
    enable_checkpointing=false \
    base_output_directory=train_output \
    dataset_path=local \
    dataset_type=synthetic \
    hardware=gpu \
    enable_goodput_recording=false \
    monitor_goodput=false \
    enable_checkpoint_cloud_logger=false \
    dcn_fsdp_parallelism=1 \
    ici_fsdp_parallelism=1 \
    ici_data_parallelism=2 \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    dcn_tensor_parallelism=1 \
    ici_pipeline_parallelism=4 \
    dcn_pipeline_parallelism=1 \
    attention=cudnn_flash_te \
    remat_policy=minimal_flash \
    gradient_clipping_threshold=0 \
    use_mmpp=$USE_MMPP
