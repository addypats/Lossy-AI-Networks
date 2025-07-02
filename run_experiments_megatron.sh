#!/bin/bash
# Complete Megatron-LM + Lossy Network Experiment Script
# This script integrates your existing experimental setup with Megatron-LM's optimized tensor parallelism

# Environment setup (your existing NCCL configuration)
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5
export NCCL_IB_DISABLE=1
export NCCL_NET_OFI_DISABLE=1
export NCCL_P2P_LEVEL=SYS

# Debugging (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Torchrun settings
TORCHRUN=$(which torchrun)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# GPU configuration - adjust based on your setup
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # For 4-GPU setup
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # For 8-GPU setup

# Experimental parameters (your existing configuration)
TP_SIZE=(2)  # Tensor parallel sizes to test
GE_CONFIG=("zero")  # Your Gilbert-Elliott configurations
DATASETS=("mnli")   # Your datasets
FP_FLAGS=(fp32)     # Precision settings
ITERATIONS=(1)      # Number of iterations

# Create output directories
mkdir -p output_megatron_lossy_mnli
mkdir -p output_megatron_lossy_sst2

echo "🚀 Starting Megatron-LM + Lossy Network Experiments"
echo "📊 This script will test both Megatron-LM (if available) and your custom tensor parallelism"

# Function to run single experiment
run_experiment() {
    local tp_size=$1
    local dataset=$2
    local ge_config=$3
    local fp_flag=$4
    local iter=$5
    local use_megatron=$6
    
    if [ "$use_megatron" = "true" ]; then
        script_name="pytorch_train_tp_gpt_megatron.py"
        run_prefix="megatron"
        echo "🔧 Using Megatron-LM integration"
    else
        script_name="pytorch_train_tp_gpt.py"
        run_prefix="custom"
        echo "🔧 Using your custom tensor parallelism"
    fi
    
    run_id="target_steps_${run_prefix}_gpt2-large_precision-${fp_flag}_Num_Nodes-${tp_size}_ge_config_${ge_config}_Iteration_${iter}"
    output_dir="output_megatron_lossy_${dataset}/${run_id}"
    
    echo
    echo "=== Starting ${run_id} ==="
    echo "📁 Output directory: ${output_dir}"
    echo "🖥️  GPUs: $CUDA_VISIBLE_DEVICES"
    echo "🔀 Tensor parallel size: $tp_size"
    echo "📈 Loss config: $ge_config"
    echo
    
    # Determine precision flag
    if [ "$fp_flag" = "fp16" ]; then
        precision_arg="--fp16"
    else
        precision_arg=""
    fi
    
    # Run the experiment
    $TORCHRUN \
        --nproc_per_node $tp_size \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        src/$script_name \
            --tensor_parallel_size $tp_size \
            --loss_type g-e \
            --ge_config $ge_config \
            --model_name "gpt2-large" \
            --dataset $dataset \
            --batch_size 8 \
            --max_length 128 \
            --learning_rate 3e-5 \
            --weight_decay 0.01 \
            --loss_rate 0.001 \
            $precision_arg \
            --seed 1234 \
            --max_samples 0 \
            --target_accuracy 0.75 \
            --eval_steps 20 \
            --patience 15 \
            --max_steps 500 \
            --output_dir "$output_dir"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed ${run_id}"
    else
        echo "❌ Failed ${run_id} with exit code $exit_code"
    fi
    
    echo
}

# Function to test if Megatron-LM is available
test_megatron_availability() {
    echo "🔍 Testing Megatron-LM availability..."
    
    python3 -c "
try:
    from megatron.core.tensor_parallel import mappings
    print('✅ Megatron-LM is available')
    exit(0)
except ImportError:
    print('⚠️  Megatron-LM is not available')
    exit(1)
" 2>/dev/null
    
    return $?
}

# Main experiment loop
main() {
    echo "🧪 Starting comprehensive lossy network experiments"
    echo
    
    # Test Megatron availability
    if test_megatron_availability; then
        MEGATRON_AVAILABLE=true
        echo "📈 Will run experiments with both Megatron-LM and custom implementations for comparison"
    else
        MEGATRON_AVAILABLE=false
        echo "📉 Will run experiments with custom tensor parallelism only"
        echo "💡 To enable Megatron-LM: git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && pip install -e ."
    fi
    
    echo
    echo "🔬 Experiment Configuration:"
    echo "   Tensor Parallel Sizes: ${TP_SIZE[*]}"
    echo "   Datasets: ${DATASETS[*]}"
    echo "   Loss Configurations: ${GE_CONFIG[*]}"
    echo "   Precision: ${FP_FLAGS[*]}"
    echo "   Iterations: ${ITERATIONS[*]}"
    echo
    
    # Run experiments
    for iter in "${ITERATIONS[@]}"; do
        echo "🔄 === Starting iteration ${iter} ==="
        
        for fp_flag in "${FP_FLAGS[@]}"; do
            echo "⚙️  === Testing precision: ${fp_flag} ==="
            
            for tp_size in "${TP_SIZE[@]}"; do
                echo "🔀 === Tensor parallelism size: ${tp_size} ==="
                
                for dataset in "${DATASETS[@]}"; do
                    echo "📊 === Dataset: ${dataset} ==="
                    
                    for ge_config in "${GE_CONFIG[@]}"; do
                        echo "🌊 === Loss configuration: ${ge_config} ==="
                        
                        # Run with custom implementation
                        run_experiment $tp_size $dataset $ge_config $fp_flag $iter "false"
                        
                        # Run with Megatron if available
                        if [ "$MEGATRON_AVAILABLE" = "true" ]; then
                            run_experiment $tp_size $dataset $ge_config $fp_flag $iter "true"
                        fi
                        
                        echo "---"
                    done
                done
            done
        done
    done
    
    echo "🎉 All experiments completed!"
    echo
    echo "📁 Results are saved in:"
    echo "   - output_megatron_lossy_mnli/ (MNLI results)"
    echo "   - output_megatron_lossy_sst2/ (SST2 results)"
    echo
    echo "📊 Each experiment directory contains:"
    echo "   - args.yaml (experiment configuration)"
    echo "   - training.log (training progress)"
    echo "   - metrics.json (detailed metrics)"
    echo "   - model_best.pt (best model checkpoint)"
    echo "   - model_final.pt (final model checkpoint)"
    echo
    echo "💡 To analyze results, use the metrics.json files and compare:"
    echo "   - Accuracy convergence between Megatron vs Custom implementations"
    echo "   - Training speed and efficiency"
    echo "   - Lossy communication impact statistics"
}

# Utility functions for result analysis
analyze_results() {
    echo "📈 Analyzing experimental results..."
    
    python3 << 'EOF'
import os
import json
import pandas as pd
from pathlib import Path

def analyze_experiment_results():
    """Analyze results from all experiments."""
    results = []
    
    # Find all output directories
    for output_dir in ['output_megatron_lossy_mnli', 'output_megatron_lossy_sst2']:
        if not os.path.exists(output_dir):
            continue
            
        for experiment_dir in Path(output_dir).iterdir():
            if not experiment_dir.is_dir():
                continue
                
            metrics_file = experiment_dir / 'metrics.json'
            args_file = experiment_dir / 'args.yaml'
            
            if metrics_file.exists() and args_file.exists():
                try:
                    # Load metrics
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    # Load args
                    import yaml
                    with open(args_file) as f:
                        args = yaml.safe_load(f)
                    
                    # Extract key information
                    if metrics:
                        final_accuracy = metrics[-1]['accuracy']
                        total_time = metrics[-1]['time']
                        total_steps = metrics[-1]['step']
                        
                        # Check if this is Megatron or custom
                        implementation = 'megatron' if 'megatron' in str(experiment_dir) else 'custom'
                        
                        result = {
                            'experiment': experiment_dir.name,
                            'implementation': implementation,
                            'dataset': args.get('dataset', 'unknown'),
                            'tensor_parallel_size': args.get('tensor_parallel_size', 1),
                            'loss_type': args.get('loss_type', 'unknown'),
                            'ge_config': args.get('ge_config', 'unknown'),
                            'final_accuracy': final_accuracy,
                            'total_time': total_time,
                            'total_steps': total_steps,
                            'time_per_step': total_time / total_steps if total_steps > 0 else 0,
                        }
                        results.append(result)
                        
                except Exception as e:
                    print(f"Error processing {experiment_dir}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        print("📊 Experiment Results Summary:")
        print(df.to_string(index=False))
        
        # Save summary
        df.to_csv('experiment_summary.csv', index=False)
        print(f"💾 Detailed summary saved to experiment_summary.csv")
        
        # Compare implementations if both exist
        if 'megatron' in df['implementation'].values and 'custom' in df['implementation'].values:
            print("\n🔍 Megatron vs Custom Implementation Comparison:")
            comparison = df.groupby('implementation').agg({
                'final_accuracy': 'mean',
                'total_time': 'mean',
                'time_per_step': 'mean'
            }).round(4)
            print(comparison)
    else:
        print("❌ No experimental results found")

if __name__ == "__main__":
    analyze_experiment_results()
EOF
}

# Command line interface
case "${1:-run}" in
    "run")
        main
        ;;
    "analyze")
        analyze_results
        ;;
    "test-megatron")
        test_megatron_availability
        ;;
    "help")
        echo "Usage: $0 [run|analyze|test-megatron|help]"
        echo "  run         - Run all experiments (default)"
        echo "  analyze     - Analyze results from completed experiments"
        echo "  test-megatron - Test if Megatron-LM is available"
        echo "  help        - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
