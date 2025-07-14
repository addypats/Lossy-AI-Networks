#!/usr/bin/env python3
"""
Monitor Megatron training and log to W&B
"""
import os
import sys
import time
import json
import wandb
import argparse
from collections import deque
import numpy as np

class TrainingMonitor:
    def __init__(self, checkpoint_dir, target_accuracy=0.75, eval_interval=20):
        self.checkpoint_dir = checkpoint_dir
        self.target_accuracy = target_accuracy
        self.eval_interval = eval_interval
        self.accuracy_history = deque(maxlen=5)
        self.step = 0
        self.start_time = time.time()
        
        # Initialize W&B
        wandb.init(
            project="megatron-gpt2-mnli-finetune",
            name=f"gpt2-large-mnli-monitoring",
            config={
                "target_accuracy": target_accuracy,
                "eval_interval": eval_interval
            }
        )
    
    def parse_log_file(self, log_file):
        """Parse Megatron log file for metrics"""
        if not os.path.exists(log_file):
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for latest evaluation results
            for line in reversed(lines):
                if "validation loss" in line.lower():
                    # Parse validation metrics
                    # This is a simplified parser - adapt based on actual Megatron output
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "iteration" in part.lower() and i < len(parts) - 1:
                            try:
                                iteration = int(parts[i + 1])
                                return {"iteration": iteration, "line": line}
                            except:
                                continue
        except Exception as e:
            print(f"Error parsing log: {e}")
        
        return None
    
    def simulate_evaluation(self):
        """Simulate evaluation results (replace with actual evaluation)"""
        # This is a placeholder - replace with actual model evaluation
        base_acc = 0.33 + (len(self.accuracy_history) * 0.05)
        noise = np.random.normal(0, 0.02)
        accuracy = min(0.85, max(0.30, base_acc + noise))
        return accuracy
    
    def check_convergence(self):
        """Check if running average meets target"""
        if len(self.accuracy_history) >= 5:
            running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
            return running_avg >= self.target_accuracy
        return False
    
    def monitor(self):
        """Main monitoring loop"""
        print(f"Monitoring training in: {self.checkpoint_dir}")
        print(f"Target accuracy: {self.target_accuracy}")
        
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            # Check for new checkpoints or logs
            log_file = os.path.join(self.checkpoint_dir, "log.txt")
            
            if self.step % self.eval_interval == 0 and self.step > 0:
                # Simulate evaluation (replace with actual)
                accuracy = self.simulate_evaluation()
                self.accuracy_history.append(accuracy)
                
                running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
                elapsed = time.time() - self.start_time
                
                print(f"Step {self.step}: Accuracy {accuracy:.4f}, "
                      f"Running Avg {running_avg:.4f}, "
                      f"Time {elapsed:.1f}s")
                
                # Log to W&B
                wandb.log({
                    "accuracy": accuracy,
                    "running_accuracy": running_avg,
                    "step": self.step,
                    "elapsed_time": elapsed
                })
                
                # Check convergence
                if self.check_convergence():
                    print(f"Target accuracy achieved! Stopping.")
                    wandb.finish()
                    break
            
            self.step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument("--eval-interval", type=int, default=20)
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.checkpoint_dir, args.target_accuracy, args.eval_interval)
    monitor.monitor()
