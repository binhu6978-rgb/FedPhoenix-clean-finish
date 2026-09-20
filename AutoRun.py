import subprocess
import itertools
import os
import time
import numpy as np
from typing import Dict, List
import pynvml  
pynvml.nvmlInit()
"""
Experiment Configuration Parameters
Modify this part of the parameters to configure different experiments
"""

#----------------------------------------
# 1. Dataset Related Parameters
#----------------------------------------
datasets = [
    'cifar10',  
    #'cifar100',
    #'timage'
]

# Number of classes for each dataset
dataset_classes = {
    'cifar10': 10,
    'cifar100': 20,
    'timage': 200,
    'femnist':62
}

#----------------------------------------
# 2. Algorithm Related Parameters
#----------------------------------------
algorithms = [
    #'FedGen',         
    #'FedMut',          
    'FedPhoenix',
    #'FedAvg',
    #'FedProx',
    #'ClusteredSampling',   

]

#----------------------------------------
# 3. Model Related Parameters
#----------------------------------------
models = [
    'resnet18',   
    #'resnet_drop', 
    #'mobnet'   ,      
    #'vgg'
    #'vggdrop',
]

#----------------------------------------

# 4. Training Related Parameters
#----------------------------------------
lrs = [0.01]           #Learning Rate
epochs = [1200]        # 
data_betas = [0.5,0.3,0.6]  #If you need to set it as IID, please use 0.5.   
weight_decays = [0.0000]  
#----------------------------------------
# 5. Federated Learning Related Parameters
#---------------------------------------- 
num_users = [100]       # Total number of clients
frac = [0.1]          # Fraction of clients participating in each round

#----------------------------------------
# 6. Specific Algorithm Parameters
#----------------------------------------
# FedPhoenix algorithm related parameters
fp_convs = [1000]      # r_s
resets = [2/64]       # Reset rate   θ
fp_fcs = [0]         
#----------------------------------------
# 7. Hardware Related Parameters
#----------------------------------------
gpus = [0]            # List of GPU IDs to use
# If you have multiple GPUs, you can set it to:
# gpus = [0,1,2,3]    # Use GPU 0, 1, 2, 3

#----------------------------------------
# 8. Resource Management Parameters
#----------------------------------------
# GPU usage threshold settings
GPU_MEMORY_THRESHOLD = 0.85  # GPU memory usage threshold (85%)
GPU_UTIL_THRESHOLD = 90   # GPU utilization threshold (90%)

# Task scheduling parameters
TASK_INIT_SLEEP = 20   # Task initialization wait time (seconds)
SCHEDULER_SLEEP_BUSY = 10    # Scheduler busy polling interval (seconds)
SCHEDULER_SLEEP_IDLE = 10    # Scheduler idle polling interval (seconds)

"""
Usage Example:

1. Running a single configuration:
   - Keep the required parameters, comment out the other options
   
2. Running multiple configurations:
   - Add multiple values in the lists
   - For example, models = ['resnet18', 'mobnet']
   
3. Common configuration combinations:
   CIFAR10 + ResNet18:
   datasets = ['cifar10']
   models = ['resnet18']
   
   CIFAR100 + MobileNet:
   datasets = ['cifar100']
   models = ['mobnet']
   
4. GPU configuration:
   Single GPU:
   gpus = [0]
   
   Multiple GPUs:
   gpus = [0,1,2,3]
"""

#----------------------------------------
# Parameter Validation
#----------------------------------------
def validate_params():
    """Validate the legality of the parameter configuration"""
    assert all(d in dataset_classes for d in datasets), "Invalid dataset name"
    # assert all(m in ['resnet18', 'mobnet'] for m in models), "Invalid model name"
    # assert all(a in ['FedGen', 'FedMut', 'ClusteredSampling', 'FedPhoenix'] 
    #           for a in algorithms), "Invalid algorithm name"
    assert all(0 < lr <= 1 for lr in lrs), "Learning rate must be in the range (0,1]"
    assert all(e > 0 for e in epochs), "Training epochs must be positive"
    assert all(0 < f <= 1 for f in frac), "Client sampling fraction must be in the range (0,1]"
    
    # Check if the GPU IDs are valid
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        assert all(g < device_count for g in gpus), f"GPU ID out of range, the system has {device_count} GPUs"
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False
    
    return True

# Validate the parameters before the main program starts
if not validate_params():
    raise ValueError("Parameter validation failed, please check the configuration")

#----------------------------------------
# Generate Parameter Combinations
#----------------------------------------
# Use itertools.product to generate all parameter combinations
variable_combinations = list(itertools.product(
    datasets,        # Datasets
    algorithms,      # Algorithms
    lrs,            # Learning rates
    epochs,         # Training epochs
    models,         # Models
    data_betas,     # Data distribution parameters
    fp_convs,       # FedPhoenix convolution layer parameters
    fp_fcs,         # FedPhoenix fully connected layer parameters
    resets,         # Reset rate
    num_users,      # Number of clients
    frac,            # Sampling fraction
    weight_decays   # Weight decay
))
# Dynamically generate fixed_params (key modification)
fixed_params = []
for params in variable_combinations:
    data_beta = params[5]  # Data beta parameter is at index 6
    if not np.isclose(data_beta, 0.5):
        fixed_params.append({'iid': 0, 'noniid_case': 5})  # Non-IID
    else:
        fixed_params.append({'iid': 1, 'noniid_case': 0})  # IID

print(f"A total of {len(variable_combinations)} parameter combinations were generated")

class GPUManager:
    """GPU Manager: Responsible for monitoring and managing the status and task allocation of multiple GPUs"""
    
    def __init__(self, gpu_ids: List[int]):
        """
        Initialize the GPU Manager
        Args:
            gpu_ids: List of GPU device IDs
        """
        self.gpu_ids = gpu_ids
        # Store the list of processes currently running on each GPU (supports multiple processes)
        self.processes: Dict[int, List[subprocess.Popen]] = {gpu: [] for gpu in gpu_ids}
        # Get the handle for each GPU, for later status queries
        self.gpu_handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) 
            for gpu in gpu_ids
        }
        
    def get_gpu_stats(self, gpu_id: int) -> dict:
        """
        Get detailed GPU status information
        Args:
            gpu_id: GPU device ID
        Returns:
            A dictionary containing information such as memory usage, GPU utilization, etc.
        """
        handle = self.gpu_handles[gpu_id]
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        return {
            'memory_used': info.used / info.total,  # Memory usage rate
            'gpu_utilization': utilization.gpu,     # GPU utilization
            'memory_total': info.total              # Total memory size
        }

    def is_gpu_available(self, gpu_id: int, memory_threshold=0.85, util_threshold=85) -> bool:
        """
        Check if a GPU is available for a new task
        Args:
            gpu_id: GPU device ID
            memory_threshold: Memory usage threshold (default 0.85, i.e., 85%)
            util_threshold: GPU utilization threshold (default 85%)
        Returns:
            Boolean indicating whether the GPU is available
        """
        try:
            stats = self.get_gpu_stats(gpu_id)
            return (stats['memory_used'] < memory_threshold and 
                   stats['gpu_utilization'] < util_threshold)
        except:
            return False

    def get_best_gpu(self) -> int:
        """
        Get the GPU with the lowest current load
        Returns:
            The device ID of the GPU with the lowest load, or None if no available GPU
        """
        gpu_loads = {}
        for gpu_id in self.gpu_ids:
            stats = self.get_gpu_stats(gpu_id)
            # Calculate the overall load score: 0.7 weight for memory usage, 0.3 weight for GPU utilization
            load_score = stats['memory_used'] * 0.7 + stats['gpu_utilization'] / 100 * 0.3
            gpu_loads[gpu_id] = load_score
        
        # Find the GPU with the lowest load that is available
        available_gpus = {gpu: load for gpu, load in gpu_loads.items() 
                         if self.is_gpu_available(gpu)}
        return min(available_gpus.items(), key=lambda x: x[1])[0] if available_gpus else None

    def add_process(self, gpu_id: int, process: subprocess.Popen):
        """
        Add a process to the list of processes running on the specified GPU
        Args:
            gpu_id: GPU device ID
            process: Process object
        """
        self.processes[gpu_id].append(process)

    def cleanup_processes(self):
        """
        Clean up completed processes
        """
        for gpu_id in self.gpu_ids:
            self.processes[gpu_id] = [p for p in self.processes[gpu_id] if p.poll() is None]

def run_training(params, gpu_id, iid, noniid_case):
    """
    Start a training task
    Args:
        params: Training parameters
        gpu_id: Specified GPU ID
        iid: Whether it is an IID setting
        noniid_case: Non-IID case setting
    Returns:
        Training process object
    """
    num_classes = dataset_classes[params[0]]
    
    # Construct the training command
    command = [
        'python', 'main_fed.py',
        '--dataset', params[0],
        '--algorithm', params[1],
        '--lr', str(params[2]),
        '--epoch', str(params[3]),
        '--model', params[4],
        '--iid', str(iid),
        '--noniid_case', str(noniid_case),
        '--data_beta', str(params[5]),
        '--generate_data', str(1),
        '--gpu', str(gpu_id),
        '--num_classes', str(num_classes),
        '--FP_conv', str(params[6]),
        '--FP_fc', str(params[7]),
        '--reset', str(params[8]),
        '--num_users', str(params[9]),
        '--frac', str(params[10]),
        '--weight_decay', str(params[11])  # Add weight decay parameter
    ]
    
    # # Set environment variables to enable dynamic GPU memory growth for optimal memory usage
    # env = os.environ.copy()
    # env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # return subprocess.Popen(command, env=env)

    # Start the process
    # Start the process, redirect output to a pipe
    process = subprocess.Popen(command)
    
   

def main():
    gpu_manager = GPUManager(gpus)
    tasks = list(zip(variable_combinations, fixed_params))
    running_tasks = []
    
    while tasks or running_tasks:
        # Clean up completed processes
        gpu_manager.cleanup_processes()
        
        # Schedule new tasks as much as possible
        while tasks:
            best_gpu = gpu_manager.get_best_gpu()
            if best_gpu is None:
                break
                
            # Get a new task
            params, fixed_param = tasks.pop(0)
            process = run_training(params, best_gpu, 
                                 fixed_param['iid'], 
                                 fixed_param['noniid_case'])
            gpu_manager.add_process(best_gpu, process)
            running_tasks.append((best_gpu, process))
            print(f"Started task on GPU {best_gpu} with params {params}")
            
            # Sleep briefly after task initialization
            time.sleep(TASK_INIT_SLEEP)
        
        # Check the status of running tasks
        running_tasks = [(gpu, p) for gpu, p in running_tasks if p.poll() is None]
        
        # If there are running tasks, sleep briefly; otherwise, sleep longer
        sleep_time = SCHEDULER_SLEEP_BUSY if running_tasks else SCHEDULER_SLEEP_IDLE
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
