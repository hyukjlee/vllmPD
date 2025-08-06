#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import signal
import json
import socket
import requests
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional
from kill_process import get_kfd_pids_with_vram, kill_pids

class VLLMDisaggregatedBenchmark:
    """
    Benchmark tool for comparing chunked prefill vs disaggregated prefill in vLLM.
    
    This class manages the lifecycle of vLLM server instances, runs benchmarks,
    and collects performance data for analysis.
    """
    
    # Configuration constants
    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    DEFAULT_RESULTS_DIR = "./results"
    DEFAULT_DATASET_NAME = "" #For sonnet or other datasets
    DEFAULT_DATASET_PATH = "" #Path to the dataset file, e.g., sonnet.txt
    
    # Server configuration
    CHUNKED_PREFILL_PORTS = [8100, 8200] 
    DISAGG_PREFILL_PORTS = [8100, 8200] 
    PROXY_PORT = 8000
    GPU_DEVICES = ['5', '6']
    
    # Benchmark parameters
    DEFAULT_NUM_PROMPTS = 100
    DEFAULT_PREFIX_LENGTH = 50
    DEFAULT_INPUT_LENGTH = 1024
    DEFAULT_OUTPUT_LENGTH = 1024
    DEFAULT_QPS_VALUES = [8]
    
    # Timeout and resource settings
    SERVER_STARTUP_TIMEOUT = 1200
    GPU_MEMORY_UTILIZATION = 0.9
    MAX_MODEL_LENGTH = 10000
    KV_BUFFER_SIZE = 5e9

    def __init__(self, model: str = None, results_dir: str = None):
        """Initialize the benchmark runner with configurable parameters."""
        self.model = model or self.DEFAULT_MODEL
        self.results_directory = Path(results_dir or self.DEFAULT_RESULTS_DIR)
        self.dataset_name = self.DEFAULT_DATASET_NAME
        self.dataset_path = Path(self.DEFAULT_DATASET_PATH)
        
        # Runtime tracking
        self.active_processes: List[subprocess.Popen] = []
        self.benchmark_config = self._create_benchmark_config()
        
    def _create_benchmark_config(self) -> Dict[str, Any]:
        """Create the benchmark configuration dictionary."""
        return {
            'num_prompts': self.DEFAULT_NUM_PROMPTS,
            'prefix_length': self.DEFAULT_PREFIX_LENGTH,
            'input_length': self.DEFAULT_INPUT_LENGTH,
            'output_length': self.DEFAULT_OUTPUT_LENGTH,
            'qps_values': self.DEFAULT_QPS_VALUES
        }

    def kill_all_ports(self) -> None:
        """
        Kill all processes running on the benchmark ports.
        
        This method forcefully terminates any processes using the ports
        defined in the class configuration (PROXY_PORT, CHUNKED_PREFILL_PORTS, 
        DISAGG_PREFILL_PORTS).
        """
        print("Killing processes on all benchmark ports...")
        
        # Collect all ports used by the benchmark system
        all_ports = [self.PROXY_PORT] + self.CHUNKED_PREFILL_PORTS + self.DISAGG_PREFILL_PORTS
        # Remove duplicates while preserving order
        unique_ports = list(dict.fromkeys(all_ports))
        
        killed_processes = 0
        
        for port in unique_ports:
            try:
                # Find processes using the port
                result = subprocess.run(['lsof', '-t', f'-i:{port}'], 
                                    capture_output=True, text=True, check=False)
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                    
                    for pid in pids:
                        try:
                            # First try graceful termination
                            subprocess.run(['kill', '-TERM', pid], check=False)
                            time.sleep(1)  # Give it a moment to terminate gracefully
                            
                            # Check if process still exists, if so force kill
                            check_result = subprocess.run(['kill', '-0', pid], 
                                                        capture_output=True, check=False)
                            if check_result.returncode == 0:
                                subprocess.run(['kill', '-9', pid], check=False)
                            
                            killed_processes += 1
                            print(f"Killed process {pid} on port {port}")
                            
                        except subprocess.CalledProcessError:
                            # Process might have already terminated
                            continue
                            
                else:
                    print(f"No processes found on port {port}")
                    
            except subprocess.CalledProcessError as e:
                print(f"Error checking port {port}: {e}")
        
        if killed_processes > 0:
            print(f"Total processes killed: {killed_processes}")
            time.sleep(2)  # Allow system to clean up
        else:
            print("No processes were killed on benchmark ports")

    def cleanup_gpu_processes(self) -> None:
        """
        Terminate all GPU processes and free up resources.
        
        This method kills processes by name, frees up specific ports,
        and terminates any processes started by this benchmark runner.
        """
        print("Cleaning up GPU processes...")
        
        #self._kill_processes_by_name(['pt_main_thread', 'python3', 'VLLM'])
        pids = get_kfd_pids_with_vram()    
        kill_pids(pids)
        os.system("rocm-smi --showpids")

        #self._kill_processes_on_ports([8000, 8100, 8200])
        self.kill_all_ports()
        self._terminate_managed_processes()
        
        time.sleep(1)  # Allow cleanup to complete
        print("GPU process cleanup completed.")

    def _kill_processes_by_name(self, process_names: List[str]) -> None:
        """Kill processes matching the given names."""
        for name in process_names:
            try:
                result = subprocess.run(['pgrep', name], capture_output=True, text=True)
                if result.returncode == 0:
                    pids = [pid for pid in result.stdout.strip().split('\n') if pid]
                    for pid in pids:
                        subprocess.run(['kill', '-9', pid], check=False)
            except subprocess.CalledProcessError:
                continue

    def _kill_processes_on_ports(self, ports: List[int]) -> None:
        """Kill processes using the specified ports."""
        for port in ports:
            try:
                result = subprocess.run(['lsof', '-t', f'-i:{port}'], capture_output=True, text=True)
                if result.returncode == 0:
                    pids = [pid for pid in result.stdout.strip().split('\n') if pid]
                    for pid in pids:
                        subprocess.run(['kill', '-9', pid], check=False)
            except subprocess.CalledProcessError:
                continue

    def _terminate_managed_processes(self) -> None:
        """Terminate processes managed by this benchmark instance."""
        for process in self.active_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        
        self.active_processes.clear()

    def wait_for_server_ready(self, port: int, timeout: int = None) -> bool:
        """
        Wait for a vLLM server to become ready on the specified port.
        
        Args:
            port: The port number to check
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False if timeout exceeded
        """
        timeout = timeout or self.SERVER_STARTUP_TIMEOUT
        print(f"Waiting for server on port {port}...")
        start_time = time.time()
        
        health_endpoints = ['/v1/models', '/health']
        
        while time.time() - start_time < timeout:
            for endpoint in health_endpoints:
                if self._check_server_endpoint(port, endpoint):
                    print(f"Server on port {port} is ready.")
                    return True
            time.sleep(1)
        
        print(f"Server on port {port} failed to start within {timeout} seconds.")
        return False

    def _check_server_endpoint(self, port: int, endpoint: str) -> bool:
        """Check if a server endpoint is responding."""
        try:
            response = requests.get(f"http://localhost:{port}{endpoint}", timeout=1)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def launch_chunked_prefill_servers(self) -> None:
        """
        Launch vLLM servers configured for chunked prefill.
        
        Starts two server instances with chunked prefill enabled and
        a round-robin proxy to distribute requests.
        """
        print("Launching chunked prefill servers...")
        
        # Launch server instances
        for i, (gpu_id, port) in enumerate(zip(self.GPU_DEVICES, self.CHUNKED_PREFILL_PORTS)):
            server_process = self._start_chunked_prefill_server(gpu_id, port)
            self.active_processes.append(server_process)
        
        # Verify both servers are ready
        if not all(self.wait_for_server_ready(port) for port in self.CHUNKED_PREFILL_PORTS):
            raise RuntimeError("Failed to start chunked prefill servers")
        
        # Start the proxy server
        proxy_process = self._start_proxy_server('round_robin_proxy.py')
        self.active_processes.append(proxy_process)
        
        print("Chunked prefill servers launched successfully.")

    def _start_chunked_prefill_server(self, gpu_id: str, port: int) -> subprocess.Popen:
        """Start a single chunked prefill server instance."""
        environment = os.environ.copy()
        environment['HIP_VISIBLE_DEVICES'] = gpu_id
        environment['VLLM_USE_V1'] = '1'
        #environment['VLLM_TORCH_PROFILER_DIR'] = './profile/'
        command = [
            'python3', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.model,
            '--port', str(port),
            '--max-model-len', str(self.MAX_MODEL_LENGTH),
            '--enable-chunked-prefill',
            '--gpu-memory-utilization', str(self.GPU_MEMORY_UTILIZATION),
            '--disable-log-requests',
        ]
        
        return subprocess.Popen(command, env=environment)

    def launch_disaggregated_prefill_servers(self) -> None:
        """
        Launch vLLM servers configured for disaggregated prefill.
        
        Starts a producer-consumer pair with KV cache transfer
        and a specialized proxy server.
        """
        print("Launching disaggregated prefill servers...")
        
        # Launch producer and consumer servers
        producer_process = self._start_disagg_producer_server()
        consumer_process = self._start_disagg_consumer_server()
        
        self.active_processes.extend([producer_process, consumer_process])
        
        # Verify both servers are ready
        if not all(self.wait_for_server_ready(port) for port in self.DISAGG_PREFILL_PORTS):
            raise RuntimeError("Failed to start disaggregated prefill servers")
        
        # Start the disaggregated prefill proxy
        proxy_process = self._start_proxy_server('disagg_prefill_proxy_server.py')
        self.active_processes.append(proxy_process)
        
        print("Disaggregated prefill servers launched successfully.")

    def _start_disagg_producer_server(self) -> subprocess.Popen:
        """Start the KV cache producer server."""
        environment = os.environ.copy()
        environment['HIP_VISIBLE_DEVICES'] = self.GPU_DEVICES[1]
        environment['VLLM_USE_V1'] = '1'
        #environment['VLLM_TORCH_PROFILER_DIR'] = './profile/'
        kv_config = {
            "kv_connector": "PyNcclConnector",
            "kv_role": "kv_producer", 
            "kv_rank": 0,
            "kv_parallel_size": 2,
            "kv_buffer_size": self.KV_BUFFER_SIZE
        }
        
        command = [
            'python3', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.model,
            '--port', str(self.DISAGG_PREFILL_PORTS[0]),
            '--max-model-len', str(self.MAX_MODEL_LENGTH),
            '--gpu-memory-utilization', str(self.GPU_MEMORY_UTILIZATION),
            '--kv-transfer-config', json.dumps(kv_config),
            '--disable-log-requests',
        ]
        
        return subprocess.Popen(command, env=environment)

    def _start_disagg_consumer_server(self) -> subprocess.Popen:
        """Start the KV cache consumer server."""
        environment = os.environ.copy()
        #environment['CUDA_VISIBLE_DEVICES'] = self.GPU_DEVICES[1]
        environment['HIP_VISIBLE_DEVICES'] = self.GPU_DEVICES[1]
        environment['VLLM_USE_V1'] = '1'
        #environment['VLLM_TORCH_PROFILER_DIR'] = './profile/'
        kv_config = {
            "kv_connector": "PyNcclConnector",
            "kv_role": "kv_consumer",
            "kv_rank": 1, 
            "kv_parallel_size": 2,
            "kv_buffer_size": self.KV_BUFFER_SIZE
        }
        
        command = [
            'python3', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.model,
            '--port', str(self.DISAGG_PREFILL_PORTS[1]),
            '--max-model-len', str(self.MAX_MODEL_LENGTH),
            '--gpu-memory-utilization', str(self.GPU_MEMORY_UTILIZATION),
            '--kv-transfer-config', json.dumps(kv_config),
            '--disable-log-requests',
        ]
        
        return subprocess.Popen(command, env=environment)

    def _start_proxy_server(self, proxy_script: str) -> subprocess.Popen:
        """Start a proxy server with the given script."""
        time.sleep(1)  # Allow servers to fully initialize
        return subprocess.Popen(['python3', proxy_script])

    def run_benchmark(self, qps: int, output_length: int, benchmark_tag: str) -> None:
        """
        Execute a single benchmark run with specified parameters.
        
        Args:
            qps: Queries per second rate
            output_length: Expected output token length
            benchmark_tag: Identifier for this benchmark configuration
        """
        print(f"Running benchmark: QPS={qps}, Output Length={output_length}, Tag={benchmark_tag}")
        
        command = self._build_benchmark_command(qps, output_length, benchmark_tag)
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Benchmark completed successfully for {benchmark_tag} QPS {qps}")
        except subprocess.CalledProcessError as error:
            print(f"Benchmark failed for {benchmark_tag} QPS {qps}: {error}")
            print(f"stdout: {error.stdout}")
            print(f"stderr: {error.stderr}")
        
        time.sleep(2)  # Allow system to stabilize between runs

    def _build_benchmark_command(self, qps: int, output_length: int, benchmark_tag: str) -> List[str]:
        """Build the benchmark command with all necessary parameters."""
        result_filename = f"{benchmark_tag}-qps-{qps}.json"
        np = qps * 8
        return [
            'vllm', 'bench', 'serve',
            '--model', self.model,
            '--dataset-name', 'random',
            '--random-input-len', str(self.benchmark_config['input_length']),
            '--random-output-len', str(output_length),
            '--num-prompts', str(np),
            '--port', str(self.PROXY_PORT),
            '--save-result',
            '--result-dir', str(self.results_directory),
            '--result-filename', result_filename,
            '--max_concurrency', str(qps),
            '--profile',
        ]

    def install_system_dependencies(self) -> None:
        """Install required system and Python dependencies."""
        print("Installing dependencies...")
        
        self._install_system_packages()
        self._install_python_packages()
        
        print("Dependency installation completed.")

    def _install_system_packages(self) -> None:
        """Install required system packages."""
        required_packages = ['wget', 'curl', 'jq', 'socat', 'lsof']
        missing_packages = [pkg for pkg in required_packages if shutil.which(pkg) is None]
        
        if missing_packages:
            print(f"Installing missing system packages: {missing_packages}")
            try:
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y'] + missing_packages, check=True)
            except subprocess.CalledProcessError:
                print("Warning: Could not install system packages. Ensure they are available.")

    def _install_python_packages(self) -> None:
        """Install required Python packages."""
        required_packages = ['quart', 'httpx', 'matplotlib', 'aiohttp', 'datasets', 'requests']
        try:
            subprocess.run(['pip', 'install'] + required_packages, check=True)
            print("Python packages installed successfully.")
        except subprocess.CalledProcessError as error:
            print(f"Warning: Could not install Python packages: {error}")

    def setup_benchmark_dataset(self) -> None:
        """Create the benchmark dataset by replicating the base sonnet file."""
        print("Setting up benchmark dataset...")
        
        current_directory = Path.cwd()
        base_sonnet_path = current_directory / "sonnet.txt"
        extended_sonnet_path = current_directory / "sonnet_4x.txt"
        
        if not base_sonnet_path.exists():
            print(f"Warning: {base_sonnet_path} does not exist. Please ensure the base sonnet.txt file is available.")
            return
        
        self._create_extended_dataset(base_sonnet_path, extended_sonnet_path)
        
        print(f"Created extended dataset: {extended_sonnet_path}")

    def _create_extended_dataset(self, source_path: Path, target_path: Path) -> None:
        """Create an extended dataset by replicating the source file."""
        with open(target_path, 'w') as output_file:
            output_file.write("\n")
            for _ in range(4):  # Replicate 4 times
                with open(source_path, 'r') as input_file:
                    output_file.write(input_file.read())

    def setup_results_directory(self) -> None:
        """Initialize the results directory, removing any existing data."""
        if self.results_directory.exists():
            shutil.rmtree(self.results_directory)
        
        self.results_directory.mkdir(parents=True, exist_ok=True)
        print(f"Results directory initialized: {self.results_directory}")

    def run_chunked_prefill_benchmark_suite(self) -> None:
        """Execute the complete chunked prefill benchmark suite."""
        print("\n=== Running Chunked Prefill Benchmark Suite ===")
        
        self.launch_chunked_prefill_servers()
        
        # Run benchmarks without profiling
        for qps in self.benchmark_config['qps_values']:
            self.run_benchmark(qps, self.benchmark_config['output_length'], "chunked_prefill")
        
        self.cleanup_gpu_processes()

    def run_disaggregated_prefill_benchmark_suite(self) -> None:
        """Execute the complete disaggregated prefill benchmark suite."""
        print("\n=== Running Disaggregated Prefill Benchmark Suite ===")
        
        self.launch_disaggregated_prefill_servers()
        
        # Run benchmarks without profiling
        for qps in self.benchmark_config['qps_values']:
            self.run_benchmark(qps, self.benchmark_config['output_length'], "disagg_prefill")
        
        self.cleanup_gpu_processes()

    def generate_benchmark_visualization(self) -> None:
        """Generate visualization of benchmark results."""
        print("Generating benchmark visualization...")
        try:
            subprocess.run(['python3', 'visualize_benchmark_results.py'], check=True)
            print("Visualization completed successfully.")
        except subprocess.CalledProcessError as error:
            print(f"Visualization generation failed: {error}")

    def get_host_ip_address(self) -> str:
        """
        Determine the host IP address for vLLM configuration.
        
        Returns:
            The host IP address as a string
        """
        # Try hostname command first
        try:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split()[0]
        except subprocess.CalledProcessError:
            pass
        
        # Fallback to socket-based detection
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def execute_benchmark_suite(self) -> None:
        """
        Execute the complete benchmark suite.
        
        This is the main entry point that orchestrates the entire benchmarking process.
        """
        try:
            # Environment setup
            os.environ['VLLM_HOST_IP'] = self.get_host_ip_address()
            
            # Change to script directory for relative path resolution
            script_directory = Path(__file__).parent
            os.chdir(script_directory)
            
            # Preparation phase
            self.install_system_dependencies()
            self.setup_benchmark_dataset()
            self.setup_results_directory()
            
            # Benchmark execution phase
            self.run_chunked_prefill_benchmark_suite()
            self.run_disaggregated_prefill_benchmark_suite()
            
            # Results visualization
            self.generate_benchmark_visualization()
            
            print("\nAll benchmarks completed successfully!")
            
        except KeyboardInterrupt:
            print("\nBenchmark suite interrupted by user.")
            self.cleanup_gpu_processes()
            sys.exit(1)
        except Exception as error:
            print(f"Error during benchmark execution: {error}")
            self.cleanup_gpu_processes()
            sys.exit(1)


def create_signal_handler(benchmark_instance: VLLMDisaggregatedBenchmark):
    """Create a signal handler for graceful shutdown."""
    def signal_handler(signal_number, frame):
        print(f"\nReceived signal {signal_number}. Cleaning up...")
        benchmark_instance.cleanup_gpu_processes()
        sys.exit(0)
    return signal_handler


def main():
    """Main entry point for the benchmark script."""
    # Create benchmark instance
    benchmark_runner = VLLMDisaggregatedBenchmark()
    
    # Set up signal handlers for graceful shutdown
    handler = create_signal_handler(benchmark_runner)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # Execute the benchmark suite
    benchmark_runner.execute_benchmark_suite()


if __name__ == "__main__":
    main()
