import subprocess
import re
import os
import signal

def get_kfd_pids_with_vram():
    try:
        output = subprocess.check_output(["rocm-smi", "--showpids"], text=True)
    except subprocess.CalledProcessError as e:
        print("Failed to run rocm-smi:", e)
        return []

    pids_to_kill = []
    capture = False
    for line in output.splitlines():
        if "PID" in line and "VRAM USED" in line:
            capture = True
            continue
        if "===" in line:
            continue
        if capture and line.strip():
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 5:
                continue
            pid = int(parts[0])
            try:
                vram = int(parts[3])
                if vram > 0:
                    pids_to_kill.append(pid)
            except ValueError:
                continue
    return pids_to_kill

def kill_pids(pids):
    for pid in pids:        
        os.system(f"sudo kill -9 {pid}")
        print("killed pid:", pid)
        
if __name__ == "__main__":
    pids = get_kfd_pids_with_vram()    
    kill_pids(pids)
    os.system("rocm-smi --showpids")
