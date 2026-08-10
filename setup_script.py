import platform
import subprocess
import sys
import os
import shutil

def run_command(command):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + command)

def check_gpu_and_install_torch(os_name):
    print(f"Detected Operating System: {os_name}")

    if os_name == "Darwin":
        print("Mac detected. Installing PyTorch with native Metal (MPS) support...")
        run_command(["torch", "torchvision", "torchaudio"])

    elif os_name == "Linux":
        print("Linux detected. Checking for GPU vendor...")
        
        if shutil.which("nvidia-smi"):
            print("NVIDIA GPU detected. Installing PyTorch with CUDA...")
            run_command(["torch", "torchvision", "torchaudio"])
            
        elif shutil.which("rocminfo") or shutil.which("rocm-smi"):
            print("AMD GPU detected. Installing PyTorch with ROCm...")
            run_command([
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/rocm5.6"
            ])
        else:
            print("No specific GPU driver detected. Installing CPU-only PyTorch...")
            run_command(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])

    elif os_name == "Windows":
        print("Windows detected. Checking for GPU vendor...")
        
        try:
            gpu_info = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
            
            if "NVIDIA" in gpu_info:
                print("NVIDIA GPU detected. Installing PyTorch with CUDA...")
                run_command(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
                
            elif "AMD" in gpu_info or "Radeon" in gpu_info:
                print("AMD GPU detected.")
                print("Note: Native ROCm is not fully supported on Windows yet. Installing DirectML/CPU fallback...")
                run_command(["torch", "torchvision", "torchaudio"])
            else:
                print("Installing default PyTorch...")
                run_command(["torch", "torchvision", "torchaudio"])
                
        except Exception:
            print("Could not detect GPU on Windows. Installing default PyTorch.")
            run_command(["torch", "torchvision", "torchaudio"])

def install_requirements():
    if os.path.exists("requirements.txt"):
        print("Installing remaining packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("requirements.txt not found. Skipping.")

if __name__ == "__main__":
    print("Starting Environment Setup...")
    
    current_os = platform.system()
    
    check_gpu_and_install_torch(current_os)
    
    install_requirements()
    
    print("Setup Complete.")