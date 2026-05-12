import os
import sys

# --- STEP 1: FORCE-LOAD THE RTX 5090 TORCH ---
# We must do this BEFORE Isaac Sim touches anything.
try:
    # 1a. Manually inject the DLL path so Windows finds c10.dll
    # We look 2 levels up from python.exe to find .venv root
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        print(f" [ANCHOR] DLL Path Added: {torch_lib}")
    
    # 1b. Import Torch to lock it in memory
    import torch
    print(f" [ANCHOR] Torch Version: {torch.__version__}")
    print(f" [ANCHOR] CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f" [ANCHOR] GPU Locked: {torch.cuda.get_device_name(0)}")
    else:
        print(" [ANCHOR] WARNING: GPU not found by Torch!")

except Exception as e:
    print(f" [FATAL] Torch Pre-load failed: {e}")
    # If this fails, Isaac Sim will almost certainly crash later.
    sys.exit(1)

# --- STEP 2: START ISAAC SIM ---
print(" [INFO] Booting Isaac Sim...")
from isaacsim import SimulationApp

# RTX 5090 Configuration
launch_config = {
    "headless": False,
    "width": 1280, 
    "height": 720,
    "renderer": "RayTracedLighting",
    "display_options": 3094,
    # Force the specific GPU UUID found in your logs
    "extra_args": [
        "--/renderer/activeGpu=GPU-6edeed59-5bbf-c940-8e42-5830baef4d84",
        "--/physics/cudaDevice=0", 
        "--/renderer/multiGpu/enabled=false",
    ]
}

simulation_app = SimulationApp(launch_config)

# --- STEP 3: SIMULATION LOOP ---
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
import numpy as np

# Create World on GPU
world = World(backend="torch", device="cuda:0")
world.reset()

# Add Floor
ground = world.scene.add(
    GroundPlane(
        prim_path="/World/Ground",
        z_position=0,
        color=np.array([0.5, 0.5, 0.5])
    )
)

print(" [SUCCESS] Simulation is running. Press Ctrl+C to stop.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()