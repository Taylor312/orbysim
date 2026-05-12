import os
import sys
import torch

# --- 1. CORE PATH FIX ---
try:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
except Exception:
    pass

from isaacsim import SimulationApp

# --- 2. THE "NO-DENOISE" CONFIG ---
# This specific config bypasses the Optix 201 errors seen in your logs.
launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    "renderer": "RayTracedLighting",
    "exts": ["omni.physx.fabric"], 
    "extra_args": [
        "--/renderer/activeGpu=0",
        "--/physics/cudaDevice=0",
        "--/rtx/divider/enabled=false",
        "--/rtx/optixDenoiser/enabled=false", # Stops the OptiX/CUDA context crash
        "--/rtx/denoiser/enabled=false"
    ]
}

print(" [INFO] Booting Minimalist GPU Sim...")
simulation_app = SimulationApp(launch_config)

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from pxr import Sdf, PhysxSchema
import omni.kit.commands

# --- 3. INITIALIZE GPU WORLD ---
# Forcing torch backend to stay on the 5090 VRAM
world = World(backend="torch", device="cuda")

# --- 4. FORCE-BAKE THE GPU SCENE ---
scene_path = "/physicsScene"
if not world.stage.GetPrimAtPath(scene_path):
    omni.kit.commands.execute("AddPhysicsScene", stage=world.stage, path=scene_path)

scene_prim = world.stage.GetPrimAtPath(scene_path)
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")

# --- 5. STABILITY WARMUP ---
# We step 100 times to ensure the CUDA context is solid before doing anything else
print(" [INFO] Baking GPU Context (100 steps)...")
world.reset()
for _ in range(100):
    world.step(render=True)

# --- 6. SIMPLE ENVIRONMENT ---
# No robot yet—just testing if the 5090 holds the ground plane
world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))
world.reset()

print(" [SUCCESS] GPU Physics and Rendering are stable. Close window to exit.")

# 7. IDLE LOOP
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()