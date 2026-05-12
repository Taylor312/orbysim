import os
import sys
import torch
import numpy as np 

# --- 1. ANCHOR FIX ---
try:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
except Exception:
    pass

from isaacsim import SimulationApp

# --- 2. CONFIGURATION ---
launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    "renderer": "RayTracedLighting",
    "exts": ["omni.physx.fabric"], 
    "extra_args": [
        "--/renderer/activeGpu=0",
        "--/renderer/multiGpu/enabled=false",
        "--/physics/cudaDevice=0", 
        "--/rtx/divider/enabled=false",
        "--/rtx/optixDenoiser/enabled=false", 
        "--/rtx/denoiser/enabled=false"
    ]
}

print(f" [INFO] Booting 5090 GPU Pipeline...")
simulation_app = SimulationApp(launch_config)

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf
import omni.kit.commands

# --- 3. START WORLD ---
world = World(backend="torch", device="cuda")

# --- 4. SCENE SETUP ---
scene_path = "/physicsScene"
if not world.stage.GetPrimAtPath(scene_path):
    omni.kit.commands.execute("AddPhysicsScene", stage=world.stage, path=scene_path)

scene_prim = world.stage.GetPrimAtPath(scene_path)
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")

# Environment
light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# Load Robot USD
USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.usd"
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/finalrobotoneBot")

# --- 5. BAKE GPU ---
print(" [INFO] Baking GPU Buffers...")
world.reset()

# --- 6. REGISTER ROBOT ---
print(" [INFO] Registering Articulation...")
robot = world.scene.add(Articulation(prim_path="/World/finalrobotoneBot", name="Swervy"))
robot.set_enabled_self_collisions(False)

# Bind View
world.reset()
robot.set_world_pose(position=torch.tensor([0.0, 0.0, 0.5], device="cuda"))

# --- 7. WARMUP ---
print(" [INFO] Warming up Simulation...")
for _ in range(60):
    world.step(render=True)

# --- 8. DIRECT GPU CONTROL ---
drive_names = ["front_left_w", "front_right_w", "back_left_w", "back_right_w"]
steer_names = ["front_left_mod", "front_right_mod", "back_left_mod", "back_right_mod"]
DIRECTIONS = {
    "front_left_w": -1.0, "front_right_w": -1.0, "back_left_w": -1.0, "back_right_w": -1.0,
    "front_left_mod": 0, "front_right_mod": 0, "back_left_mod": 0, "back_right_mod": 0
}
drive_indices = [robot.get_dof_index(n) for n in drive_names]
steer_indices = [robot.get_dof_index(n) for n in steer_names]
num_dof = robot.num_dof

# FIX: Access the underlying Physics View directly
# This object speaks pure Tensor, bypassing the CPU checks
physics_view = robot._articulation_view

print(f" [INFO] Setting Gains via Physics View...")
physics_view.set_gains(
    kps=torch.zeros(num_dof, device="cuda"), 
    kds=torch.full((num_dof,), 1000.0, device="cuda")
)

print(f" [SUCCESS] Loop Starting...")

# --- 9. MAIN LOOP ---
while simulation_app.is_running():
    # 1. Create Tensor
    velocities = torch.zeros(num_dof, device="cuda")
    
    # 2. Fill Tensor
    for name, idx in zip(drive_names, drive_indices):
        velocities[idx] = 5.0 * DIRECTIONS[name]
    for name, idx in zip(steer_names, steer_indices):
        velocities[idx] = 1.0 * DIRECTIONS[name]
    
    # 3. Direct Tensor Command
    physics_view.set_joint_velocity_targets(velocities)
    
    world.step(render=True)

simulation_app.close()