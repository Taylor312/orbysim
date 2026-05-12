import os
import sys
import numpy as np
import time

# --- 1. DLL ANCHOR ---
try:
    v_root = os.path.dirname(os.path.dirname(sys.executable))
    t_lib = os.path.join(v_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(t_lib):
        os.add_dll_directory(t_lib)
    import torch
except Exception:
    pass

from isaacsim import SimulationApp

# --- 2. CONFIGURATION ---
target_uuid = "GPU-6edeed59-5bbf-c940-8e42-5830baef4d84"
launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    "renderer": "RayTracedLighting",
    "display_options": 3094,
    "extra_args": [
        f"--/renderer/activeGpu={target_uuid}",
        "--/renderer/multiGpu/enabled=false",
        "--/physics/cudaDevice=0", 
    ]
}

print(f" [INFO] Booting 5090 GPU-Accelerated Swerve Sim...")
simulation_app = SimulationApp(launch_config)

# --- 3. UPDATED IMPORTS (2026 CONSOLIDATED API) ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
# NOTE: ArticulationView is now just Articulation in the prims namespace
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf
import omni.kit.commands

# --- 4. START WORLD ---
world = World(backend="torch", device="cuda:0")

# --- 5. ENVIRONMENT ---
light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# --- 6. URDF TO USD CONVERSION ---
URDF_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.xacro"
USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.usd"

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = True 
import_config.fix_base = False
import_config.make_default_prim = True  
import_config.create_physics_scene = True
import_config.distance_scale = 1.0

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    dest_path=USD_OUTPUT_PATH
)

# --- 7. LOAD ROBOT VIA VECTORIZED CLASS ---
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/finalrobotoneBot")

# In 2026, 'Articulation' is the vectorized class that replaces ArticulationView
robot_prim = Articulation(prim_paths_expr="/World/finalrobotoneBot", name="SwerveRobot")
world.scene.add(robot_prim)

world.reset()

# --- 8. JOINT MAPPING ---
drive_names = ["back_right_w", "back_left_w", "front_left_w", "front_right_w"]
steer_names = ["back_right_mod", "back_left_mod", "front_right_mod", "front_left_mod"]

drive_pol = torch.tensor([-1.0, -1.0, -1.0, -1.0], device="cuda:0")

# Indices are now relative to the Articulation prim
drive_indices = [robot_prim.get_dof_index(n) for n in drive_names]
steer_indices = [robot_prim.get_dof_index(n) for n in steer_names]

# Initialize Gains directly on GPU
num_dof = robot_prim.num_dof
kps = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
kds = torch.full((robot_prim.count, num_dof), 1000.0, device="cuda:0")

robot_prim.set_gains(kps=kps, kds=kds)

print(f" [INFO] GPU-Tensor Control Active. Drive indices: {drive_indices}")

while simulation_app.is_running():
    # 9. Create Command Tensor
    velocities = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
    
    # Apply Drive (5.0 rad/s)
    velocities[0, drive_indices] = 5.0 * drive_pol
    # Apply Steer (1.0 rad/s)
    velocities[0, steer_indices] = 1.0 

    # Push to GPU buffer
    robot_prim.set_joint_velocities(velocities)
    
    world.step(render=True)

simulation_app.close()