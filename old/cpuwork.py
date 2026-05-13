import os
import sys

# --- 1. THE BLINDFOLD (Must be before Torch/IsaacSim imports) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"

import numpy as np
import time

try:
    import torch
    # Ensure torch doesn't try to initialize CUDA
    torch.cuda.is_available = lambda : False
except ImportError:
    pass

from isaacsim import SimulationApp

# --- 2. CONFIGURATION (TOTAL CUDA LOCKDOWN) ---
launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    # We switch to "RealTime" to avoid RTX-only lighting requirements
    "renderer": "RayTracedLighting", 
    "display_options": 3094,
    "extra_args": [
        "--/renderer/activeGpu=-1",      # No GPU rendering
        "--/physics/cudaDevice=-1",     # No GPU physics
        "--/omni.physx.cuda/enabled=false", # KILL the CUDA physics plugin
        "--/rtx/denoiser/enabled=false"
    ]
}

print(f" [INFO] Booting Isolated CPU Swerve Sim...")
simulation_app = SimulationApp(launch_config)

# --- 3. IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.kit.commands

# --- 4. START WORLD (NUMPY BACKEND) ---
# This is the most stable mode for CPU-only work
world = World(backend="numpy", device="cpu")

# --- 5. ENVIRONMENT ---
from pxr import UsdLux, Sdf
light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# --- 6. URDF TO USD ---
URDF_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.xacro"
USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.usd"

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = True 
import_config.fix_base = False
import_config.create_physics_scene = True
import_config.make_default_prim = True

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    dest_path=USD_OUTPUT_PATH
)

# --- 7. LOAD ROBOT ---
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/finalrobotoneBot")
world.reset()

robot_prim = Articulation(prim_paths_expr="/World/finalrobotoneBot", name="SwerveRobot")
world.scene.add(robot_prim)
world.reset()

# --- 8. JOINT MAPPING (NUMPY ARRAYS) ---
drive_names = ["back_right_w", "back_left_w", "front_left_w", "front_right_w"]
steer_names = ["back_right_mod", "back_left_mod", "front_right_mod", "front_left_mod"]

drive_pol = np.array([-1.0, -1.0, -1.0, -1.0])
drive_indices = [robot_prim.get_dof_index(n) for n in drive_names]
steer_indices = [robot_prim.get_dof_index(n) for n in steer_names]

num_dof = robot_prim.num_dof
kps = np.zeros((robot_prim.count, num_dof))
kds = np.full((robot_prim.count, num_dof), 400.0) # Lower damping for CPU stability

robot_prim.set_gains(kps=kps, kds=kds)

print(f" [SUCCESS] CPU-Safe Swerve Active.")

while simulation_app.is_running():
    # 9. Numpy Command Logic
    velocities = np.zeros((robot_prim.count, num_dof))
    
    velocities[0, drive_indices] = 5.0 * drive_pol
    velocities[0, steer_indices] = 1.0 

    robot_prim.set_joint_velocities(velocities)
    
    world.step(render=True)

simulation_app.close()