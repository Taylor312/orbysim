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

# --- 2. CONFIGURATION (5090 GPU MODE) ---
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

print(f" [INFO] Booting 5090 GPU-Accelerated Tank Sim...")
simulation_app = SimulationApp(launch_config)

# --- 3. IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
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

# --- 6. URDF TO USD CONVERSION (FIXED PATHS) ---
# Hardcoded to match your exact directory structure without double-appending
URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.xacro"
USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.usd"

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = True 
import_config.fix_base = False
import_config.make_default_prim = True  
import_config.create_physics_scene = True
import_config.distance_scale = 1.0

print(f" [INFO] Converting {URDF_PATH} to USD...")
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    dest_path=USD_OUTPUT_PATH
)

# --- 7. LOAD ROBOT ---
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/TankRover")
robot_prim = Articulation(prim_paths_expr="/World/TankRover", name="Rover")
world.scene.add(robot_prim)
world.reset()

# --- DIAGNOSTIC: PRINT AVAILABLE JOINTS ---
# If the script crashes below this line, check this printout to see what your joints are actually named!
print(f" [DIAGNOSTIC] Available Robot Joints: {robot_prim.dof_names}")
# --- 8. TANK JOINT MAPPING ---
# Grab individual indices so we can control their polarities
fl_idx = robot_prim.get_dof_index("Revolute_4")
bl_idx = robot_prim.get_dof_index("Revolute_3")
fr_idx = robot_prim.get_dof_index("Revolute_1")
br_idx = robot_prim.get_dof_index("Revolute_2")

# Set Gains
num_dof = robot_prim.num_dof
kps = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
kds = torch.full((robot_prim.count, num_dof), 1000.0, device="cuda:0")
robot_prim.set_gains(kps=kps, kds=kds)

print(f" [SUCCESS] Tank Control Active. DOF Count: {num_dof}")

while simulation_app.is_running():
    # 9. Corrected Tank Logic
    velocities = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
    
    # --- LEFT SIDE ---
    velocities[0, fl_idx] = 20.0
    velocities[0, bl_idx] = -20.0  # Negative because CAD axis is -1.0
    
    # --- RIGHT SIDE ---
    # The right side generally needs to be the inverse of the left side to drive straight
    velocities[0, fr_idx] = 20.0 
    velocities[0, br_idx] = -20.0   # Positive because it's inverse AND the CAD axis is -1.0

    robot_prim.set_joint_velocities(velocities)
    world.step(render=True)

simulation_app.close()