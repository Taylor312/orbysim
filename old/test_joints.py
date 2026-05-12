import os
import sys
import numpy as np

# --- 1. ANCHOR FIX ---
try:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
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
    ]
}

print(f" [INFO] Booting Debug Sim...")
simulation_app = SimulationApp(launch_config)

from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdLux, Sdf 
import omni.kit.commands

# 3. Start World
world = World(backend="numpy", device="cpu") 
world.reset()

# 4. Environment
light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
URDF_PATH = "C:/Users/taylo/Documents/robotexports/supercut_description/urdf/supercut.xacro"
USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/supercut_description/urdf/supercut.usd"

if not os.path.exists(URDF_PATH):
    print(f" [ERROR] Missing XACRO: {URDF_PATH}")
    simulation_app.close()
    sys.exit()

# 5. REPAIR USD FILE
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.fix_base = True            
import_config.make_default_prim = True   
import_config.create_physics_scene = True
import_config.distance_scale = 1.0

omni.kit.commands.execute(
    "URDFParseAndImportFile", 
    urdf_path=URDF_PATH, 
    import_config=import_config, 
    dest_path=USD_OUTPUT_PATH 
)

# 6. Load Robot
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/supercutBot")
robot = Articulation(prim_path="/World/supercutBot", name="Swervy")
world.scene.add(robot)
robot.set_enabled_self_collisions(False)

world.reset()
robot.set_world_pose(position=np.array([0, 0, 1.0])) 

# --- LOCKING LOGIC ---
num_dof = robot.num_dof
# High stiffness to freeze the robot in place
kps = np.full(num_dof, 1000000.0) 
kds = np.full(num_dof, 1000.0)

controller = robot.get_articulation_controller()
controller.set_gains(kps=kps, kds=kds)

print(f" [INFO] Sim Running in FREEZE MODE. Joints locked to 0.0")

while simulation_app.is_running():
    # Send strictly ZEROS to hold the URDF zero-pose
    action = ArticulationAction(
        joint_positions=np.zeros(num_dof),
        joint_velocities=np.zeros(num_dof),
        joint_indices=np.array(range(num_dof))
    )
    controller.apply_action(action)
    
    world.step(render=True)

simulation_app.close()