import os
import sys

# --- ANCHOR FIX: FORCE-LOAD TORCH (DO NOT TOUCH) ---
try:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
    import torch
except Exception:
    pass
# ---------------------------------------------------

from isaacsim import SimulationApp

# --- CONFIGURATION ---
target_uuid = "GPU-6edeed59-5bbf-c940-8e42-5830baef4d84"
launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    "renderer": "RayTracedLighting",
    "display_options": 3094,
    "extra_args": [
        f"--/renderer/activeGpu={target_uuid}",
        f"--/physics/cudaDevice={target_uuid}", 
        "--/renderer/multiGpu/enabled=false",
    ]
}

print(f" [INFO] Booting Isaac Sim on GPU: {target_uuid}...")
simulation_app = SimulationApp(launch_config)

# --- IMPORTS ---
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.articulations import Articulation # <--- NEW IMPORT
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf 
import omni.kit.commands
import numpy as np

# 1. Start Physics Engine (backend="torch" runs physics on GPU via PyTorch)
world = World(backend="torch", device="cuda:0")

# 2. Add Lighting
light_prim = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/SkyLight"))
light_prim.CreateIntensityAttr(1000)

# 3. Add Ground
ground = world.scene.add(
    GroundPlane(
        prim_path="/World/Ground",
        z_position=0,
        color=np.array([0.5, 0.5, 0.5])
    )
)

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
URDF_PATH = "C:/Users/taylo/Documents/robotexports/testingbot_description/urdf/testingbot.xacro"
USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/testingbot_description/urdf/testingbot.usd"
# ---------------------------------------------------------

if not os.path.exists(URDF_PATH):
    print(f" [ERROR] Could not find file: {URDF_PATH}")
    simulation_app.close()
    exit()

# 4. Convert URDF to USD
# We only need to do this once, but doing it every time ensures updates apply
print(f" [INFO] Converting URDF to USD...")
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
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

# 5. Spawn and Register the Robot
print(f" [INFO] Spawning Robot...")
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/TestingBot")

# Wrap the prim as an Articulation so the physics engine tracks it
robot = Articulation(prim_path="/World/TestingBot", name="SwervyBot")
world.scene.add(robot) # <--- CRITICAL: Tells the World this object exists

# 6. RESET THE WORLD (This wakes up the physics for the new robot)
world.reset()

# 7. Position (Must happen AFTER reset, or reset will overwrite it)
robot.set_world_pose(position=np.array([0, 0, 0.5])) 

print(" [SUCCESS] Robot Dropping! Press Ctrl+C to exit.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()