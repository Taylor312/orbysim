import os

import sys

import numpy as np

import time



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



print(f" [INFO] Booting Debug Sim on 5090...")

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

URDF_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.xacro"

USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.usd"



# 5. REPAIR USD FILE

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



# 6. Load Robot

add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/finalrobotoneBot")

robot = world.scene.add(Articulation(prim_path="/World/finalrobotoneBot", name="Swervy"))

robot.set_enabled_self_collisions(False)



world.reset()

robot.set_world_pose(position=np.array([0, 0, 0.5]))



# --- 7. JOINT MAPPING ---

# We map your Fusion names to indices so we don't have to guess

# Drive Wheels (The 'w' joints)

drive_names = ["front_left_w", "front_right_w", "back_left_w", "back_right_w"]

# Steer Modules (The 'mod' joints)

steer_names = ["front_left_mod", "front_right_mod", "back_left_mod", "back_right_mod"]



# Dictionary to handle motor inversions (Trial and Error here!)

# If a wheel spins backwards, change its 1.0 to -1.0

DIRECTIONS = {

    "front_left_w": -1.0,

    "front_right_w": -1.0,

    "back_left_w": -1.0,

    "back_right_w": -1.0,

    "front_left_mod": 0,

    "front_right_mod": 0,

    "back_left_mod": 0,

    "back_right_mod": 0

}



# Convert names to actual Sim Indices

drive_indices = [robot.get_dof_index(n) for n in drive_names]

steer_indices = [robot.get_dof_index(n) for n in steer_names]



# --- VELOCITY CONTROL SETUP ---

num_dof = robot.num_dof

kps = np.zeros(num_dof)

kds = np.full(num_dof, 1000.0)



controller = robot.get_articulation_controller()

controller.set_gains(kps=kps, kds=kds)



print(f" [INFO] Starting Named Control. Drive indices: {drive_indices}")



while simulation_app.is_running():

    # Initialize all velocities to zero

    velocities = np.zeros(num_dof)

   

    # Command all drive wheels to spin at 5.0 rad/s

    for name, idx in zip(drive_names, drive_indices):

        velocities[idx] = 5.0 * DIRECTIONS[name]

       

    # Command all steer modules to spin slowly at 1.0 rad/s

    for name, idx in zip(steer_names, steer_indices):

        velocities[idx] = 1.0 * DIRECTIONS[name]

   

    action = ArticulationAction(

        joint_velocities=velocities,

        joint_indices=np.array(range(num_dof))

    )

    controller.apply_action(action)

   

    world.step(render=True)



simulation_app.close()