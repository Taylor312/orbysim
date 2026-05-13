import os
import sys
import numpy as np

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

# --- 2. CONFIGURATION (5090 GPU) ---
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

print(f" [INFO] Booting Interactive Torque-Controlled Sim...")
simulation_app = SimulationApp(launch_config)

# --- 3. IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, Gf, UsdShade
import omni.kit.commands
import omni.appwindow
import carb.input


# --- 4. START WORLD ---
world = World(backend="torch", device="cuda:0")

# --- 5. ENVIRONMENT ---
light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# --- 6. URDF TO USD CONVERSION ---
URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.xacro"
USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.usd"

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

# --- 7. LOAD ROBOT & ASSIGN MATERIAL ---
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/TankRover")
robot_prim = Articulation(prim_paths_expr="/World/TankRover", name="Rover")
world.scene.add(robot_prim)

# PROCEDURAL MATERIAL (Native USD Method)
omni.kit.commands.execute(
    "CreateMdlMaterialPrim",
    mtl_url="OmniPBR.mdl",
    mtl_name="OmniPBR",
    mtl_path="/World/Looks/OrangePaint"
)

stage = world.stage
mat_prim = stage.GetPrimAtPath("/World/Looks/OrangePaint")
material = UsdShade.Material(mat_prim)
rover_usd_prim = stage.GetPrimAtPath("/World/TankRover")
binding_api = UsdShade.MaterialBindingAPI.Apply(rover_usd_prim)
binding_api.Bind(material)

shader_prim = stage.GetPrimAtPath("/World/Looks/OrangePaint/Shader")
if shader_prim:
    shader = UsdShade.Shader(shader_prim)
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.35, 0.05))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.7)

world.reset()

# --- 8. TANK JOINT MAPPING & EFFORT SETUP ---
fl_idx = robot_prim.get_dof_index("Revolute_4")
bl_idx = robot_prim.get_dof_index("Revolute_3")
fr_idx = robot_prim.get_dof_index("Revolute_1")
br_idx = robot_prim.get_dof_index("Revolute_2")

num_dof = robot_prim.num_dof

# Stiffness (kP) = 0 for Torque control.
# Damping (kD) = 2.0 to simulate back-EMF and mechanical friction.
kps = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
kds = torch.full((robot_prim.count, num_dof), 5.0, device="cuda:0")
robot_prim.set_gains(kps=kps, kds=kds)

# --- 9. CARBONITE INPUT INTERFACE ---
# Corrected for 2026: Using acquire_input_interface() to read states
input_interface = carb.input.acquire_input_interface()
appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()

MAX_TORQUE = 0.3  # Increased slightly for weight
print(f" [SUCCESS] Keyboard Torque Control Active. Use Arrow Keys.")

# --- 10. MAIN SIMULATION LOOP ---
while simulation_app.is_running():
    
    # Poll the Keyboard State using the Input Interface
    up = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.UP)
    down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.DOWN)
    left = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT)
    right = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT)

    # Mixing Logic
    forward_axis = up - down
    turn_axis = right - left

    torque_left = (forward_axis + turn_axis) * MAX_TORQUE
    torque_right = (forward_axis - turn_axis) * MAX_TORQUE

    # Apply to GPU Buffer with Axis Flips
    efforts = torch.zeros((robot_prim.count, num_dof), device="cuda:0")
    
    # Left Side
    efforts[0, fl_idx] = torque_left
    efforts[0, bl_idx] = -torque_left
    
    # Right Side
    efforts[0, fr_idx] = torque_right
    efforts[0, br_idx] = -torque_right

    # Use set_joint_efforts for raw Torque mode
    robot_prim.set_joint_efforts(efforts)
    
    world.step(render=True)

simulation_app.close()