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

print(f" [INFO] Booting 5090 Unified Vehicle SDK (2026.1 API)...")
simulation_app = SimulationApp(launch_config)

# --- 3. IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, Gf, UsdShade, UsdPhysics, PhysxSchema
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

# --- 7. LOAD ROBOT ---
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/TankRover")
robot_prim = Articulation(prim_paths_expr="/World/TankRover", name="Rover")
world.scene.add(robot_prim)

stage = world.stage

# --- 8. THE VEHICLE SDK SURGERY (2026.1 COMPLIANT) ---
wheel_paths = [
    "/World/TankRover/frontleft_1",
    "/World/TankRover/frontright_1",
    "/World/TankRover/backleft_1",
    "/World/TankRover/backright_1"
]

for path in wheel_paths:
    wheel_prim = stage.GetPrimAtPath(path)
    if not wheel_prim: continue

    # A. Disable standard 3D Collisions
    collision_api = UsdPhysics.CollisionAPI(wheel_prim)
    if collision_api:
        collision_api.CreateCollisionEnabledAttr(False)

    # B. Apply Wheel/Suspension/Tire APIs
    PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wheel_prim).CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
    PhysxSchema.PhysxVehicleWheelAPI.Apply(wheel_prim).CreateRadiusAttr(0.05) 
    
    susp_api = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(wheel_prim)
    susp_api.CreateTravelDistanceAttr(0.05)   
    susp_api.CreateSpringStrengthAttr(1200.0)
    susp_api.CreateSpringDamperRateAttr(25.0) 

    tire_api = PhysxSchema.PhysxVehicleTireAPI.Apply(wheel_prim)
    tire_api.CreateLatStiffXAttr(0.0001) 
    tire_api.CreateLongitudinalStiffnessAttr(1000.0)

# C. OVERRIDE MASS
chassis_prim = stage.GetPrimAtPath("/World/TankRover/base_link")
if chassis_prim:
    mass_api = UsdPhysics.MassAPI.Apply(chassis_prim)
    mass_api.CreateMassAttr(25.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -0.05))

# D. UNIFIED DRIVE & DIFFERENTIAL APPLICATION
PhysxSchema.PhysxVehicleAPI.Apply(chassis_prim)
drive_api = PhysxSchema.PhysxVehicleDriveBasicAPI.Apply(chassis_prim)
diff_api = PhysxSchema.PhysxVehicleMultiWheelDifferentialAPI.Apply(chassis_prim)
controller_api = PhysxSchema.PhysxVehicleControllerAPI.Apply(chassis_prim)

# --- 9. INITIALIZE PHYSICS ---
# CRITICAL: We reset the world BEFORE querying robot_prim.num_dof
world.reset()

# Now that the world is reset, num_dof will be valid (integer) instead of None
num_dof = robot_prim.num_dof
if num_dof is not None:
    robot_prim.set_gains(
        kps=torch.zeros((1, num_dof), device="cuda:0"), 
        kds=torch.zeros((1, num_dof), device="cuda:0")
    )

# --- 10. CARBONITE INPUT ---
input_interface = carb.input.acquire_input_interface()
appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()

print(f" [SUCCESS] 2026.1 Unified Vehicle Differential Active. Drive with Arrows.")

# --- 11. MAIN SIMULATION LOOP ---
while simulation_app.is_running():
    up = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.UP)
    down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.DOWN)
    left = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT)
    right = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT)

    accel = float(up - down)
    steer = float(right - left)

    if not controller_api.GetAcceleratorAttr():
        controller_api.CreateAcceleratorAttr(accel)
        controller_api.CreateSteerAttr(steer)
    else:
        controller_api.GetAcceleratorAttr().Set(accel)
        controller_api.GetSteerAttr().Set(steer)

    world.step(render=True)

simulation_app.close()