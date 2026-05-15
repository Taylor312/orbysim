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

# --- 2. CONFIGURATION (5090 Optimized) ---
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

print(f" [INFO] Taylor Anthony's 5090 Unified Drive Test booting...")
simulation_app = SimulationApp(launch_config)

# --- 3. IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, Gf, UsdShade, UsdPhysics, PhysxSchema
import omni.kit.commands

# --- 4. START WORLD ---
world = World(backend="torch", device="cuda:0")

# --- 5. ENVIRONMENT (Reverted to standard floor) ---
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

# --- 8. THE VEHICLE SDK SURGERY ---

# A. VIOLENTLY REMOVE ARTICULATION ROOT
robot_root = stage.GetPrimAtPath("/World/TankRover")
if robot_root.HasAPI(UsdPhysics.ArticulationRootAPI):
    robot_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)

base_link = stage.GetPrimAtPath("/World/TankRover/base_link")
if base_link.HasAPI(UsdPhysics.ArticulationRootAPI):
    base_link.RemoveAPI(UsdPhysics.ArticulationRootAPI)

# B. WHEEL SURGERY
wheel_paths = [
    "/World/TankRover/frontleft_1",
    "/World/TankRover/frontright_1",
    "/World/TankRover/backleft_1",
    "/World/TankRover/backright_1"
]

for path in wheel_paths:
    wheel_prim = stage.GetPrimAtPath(path)
    if not wheel_prim: continue

    UsdPhysics.CollisionAPI.Apply(wheel_prim).CreateCollisionEnabledAttr(False)

    PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wheel_prim).CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
    
    wheel_api = PhysxSchema.PhysxVehicleWheelAPI.Apply(wheel_prim)
    wheel_api.CreateRadiusAttr(0.05) 
    wheel_api.CreateMassAttr(1.5) 
    wheel_api.CreateMoiAttr(0.01) 
    
    susp_api = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(wheel_prim)
    susp_api.CreateTravelDistanceAttr(0.05)   
    susp_api.CreateSpringStrengthAttr(1500.0) 
    susp_api.CreateSpringDamperRateAttr(40.0) 

    tire_api = PhysxSchema.PhysxVehicleTireAPI.Apply(wheel_prim)
    tire_api.CreateLatStiffXAttr(0.0001) 
    tire_api.CreateLongitudinalStiffnessAttr(2000.0) 

# C. MASTER RIGID BODY FIX
# 1. Strip the physics from the child link to prevent nested conflicts
if base_link.HasAPI(UsdPhysics.RigidBodyAPI):
    base_link.RemoveAPI(UsdPhysics.RigidBodyAPI)
if base_link.HasAPI(UsdPhysics.MassAPI):
    base_link.RemoveAPI(UsdPhysics.MassAPI)

# 2. Elevate the RigidBody and Mass to the exact same prim as the VehicleAPI
PhysxSchema.PhysxRigidBodyAPI.Apply(robot_root).CreateSleepThresholdAttr(0.0)
mass_api = UsdPhysics.MassAPI.Apply(robot_root)
mass_api.CreateMassAttr(25.0)
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -0.05))

# D. UNIFIED DRIVE APPLICATION (Root Logic)
PhysxSchema.PhysxVehicleAPI.Apply(robot_root)

drive_api = PhysxSchema.PhysxVehicleDriveBasicAPI.Apply(robot_root)
drive_api.CreatePeakTorqueAttr(1000.0) 

diff_api = PhysxSchema.PhysxVehicleMultiWheelDifferentialAPI.Apply(robot_root)
diff_api.CreateWheelsAttr([0, 1, 2, 3])
diff_api.CreateTorqueRatiosAttr([1.0, 1.0, 1.0, 1.0])

controller_api = PhysxSchema.PhysxVehicleControllerAPI.Apply(robot_root)

# --- 9. INITIALIZE PHYSICS ---
world.reset()

print(f" [SUCCESS] 2026.1 Unified Drive Active. Automatic throttle engaged.")

# --- 10. MAIN SIMULATION LOOP ---
while simulation_app.is_running():
    # Constant throttle bench-test
    accel = 0.5 
    steer = 0.0

    controller_api.GetAcceleratorAttr().Set(accel)
    controller_api.GetSteerAttr().Set(steer)

    world.step(render=True)

simulation_app.close()