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

print(f" [INFO] Orbitron Architecture: Version 2026.1.7 (Half-Scale Test Mule)")
simulation_app = SimulationApp(launch_config)

# --- 3. STABLE IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade
import omni.kit.commands
import carb.input
import omni.appwindow

# --- 4. START WORLD & TGS SOLVER ---
world = World(backend="torch", device="cuda:0")
stage = world.stage

scene_prim = stage.GetPrimAtPath("/physicsScene")
if not scene_prim:
    scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
physx_scene.CreateSolverTypeAttr("TGS")
physx_scene.CreateMaxPositionIterationCountAttr(16)
physx_scene.CreateMaxVelocityIterationCountAttr(4)

# --- 5. ENVIRONMENT ---
light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

# --- 6. URDF IMPORT ---
URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.xacro"
USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.usd"

if os.path.exists(USD_OUTPUT_PATH):
    os.remove(USD_OUTPUT_PATH)

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False 
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.make_default_prim = True

omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=URDF_PATH, import_config=import_config, dest_path=USD_OUTPUT_PATH)

robot_path = "/World/TankRover"
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path=robot_path)

# --- 7. DYNAMIC PHYSICS INJECTION ---
def setup_orbitron_physics(root_path):
    mat_path = "/World/Physics_Materials/Rubber"
    stage.DefinePrim("/World/Physics_Materials", "Scope")
    UsdShade.Material.Define(stage, mat_path)
    
    rubber_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    rubber_mat.CreateStaticFrictionAttr(1.5)
    rubber_mat.CreateDynamicFrictionAttr(1.2)
    rubber_mat.CreateRestitutionAttr(0.0)

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    root_prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(root_prim).CreateEnabledSelfCollisionsAttr(False)

    for prim in stage.Traverse():
        p_path = str(prim.GetPath())
        if not p_path.startswith(root_path): continue
        if "physics_cylinder" in p_path: continue

        if "visuals" in p_path or "collisions" in p_path:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            continue

        if p_path.endswith("base_link") and prim.IsA(UsdGeom.Xformable):
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(25.0)
            UsdPhysics.RigidBodyAPI.Apply(prim)

        # CYLINDER WHEEL SURGERY
        if ("left" in p_path or "right" in p_path) and prim.IsA(UsdGeom.Xformable):
            if "joints" in p_path: continue
            
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(False)

            cylinder_path = f"{p_path}/physics_cylinder"
            cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
            
            cylinder_geom.CreateRadiusAttr(0.03)
            cylinder_geom.CreateHeightAttr(0.02)
            
            # --- FIX 1: AXIS ALIGNMENT ---
            AXLE_AXIS = "X" 
            cylinder_geom.CreateAxisAttr(AXLE_AXIS) 
            
            # --- FIX 2: CONCENTRIC OFFSET ---
            OFFSET_DISTANCE = 0.015 
            
            direction_multiplier = 1.0 if "left" in p_path else -1.0
            
            transform_api = UsdGeom.XformCommonAPI(cylinder_geom)
            
            # CRITICAL FIX: Changed Gf.Vec3f to Gf.Vec3d to match C++ signature
            if AXLE_AXIS == "X":
                transform_api.SetTranslate(Gf.Vec3d(OFFSET_DISTANCE * direction_multiplier, 0.0, 0.0))
            elif AXLE_AXIS == "Y":
                transform_api.SetTranslate(Gf.Vec3d(0.0, OFFSET_DISTANCE * direction_multiplier, 0.0))
            elif AXLE_AXIS == "Z":
                transform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, OFFSET_DISTANCE * direction_multiplier))

            # Maintain the debug visualization
            #cylinder_geom.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
            #cylinder_geom.CreateDisplayOpacityAttr([0.6])
            cylinder_geom.CreateDisplayOpacityAttr([0.0])
            
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cylinder_path))
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(cylinder_path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
            )

            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(1.5)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(1000.0) 
            drive_api.CreateMaxForceAttr(15.0)

setup_orbitron_physics(robot_path)

# --- 8. INITIALIZE ---
orbitron = Articulation(prim_paths_expr=robot_path, name="Orbitron")
world.scene.add(orbitron)

# --- 8.1 EXTRACT CHASSIS VIEW FOR DOWNFORCE INJECTION ---
chassis_path = f"{robot_path}/base_link" 
chassis = RigidPrim(prim_paths_expr=chassis_path, name="orbitron_chassis")
world.scene.add(chassis)

world.reset()

dof_names = orbitron.dof_names
print(f" [INFO] Detected Joints: {dof_names}")

def get_dof_idx(keywords):
    for i, name in enumerate(dof_names):
        if all(k in name.lower() for k in keywords):
            return i
    return 0

fl_idx = get_dof_idx(["front", "left"])
fr_idx = get_dof_idx(["front", "right"])
bl_idx = get_dof_idx(["back", "left"])
br_idx = get_dof_idx(["back", "right"])

# --- 9. KEYBOARD SETUP ---
input_interface = carb.input.acquire_input_interface()
appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()

print(f" [SUCCESS] Test Mule Ready. Drive with Arrow Keys.")

# --- 10. MAIN LOOP ---
MAX_SPEED = 30.0 

IDX_FR = 0 
IDX_BR = 1 
IDX_BL = 2 
IDX_FL = 3 

INV_FR = 1.0
INV_BR = -1.0 
INV_BL = -1.0 
INV_FL = 1.0  

DOWNFORCE_N = -1500.0 

while simulation_app.is_running():
    up = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.UP)
    down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.DOWN)
    left = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT)
    right = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT)

    accel = float(up - down)       
    steer = float((right - left))    

    left_mix = accel + steer
    right_mix = accel - steer

    max_mag = max(1.0, abs(left_mix), abs(right_mix))
    
    cmd_fl = (left_mix / max_mag) * MAX_SPEED * INV_FL
    cmd_bl = (left_mix / max_mag) * MAX_SPEED * INV_BL
    cmd_fr = (right_mix / max_mag) * MAX_SPEED * INV_FR
    cmd_br = (right_mix / max_mag) * MAX_SPEED * INV_BR

    num_dof = orbitron.num_dof
    if num_dof >= 4:
        action = torch.zeros(num_dof, device="cuda:0")
        action[IDX_FR] = cmd_fr
        action[IDX_BR] = cmd_br
        action[IDX_BL] = cmd_bl
        action[IDX_FL] = cmd_fl
            
        orbitron.set_joint_velocity_targets(action)

    downforce_tensor = torch.tensor([[0.0, 0.0, DOWNFORCE_N]], dtype=torch.float32, device="cuda:0")
    
    chassis.apply_forces(
        forces=downforce_tensor, 
        is_global=False 
    )

    world.step(render=True)

simulation_app.close()