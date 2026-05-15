import os
import sys
import numpy as np
#It works!
# --- 1. DLL ANCHOR (Taylor's 5090 Rig) ---
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

print(f" [INFO] Orbitron Architecture: Version 2026.1.6 (Recursion Fix)")
simulation_app = SimulationApp(launch_config)

# --- 3. STABLE IMPORTS ---
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import Articulation 
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade
import omni.kit.commands

# --- 4. START WORLD & TGS SOLVER ---
world = World(backend="torch", device="cuda:0")
stage = world.stage

scene_prim = stage.GetPrimAtPath("/physicsScene")
if not scene_prim:
    scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()
PhysxSchema.PhysxSceneAPI.Apply(scene_prim).CreateSolverTypeAttr("TGS")

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

# --- 7. DYNAMIC PHYSICS INJECTION (Recursion-Proof) ---
def setup_orbitron_physics(root_path):
    mat_path = "/World/Physics_Materials/Rubber"
    stage.DefinePrim("/World/Physics_Materials", "Scope")
    UsdShade.Material.Define(stage, mat_path)
    rubber_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    rubber_mat.CreateStaticFrictionAttr(1.5)
    rubber_mat.CreateRestitutionAttr(0.0)

    # Apply root Articulation
    root_prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(root_prim).CreateEnabledSelfCollisionsAttr(False)

    for prim in stage.Traverse():
        p_path = str(prim.GetPath())
        if not p_path.startswith(root_path): continue
        
        # GATEKEEPER: If we are already looking at a sphere we created, SKIP IT.
        if "physics_sphere" in p_path:
            continue

        # A. CHASSIS RIGID BODY
        if "base_link" in p_path and prim.IsA(UsdGeom.Xformable):
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(25.0)
            UsdPhysics.RigidBodyAPI.Apply(prim)

        # B. WHEEL SURGERY
        if ("left" in p_path or "right" in p_path) and prim.IsA(UsdGeom.Xformable):
            # Skip non-link folders
            if "visuals" in p_path or "collisions" in p_path or "joints" in p_path:
                continue
            
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

            print(f" [WHEEL] Injecting Sphere Collider: {p_path}")
            
            # Disable visual meshes
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(False)

            # Author the Sphere Collider (Safe from recursion now)
            sphere_path = f"{p_path}/physics_sphere"
            sphere_geom = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere_geom.CreateRadiusAttr(0.05)
            
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(sphere_path))
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(sphere_path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
            )

            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(1.5)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

        # C. JOINT DRIVE
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(1e6)

setup_orbitron_physics(robot_path)

# --- 8. INITIALIZE ---
orbitron = Articulation(prim_paths_expr=robot_path, name="Orbitron")
world.scene.add(orbitron)
world.reset()

print(f" [SUCCESS] Orbitron Active. Drivetrain Bench-test engaged.")

# --- 9. MAIN LOOP ---
while simulation_app.is_running():
    num_dof = orbitron.num_dof
    if num_dof:
        action = torch.zeros(num_dof, device="cuda:0")
        if num_dof >= 4:
            action[0:4] = torch.tensor([-10.0,      10.0, 
                                        10.0,      -10.0], device="cuda:0")
        
        orbitron.set_joint_velocity_targets(action)

    world.step(render=True)

simulation_app.close()