import os
import sys
import numpy as np
import multiprocessing
import time
import torch
import pygame

# --- 1. DLL ANCHOR ---
try:
    v_root = os.path.dirname(os.path.dirname(sys.executable))
    t_lib = os.path.join(v_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(t_lib):
        os.add_dll_directory(t_lib)
except Exception:
    pass

# THE FIX: Added an extra os.path.dirname() to step out of the 'run/' folder 
# and target the root project directory so Python can find the 'utils/' package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import OrbitronConfig
from utils.telemetry import run_telemetry
from utils.motor_sim import Apex3Motor
from utils.battery_sim import OrbyBattery

def main(telemetry_queue, cmd_queue):
    # --- 2. CONFIGURATION ---
    from isaacsim import SimulationApp  
    
    launch_config = {
        "headless": False,
        "width": 1280, "height": 720,
        "renderer": "RayTracedLighting",
        "display_options": 3094,
        "extra_args": [
            f"--/renderer/activeGpu={OrbitronConfig.TARGET_GPU_UUID}",
            "--/renderer/multiGpu/enabled=false",
            "--/physics/cudaDevice=0", 
        ]
    }

    print(f" [INFO] Orbitron Architecture: 144Hz HITL Tuned Edition")
    simulation_app = SimulationApp(launch_config)

    # --- 3. IMPORTS ---
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import Articulation, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade
    import omni.kit.commands

    # --- 4. START WORLD & 144Hz TGS SOLVER ---
    # DECOUPLED CLOCKS: Physics at 144 Hz, Render at 60 Hz
    SIM_PHYSICS_DT = 1.0 / 144.0
    SIM_RENDER_DT = 1.0 / 60.0
    
    world = World(backend="torch", device="cuda:0", physics_dt=SIM_PHYSICS_DT, rendering_dt=SIM_RENDER_DT)
    stage = world.stage

    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim.IsValid():
        scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateTimeStepsPerSecondAttr(144) 
    physx_scene.CreateMaxPositionIterationCountAttr(32)
    physx_scene.CreateMaxVelocityIterationCountAttr(16)

    # --- 5. ENVIRONMENT ---
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
    light.CreateIntensityAttr(1000)
    ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

    if os.path.exists(OrbitronConfig.USD_OUTPUT_PATH):
        os.remove(OrbitronConfig.USD_OUTPUT_PATH)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False 
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = True

    omni.kit.commands.execute(
        "URDFParseAndImportFile", 
        urdf_path=OrbitronConfig.URDF_PATH, 
        import_config=import_config, 
        dest_path=OrbitronConfig.USD_OUTPUT_PATH
    )

    add_reference_to_stage(usd_path=OrbitronConfig.USD_OUTPUT_PATH, prim_path=OrbitronConfig.ROBOT_PRIM_PATH)

    # --- 6. PHYSICS INJECTION ---
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
                if prim.HasAPI(UsdPhysics.RigidBodyAPI): prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                continue

            if p_path.endswith("base_link") and prim.IsA(UsdGeom.Xformable):
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(25.0)
                UsdPhysics.RigidBodyAPI.Apply(prim)

            if ("left" in p_path or "right" in p_path) and prim.IsA(UsdGeom.Xformable):
                if "joints" in p_path: continue
                if prim.IsInstanceable(): prim.SetInstanceable(False)
                
                # Turn off the collision AND the visibility of the original CAD wheels
                for child in prim.GetChildren():
                    if child.IsA(UsdGeom.Mesh): 
                        UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(False)
                        UsdGeom.Imageable(child).MakeInvisible()

                cylinder_path = f"{p_path}/physics_cylinder"
                cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
                cylinder_geom.CreateRadiusAttr(0.03)
                cylinder_geom.CreateHeightAttr(0.02)
                cylinder_geom.CreateAxisAttr("X") 
                
                direction_multiplier = 1.0 if "left" in p_path else -1.0
                UsdGeom.XformCommonAPI(cylinder_geom).SetTranslate(Gf.Vec3d(0.015 * direction_multiplier, 0.0, 0.0))
                
                cylinder_geom.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
                cylinder_geom.CreateDisplayOpacityAttr([0.5]) 
                
                UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cylinder_path))
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(cylinder_path)).Bind(
                    UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
                )

                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr(1.5)
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                
                # Lower virtual inertia by 10x
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.005, 0.005, 0.005))

            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(0.0)
                # Lower mechanical gearbox drag to allow rapid spool-up
                drive_api.CreateDampingAttr(0.2) 

    setup_orbitron_physics(OrbitronConfig.ROBOT_PRIM_PATH)

    # --- 7. HARDWARE INIT ---
    orbitron = Articulation(prim_paths_expr=OrbitronConfig.ROBOT_PRIM_PATH, name="Orbitron")
    world.scene.add(orbitron)

    chassis = RigidPrim(prim_paths_expr=f"{OrbitronConfig.ROBOT_PRIM_PATH}/base_link", name="orbitron_chassis")
    world.scene.add(chassis)

    world.reset()

    dof_names = orbitron.dof_names
    def get_dof_idx(keywords):
        for i, name in enumerate(dof_names):
            if all(k in name.lower() for k in keywords): return i
        return 0

    fl_idx = get_dof_idx(["revolute_4"])
    fr_idx = get_dof_idx(["revolute_1"])
    bl_idx = get_dof_idx(["revolute_3"])
    br_idx = get_dof_idx(["revolute_2"])

    pygame.init()
    pygame.joystick.init()
    joystick = None
    use_gamepad = False

    for i in range(pygame.joystick.get_count()):
        temp_joy = pygame.joystick.Joystick(i)
        if any(kw in temp_joy.get_name().lower() for kw in ["xbox", "controller", "xinput", "gamepad"]):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            use_gamepad = True
            break

    battery = OrbyBattery()
    motor_fl = Apex3Motor()
    motor_fr = Apex3Motor()
    motor_bl = Apex3Motor()
    motor_br = Apex3Motor()

    # Max Trampa VESC limit (250A) for violent acceleration
    MULE_CURRENT_LIMIT = 250.0 
    motor_fl.current_limit = MULE_CURRENT_LIMIT
    motor_fr.current_limit = MULE_CURRENT_LIMIT
    motor_bl.current_limit = MULE_CURRENT_LIMIT
    motor_br.current_limit = MULE_CURRENT_LIMIT

    GEAR_RATIO = 25.0 
    WHEEL_RPM_CAP = 1600.0 
    DOWNFORCE_N = -1500.0 

    frame_count = 0

    # --- 8. MAIN LOOP ---
    while simulation_app.is_running():
        frame_count += 1
        accel = 0.0
        steer = 0.0
        
        while not cmd_queue.empty():
            try:
                cmds = cmd_queue.get_nowait()
                if "max_rpm" in cmds:
                    WHEEL_RPM_CAP = float(cmds["max_rpm"])
            except:
                pass

        if use_gamepad:
            pygame.event.pump()
            accel = -joystick.get_axis(1) 
            steer = joystick.get_axis(2) 
            if abs(accel) < 0.1: accel = 0.0
            if abs(steer) < 0.1: steer = 0.0

        sim_time = frame_count * SIM_PHYSICS_DT
        ramp_factor = min(1.0, sim_time / 1.5)
        current_downforce_n = DOWNFORCE_N * ramp_factor

        left_mix = accel + steer
        right_mix = accel - steer
        max_mag = max(1.0, abs(left_mix), abs(right_mix))
        
        t_motor_rpm_fl = (left_mix / max_mag) * WHEEL_RPM_CAP * GEAR_RATIO
        t_motor_rpm_bl = (left_mix / max_mag) * WHEEL_RPM_CAP * GEAR_RATIO
        t_motor_rpm_fr = (right_mix / max_mag) * WHEEL_RPM_CAP * GEAR_RATIO
        t_motor_rpm_br = (right_mix / max_mag) * WHEEL_RPM_CAP * GEAR_RATIO

        if orbitron.num_dof >= 4:
            current_rads = orbitron.get_joint_velocities()[0]
            
            w_rpm_fl = current_rads[fl_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_FL
            w_rpm_fr = current_rads[fr_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_FR
            w_rpm_bl = current_rads[bl_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_BL
            w_rpm_br = current_rads[br_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_BR

            c_motor_rpm_fl = w_rpm_fl * GEAR_RATIO
            c_motor_rpm_fr = w_rpm_fr * GEAR_RATIO
            c_motor_rpm_bl = w_rpm_bl * GEAR_RATIO
            c_motor_rpm_br = w_rpm_br * GEAR_RATIO

            dt = SIM_PHYSICS_DT
            
            m_torque_fl, _ = motor_fl.compute_torque(t_motor_rpm_fl, c_motor_rpm_fl, dt, battery)
            m_torque_fr, _ = motor_fr.compute_torque(t_motor_rpm_fr, c_motor_rpm_fr, dt, battery)
            m_torque_bl, _ = motor_bl.compute_torque(t_motor_rpm_bl, c_motor_rpm_bl, dt, battery)
            m_torque_br, _ = motor_br.compute_torque(t_motor_rpm_br, c_motor_rpm_br, dt, battery)

            def sanitize_torque(t):
                if t is None or np.isnan(t) or np.isinf(t): return 0.0
                return float(t)

            efforts = torch.zeros(orbitron.num_dof, device="cuda:0")
            efforts[fl_idx] = sanitize_torque(m_torque_fl) * GEAR_RATIO * 0.90 * OrbitronConfig.INV_FL
            efforts[fr_idx] = sanitize_torque(m_torque_fr) * GEAR_RATIO * 0.90 * OrbitronConfig.INV_FR
            efforts[bl_idx] = sanitize_torque(m_torque_bl) * GEAR_RATIO * 0.90 * OrbitronConfig.INV_BL
            efforts[br_idx] = sanitize_torque(m_torque_br) * GEAR_RATIO * 0.90 * OrbitronConfig.INV_BR
            
            efforts = torch.clamp(efforts, min=-200.0, max=200.0)
            orbitron.set_joint_efforts(efforts)

            downforce_tensor = torch.tensor([[0.0, 0.0, current_downforce_n]], dtype=torch.float32, device="cuda:0")
            chassis.apply_forces(forces=downforce_tensor, is_global=True)

            try:
                telemetry_queue.put_nowait({
                    "accel": accel, "steer": steer,
                    "t_rpm_fl": t_motor_rpm_fl / GEAR_RATIO, 
                    "c_rpm_fl": c_motor_rpm_fl / GEAR_RATIO,
                    "f_thrust": 0.0, "t_yaw": 0.0
                })
            except:
                pass 

        world.step(render=True)

    simulation_app.close()

if __name__ == '__main__':
    tele_queue = multiprocessing.Queue()
    cmd_queue = multiprocessing.Queue()
    tele_process = multiprocessing.Process(target=run_telemetry, args=(tele_queue, cmd_queue))
    tele_process.start()

    try:
        main(tele_queue, cmd_queue)
    finally:
        tele_process.terminate()