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

# Step back to the root directory to access utils
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

    print(f" [INFO] Orbitron Architecture: Refactored & Stabilized HITL")
    simulation_app = SimulationApp(launch_config)

    # --- 3. IMPORTS ---
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import Articulation, RigidPrim
    from pxr import Usd, UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade
    import omni.kit.commands
    from isaacsim.core.utils.stage import add_reference_to_stage

    # --- 4. START WORLD & TGS SOLVER ---
    world = World(
        backend="torch", 
        device="cuda:0", 
        physics_dt=OrbitronConfig.PHYSICS_DT, 
        rendering_dt=OrbitronConfig.RENDER_DT
    )
    stage = world.stage

    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim.IsValid():
        scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateTimeStepsPerSecondAttr(OrbitronConfig.PHYSICS_HZ) 
    physx_scene.CreateMaxPositionIterationCountAttr(32)
    physx_scene.CreateMaxVelocityIterationCountAttr(16)

    # --- 5. ENVIRONMENT ---
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
    light.CreateIntensityAttr(1500)
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
        rubber_mat.CreateStaticFrictionAttr(OrbitronConfig.STATIC_FRICTION)
        rubber_mat.CreateDynamicFrictionAttr(OrbitronConfig.DYNAMIC_FRICTION)
        rubber_mat.CreateRestitutionAttr(OrbitronConfig.RESTITUTION)

        root_prim = stage.GetPrimAtPath(root_path)

        # ---------------------------------------------------------
        # PHASE 1: THE BRUTE FORCE HIERARCHY CLEANER
        # Scans the entire URDF tree and nukes all auto-generated 
        # convex hulls and accidental nested RigidBodies.
        # ---------------------------------------------------------
        for desc in Usd.PrimRange(root_prim):
            p_path = str(desc.GetPath())
            
            # Remove rogue Articulation Roots
            if p_path != root_path and desc.HasAPI(UsdPhysics.ArticulationRootAPI):
                desc.RemoveAPI(UsdPhysics.ArticulationRootAPI)

            # Strip Nested RigidBodies (Fixes the missing xformstack errors)
            if ("visuals" in p_path or "collisions" in p_path) and desc.HasAPI(UsdPhysics.RigidBodyAPI):
                desc.RemoveAPI(UsdPhysics.RigidBodyAPI)

            # Annihilate auto-generated collisions on Visual Meshes (Fixes the floating bot)
            if desc.IsA(UsdGeom.Mesh) and "visuals" in p_path:
                UsdPhysics.CollisionAPI.Apply(desc).CreateCollisionEnabledAttr(False)

        # Apply true Articulation Root
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(root_prim).CreateEnabledSelfCollisionsAttr(False)

        # ---------------------------------------------------------
        # PHASE 2: EXPLICIT PHYSICS OVERRIDES & COLORIZATION
        # ---------------------------------------------------------
        links_to_parse = ["base_link", "front_spinner_1", "back_spinner_1", "FL_1", "FR_1", "BL_1", "BR_1"]

        for prim in Usd.PrimRange(root_prim):
            p_path = str(prim.GetPath())
            prim_name = prim.GetName()
            
            if prim_name not in links_to_parse: continue
            if "visuals" in p_path or "collisions" in p_path: continue
            if prim.IsInstanceable(): prim.SetInstanceable(False)

            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)

            # 1. CHASSIS
            if prim_name == "base_link":
                mass_api.CreateMassAttr(OrbitronConfig.CHASSIS_MASS_KG)
                # THE FIX: Lock the Center of Mass to the defined coordinates
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*OrbitronConfig.CHASSIS_COM_XYZ))
                
                # Paint Chassis Steel Grey
                for desc in Usd.PrimRange(prim):
                    if desc.IsA(UsdGeom.Mesh) and "visuals" in str(desc.GetPath()):
                        UsdGeom.Mesh(desc).CreateDisplayColorAttr([(0.3, 0.3, 0.35)])

            # 2. WEAPONS
            elif "spinner" in prim_name:
                mass_api.CreateMassAttr(OrbitronConfig.WEAPON_MASS_KG)
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                # THE FIX: Apply the explicit heavy weapon inertia
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(
                    OrbitronConfig.WEAPON_INERTIA, 
                    OrbitronConfig.WEAPON_INERTIA, 
                    OrbitronConfig.WEAPON_INERTIA
                ))
                
                # Paint Spinners Bright Silver
                for desc in Usd.PrimRange(prim):
                    if desc.IsA(UsdGeom.Mesh) and "visuals" in str(desc.GetPath()):
                        UsdGeom.Mesh(desc).CreateDisplayColorAttr([(0.8, 0.8, 0.85)])

            # 3. WHEELS
            elif prim_name in ["FL_1", "FR_1", "BL_1", "BR_1"]:
                mass_api.CreateMassAttr(OrbitronConfig.WHEEL_MASS_KG)
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(
                    OrbitronConfig.WHEEL_INERTIA, 
                    OrbitronConfig.WHEEL_INERTIA, 
                    OrbitronConfig.WHEEL_INERTIA
                ))

                # CREATE MATHEMATICAL PHYSICS CYLINDER
                cylinder_path = f"{p_path}/physics_cylinder"
                cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
                cylinder_geom.CreateRadiusAttr(OrbitronConfig.WHEEL_RADIUS_M)
                cylinder_geom.CreateHeightAttr(OrbitronConfig.WHEEL_WIDTH_M)
                cylinder_geom.CreateAxisAttr("X") 
                
                # THE FIX: Directly read the WHEEL_OFFSET_M from constants. 
                # (Since you set it to 0.0, the cylinders will spawn dead-center at the joint)
                direction_multiplier = -1.0 if "L_" in prim_name else 1.0
                UsdGeom.XformCommonAPI(cylinder_geom).SetTranslate(
                    Gf.Vec3d(OrbitronConfig.WHEEL_OFFSET_M * direction_multiplier, 0.0, 0.0)
                )
                
                # Show the green physics cylinders (semi-transparent) for debug
               # cylinder_geom.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
                #cylinder_geom.CreateDisplayOpacityAttr([0.5]) 
                UsdGeom.Imageable(cylinder_geom).MakeInvisible()
                
                UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cylinder_path))
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(cylinder_path)).Bind(
                    UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
                )

                # Paint Visual CAD Tires Matte Black
                for desc in Usd.PrimRange(prim):
                    if desc.IsA(UsdGeom.Mesh) and "visuals" in str(desc.GetPath()):
                        UsdGeom.Mesh(desc).CreateDisplayColorAttr([(0.05, 0.05, 0.05)])

        # ---------------------------------------------------------
        # PHASE 3: JOINT STABILIZATION
        # ---------------------------------------------------------
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                p_path = str(prim.GetPath())
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(0.0)
                
                if "spin" in p_path:
                    drive_api.CreateDampingAttr(1e6) # Hard lock the weapon bars for stability
                else:
                    drive_api.CreateDampingAttr(OrbitronConfig.GEARBOX_DRAG_DAMPING) 

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

    fl_idx = get_dof_idx(["frontleft"])
    fr_idx = get_dof_idx(["frontright"])
    bl_idx = get_dof_idx(["backleft"])
    br_idx = get_dof_idx(["backright"])

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

    motor_fl.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT
    motor_fr.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT
    motor_bl.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT
    motor_br.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT

    WHEEL_RPM_CAP = OrbitronConfig.DEFAULT_RPM_CAP 
    frame_count = 0
    last_speed_mps = 0.0

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

        sim_time = frame_count * OrbitronConfig.PHYSICS_DT
        ramp_factor = min(1.0, sim_time / OrbitronConfig.DOWNFORCE_RAMP_TIME_S)
        current_downforce_n = OrbitronConfig.DOWNFORCE_N * ramp_factor

        left_mix = accel + steer
        right_mix = accel - steer
        max_mag = max(1.0, abs(left_mix), abs(right_mix))
        
        t_motor_rpm_fl = (left_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_bl = (left_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_fr = (right_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_br = (right_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO

        if orbitron.num_dof >= 4:
            current_rads = orbitron.get_joint_velocities()[0]
            
            w_rpm_fl = current_rads[fl_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_FL
            w_rpm_fr = current_rads[fr_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_FR
            w_rpm_bl = current_rads[bl_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_BL
            w_rpm_br = current_rads[br_idx].item() * (30.0 / np.pi) * OrbitronConfig.INV_BR

            c_motor_rpm_fl = w_rpm_fl * OrbitronConfig.GEAR_RATIO
            c_motor_rpm_fr = w_rpm_fr * OrbitronConfig.GEAR_RATIO
            c_motor_rpm_bl = w_rpm_bl * OrbitronConfig.GEAR_RATIO
            c_motor_rpm_br = w_rpm_br * OrbitronConfig.GEAR_RATIO
            
            m_torque_fl, _ = motor_fl.compute_torque(t_motor_rpm_fl, c_motor_rpm_fl, OrbitronConfig.PHYSICS_DT, battery)
            m_torque_fr, _ = motor_fr.compute_torque(t_motor_rpm_fr, c_motor_rpm_fr, OrbitronConfig.PHYSICS_DT, battery)
            m_torque_bl, _ = motor_bl.compute_torque(t_motor_rpm_bl, c_motor_rpm_bl, OrbitronConfig.PHYSICS_DT, battery)
            m_torque_br, _ = motor_br.compute_torque(t_motor_rpm_br, c_motor_rpm_br, OrbitronConfig.PHYSICS_DT, battery)

            def sanitize_torque(t):
                if t is None or np.isnan(t) or np.isinf(t): return 0.0
                return float(t)

            efforts = torch.zeros(orbitron.num_dof, device="cuda:0")
            efforts[fl_idx] = sanitize_torque(m_torque_fl) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_FL
            efforts[fr_idx] = sanitize_torque(m_torque_fr) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_FR
            efforts[bl_idx] = sanitize_torque(m_torque_bl) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_BL
            efforts[br_idx] = sanitize_torque(m_torque_br) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_BR
            
            efforts = torch.clamp(efforts, min=-200.0, max=200.0)
            orbitron.set_joint_efforts(efforts)

            downforce_tensor = torch.tensor([[0.0, 0.0, current_downforce_n]], dtype=torch.float32, device="cuda:0")
            chassis.apply_forces(forces=downforce_tensor, is_global=True)

            # --- KINEMATICS CALCULATIONS ---
            # 1. Velocity (MPH)
            # get_linear_velocities() returns a [X, Y, Z] tensor
            chassis_vel = chassis.get_linear_velocities()[0]
            # Speed magnitude in the X/Y plane
            speed_mps = torch.norm(chassis_vel[0:2]).item() 
            speed_mph = speed_mps * 2.23694
            
            # 2. Acceleration (G-Force)
            accel_mps2 = (speed_mps - last_speed_mps) / OrbitronConfig.PHYSICS_DT
            accel_g = accel_mps2 / 9.81
            last_speed_mps = speed_mps

            # 3. Slip Percentage
            slip_pct = -1.0 # Default to -1 (Off)
            if abs(steer) < 0.1: # Only calculate when moving perfectly straight
                # Theoretical speed = (Avg Wheel RPM * 2pi/60) * Radius
                avg_wheel_rpm = (abs(w_rpm_fl) + abs(w_rpm_fr) + abs(w_rpm_bl) + abs(w_rpm_br)) / 4.0
                theoretical_speed_mps = (avg_wheel_rpm * 2.0 * np.pi / 60.0) * OrbitronConfig.WHEEL_RADIUS_M
                
                # Prevent divide-by-zero when standing still
                if theoretical_speed_mps > 0.1: 
                    slip_pct = ((theoretical_speed_mps - speed_mps) / theoretical_speed_mps) * 100.0
                    slip_pct = max(0.0, min(100.0, slip_pct)) # Clamp between 0% and 100%

            # --- TELEMETRY PAYLOAD ---
            try:
                telemetry_queue.put_nowait({
                    "accel": accel, 
                    "steer": steer,
                    "t_wheel_rpm": t_motor_rpm_fl / OrbitronConfig.GEAR_RATIO, 
                    "c_wheel_rpm": c_motor_rpm_fl / OrbitronConfig.GEAR_RATIO,
                    "c_motor_rpm": c_motor_rpm_fl,
                    "t_wheel_fl": efforts[fl_idx].item(),
                    "t_wheel_fr": efforts[fr_idx].item(),
                    "t_wheel_bl": efforts[bl_idx].item(),
                    "t_wheel_br": efforts[br_idx].item(),
                    "t_motor_fl": m_torque_fl, # Original ESC torque before gearing
                    "speed_mph": speed_mph,
                    "accel_g": accel_g,
                    "slip_pct": slip_pct
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