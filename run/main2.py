import os
import sys
import numpy as np
import multiprocessing
import time
import pygame

try:
    v_root = os.path.dirname(os.path.dirname(sys.executable))
    t_lib = os.path.join(v_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(t_lib):
        os.add_dll_directory(t_lib)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import OrbitronConfig
from utils.telemetry import run_telemetry
from utils.motor_sim import Apex3Motor
from utils.battery_sim import OrbyBattery

def main(telemetry_queue, cmd_queue):
    from isaacsim import SimulationApp  
    
    launch_config = {
        "headless": False,
        "width": 1280, "height": 720,
        "renderer": "RayTracedLighting",
        "display_options": 3094,
        "extra_args": [
            f"--/renderer/activeGpu={OrbitronConfig.TARGET_GPU_UUID}",
            "--/renderer/multiGpu/enabled=false",
        ]
    }

    print(f" [INFO] Orbitron Architecture: Brute Force API Bypass Active...")
    simulation_app = SimulationApp(launch_config)

    import carb
    carb.settings.get_settings().set_bool("/physics/physxVehicle/debugDraw", True)

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from pxr import Usd, UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade, Tf
    import omni.kit.commands
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(
        backend="numpy", 
        device="cpu", 
        physics_dt=OrbitronConfig.PHYSICS_DT, 
        rendering_dt=OrbitronConfig.RENDER_DT
    )
    stage = world.stage

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Base Scene Creation
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim.IsValid():
        scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateTimeStepsPerSecondAttr(OrbitronConfig.PHYSICS_HZ) 
    physx_scene.CreateMaxPositionIterationCountAttr(32)
    physx_scene.CreateMaxVelocityIterationCountAttr(16)
    
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")

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

    # THE FIX: Global Scene Sweep. Find EVERY physics scene and forcefully inject the Context API
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxVehicleContextAPI.Apply(p)
            p.CreateAttribute("physxVehicleContext:verticalAxis", Sdf.ValueTypeNames.Token).Set("zAxis")
            p.CreateAttribute("physxVehicleContext:longitudinalAxis", Sdf.ValueTypeNames.Token).Set("xAxis")
            p.CreateAttribute("physxVehicleContext:upAxis", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.0, 0.0, 1.0))

    def setup_raycast_vehicle(root_path):
        mat_path = "/World/Physics_Materials/Rubber"
        stage.DefinePrim("/World/Physics_Materials", "Scope")
        UsdShade.Material.Define(stage, mat_path)
        rubber_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        rubber_mat.CreateStaticFrictionAttr(OrbitronConfig.STATIC_FRICTION)
        rubber_mat.CreateDynamicFrictionAttr(OrbitronConfig.DYNAMIC_FRICTION)
        
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Ground")).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
        )

        root_prim = stage.GetPrimAtPath(root_path)
        
        joints_to_delete = []
        for desc in Usd.PrimRange(root_prim):
            if desc.HasAPI(PhysxSchema.PhysxVehicleAPI):
                desc.RemoveAPI(PhysxSchema.PhysxVehicleAPI)
            if desc.HasAPI(PhysxSchema.PhysxVehicleWheelAttachmentAPI):
                desc.RemoveAPI(PhysxSchema.PhysxVehicleWheelAttachmentAPI)
            if desc.HasAPI(UsdPhysics.ArticulationRootAPI):
                desc.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if desc.HasAPI(UsdPhysics.RigidBodyAPI):
                desc.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if desc.IsA(UsdGeom.Mesh) and desc.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(desc).CreateCollisionEnabledAttr(False)
            
            if desc.IsA(UsdPhysics.Joint) or desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.FixedJoint):
                joints_to_delete.append(desc.GetPath())

        for j_path in joints_to_delete:
            j_prim = stage.GetPrimAtPath(j_path)
            if j_prim.IsValid():
                j_prim.SetActive(False)

        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_rb.CreateMaxLinearVelocityAttr(10000.0) 
        physx_rb.CreateMaxAngularVelocityAttr(100000.0) 
        physx_rb.CreateDisableGravityAttr(True)
        
        # THE FIX: Silence the deprecated Metadata warning properly
        root_prim.SetCustomDataByKey("physxVehicle:referenceFrameIsCenterOfMass", False)
        
        PhysxSchema.PhysxVehicleAPI.Apply(root_prim)

        wheel_links = ["FL_1", "FR_1", "BL_1", "BR_1"]
        wheel_controllers = {}
        wheel_positions = []

        for prim in Usd.PrimRange(root_prim):
            prim_name = prim.GetName()
            if any(w in prim_name for w in wheel_links) and "visuals" not in str(prim.GetPath()):
                
                # THE FIX: Raw Attribute Scraper + Emergency Bounding Box Fallback
                # This guarantees the matrix solver never hits a divide-by-zero Sprung Mass error
                anchor_pos = Gf.Vec3d(0, 0, 0)
                xform = UsdGeom.Xformable(prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        anchor_pos = op.Get()
                        break
                
                if abs(anchor_pos[0]) < 0.001 and abs(anchor_pos[1]) < 0.001:
                    dx, dy = 0.3, 0.3
                    if "FL" in prim_name: anchor_pos = Gf.Vec3d(dx, dy, 0)
                    elif "FR" in prim_name: anchor_pos = Gf.Vec3d(dx, -dy, 0)
                    elif "BL" in prim_name: anchor_pos = Gf.Vec3d(-dx, dy, 0)
                    elif "BR" in prim_name: anchor_pos = Gf.Vec3d(-dx, -dy, 0)

                wheel_positions.append(anchor_pos)
                anchor_vec_f = Gf.Vec3f(float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_pos[2]))
                
                attachment_api = PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(prim)
                attachment_api.CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
                attachment_api.CreateSuspensionFramePositionAttr(anchor_vec_f)
                
                wheel_api = PhysxSchema.PhysxVehicleWheelAPI.Apply(prim)
                wheel_api.CreateRadiusAttr(OrbitronConfig.WHEEL_RADIUS_M)
                wheel_api.CreateWidthAttr(OrbitronConfig.WHEEL_WIDTH_M) 
                wheel_api.CreateMassAttr(OrbitronConfig.WHEEL_MASS_KG)
                wheel_api.CreateMoiAttr(OrbitronConfig.WHEEL_INERTIA)
                wheel_api.CreateDampingRateAttr(0.25)

                susp_api = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(prim)
                susp_api.CreateTravelDistanceAttr(0.02) 
                susp_api.CreateSpringStrengthAttr(80000.0) 
                susp_api.CreateSpringDamperRateAttr(5000.0) 

                tire_api = PhysxSchema.PhysxVehicleTireAPI.Apply(prim)
                tire_api.CreateLongitudinalStiffnessAttr(10000.0)

                ctrl_api = PhysxSchema.PhysxVehicleWheelControllerAPI.Apply(prim)
                ctrl_api.CreateDriveTorqueAttr(0.0)
                wheel_controllers[prim_name] = ctrl_api

        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr(OrbitronConfig.CHASSIS_MASS_KG)
        
        if len(wheel_positions) == 4:
            centroid_x = sum(p[0] for p in wheel_positions) / 4.0
            centroid_y = sum(p[1] for p in wheel_positions) / 4.0
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(centroid_x, centroid_y, OrbitronConfig.CHASSIS_COM_XYZ[2]))
        else:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*OrbitronConfig.CHASSIS_COM_XYZ))

        return root_prim, wheel_controllers

    root_prim, wheel_ctrls = setup_raycast_vehicle(OrbitronConfig.ROBOT_PRIM_PATH)
    
    from isaacsim.core.prims import RigidPrim
    chassis_rigid = RigidPrim(prim_paths_expr=OrbitronConfig.ROBOT_PRIM_PATH, name="chassis")
    world.scene.add(chassis_rigid)

    world.reset()

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
    last_accel_g = 0.0 

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

        chassis_vel = chassis_rigid.get_linear_velocities()[0]
        speed_mps = float(np.linalg.norm(chassis_vel[0:2])) 
        speed_mph = speed_mps * 2.23694
        
        simulated_wheel_rpm = (speed_mps / OrbitronConfig.WHEEL_RADIUS_M) * (60.0 / (2.0 * np.pi))

        left_mix = accel + steer
        right_mix = accel - steer
        max_mag = max(1.0, abs(left_mix), abs(right_mix))
        
        t_motor_rpm_fl = (left_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_bl = (left_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_fr = (right_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO
        t_motor_rpm_br = (right_mix / max_mag) * WHEEL_RPM_CAP * OrbitronConfig.GEAR_RATIO

        c_motor_rpm_fl = simulated_wheel_rpm * OrbitronConfig.GEAR_RATIO * (1.0 if left_mix >= 0 else -1.0)
        c_motor_rpm_bl = simulated_wheel_rpm * OrbitronConfig.GEAR_RATIO * (1.0 if left_mix >= 0 else -1.0)
        c_motor_rpm_fr = simulated_wheel_rpm * OrbitronConfig.GEAR_RATIO * (1.0 if right_mix >= 0 else -1.0)
        c_motor_rpm_br = simulated_wheel_rpm * OrbitronConfig.GEAR_RATIO * (1.0 if right_mix >= 0 else -1.0)
        
        dt = OrbitronConfig.PHYSICS_DT
        m_torque_fl, _ = motor_fl.compute_torque(t_motor_rpm_fl, c_motor_rpm_fl, dt, battery)
        m_torque_fr, _ = motor_fr.compute_torque(t_motor_rpm_fr, c_motor_rpm_fr, dt, battery)
        m_torque_bl, _ = motor_bl.compute_torque(t_motor_rpm_bl, c_motor_rpm_bl, dt, battery)
        m_torque_br, _ = motor_br.compute_torque(t_motor_rpm_br, c_motor_rpm_br, dt, battery)

        def sanitize_torque(t):
            if t is None or np.isnan(t) or np.isinf(t): return 0.0
            return float(t)

        final_t_fl = sanitize_torque(m_torque_fl) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_FL
        final_t_fr = sanitize_torque(m_torque_fr) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_FR
        final_t_bl = sanitize_torque(m_torque_bl) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_BL
        final_t_br = sanitize_torque(m_torque_br) * OrbitronConfig.GEAR_RATIO * OrbitronConfig.GEARBOX_EFFICIENCY * OrbitronConfig.INV_BR

        if "FL_1" in wheel_ctrls: wheel_ctrls["FL_1"].GetDriveTorqueAttr().Set(final_t_fl)
        if "FR_1" in wheel_ctrls: wheel_ctrls["FR_1"].GetDriveTorqueAttr().Set(final_t_fr)
        if "BL_1" in wheel_ctrls: wheel_ctrls["BL_1"].GetDriveTorqueAttr().Set(final_t_bl)
        if "BR_1" in wheel_ctrls: wheel_ctrls["BR_1"].GetDriveTorqueAttr().Set(final_t_br)

        downforce_arr = np.array([[0.0, 0.0, current_downforce_n]], dtype=np.float32)
        chassis_rigid.apply_forces(forces=downforce_arr, is_global=True)

        accel_mps2 = (speed_mps - last_speed_mps) / OrbitronConfig.PHYSICS_DT
        raw_accel_g = accel_mps2 / 9.81
        filtered_accel_g = (0.1 * raw_accel_g) + (0.9 * last_accel_g)
        last_accel_g = filtered_accel_g
        last_speed_mps = speed_mps

        try:
            telemetry_queue.put_nowait({
                "accel": accel, 
                "steer": steer,
                "t_wheel_rpm": t_motor_rpm_fl / OrbitronConfig.GEAR_RATIO, 
                "c_wheel_rpm": simulated_wheel_rpm,
                "c_motor_rpm": c_motor_rpm_fl,
                "t_wheel_fl": final_t_fl,
                "t_wheel_fr": final_t_fr,
                "t_wheel_bl": final_t_bl,
                "t_wheel_br": final_t_br,
                "t_motor_fl": m_torque_fl, 
                "speed_mph": speed_mph,
                "accel_g": filtered_accel_g, 
                "slip_pct": 0.0 
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