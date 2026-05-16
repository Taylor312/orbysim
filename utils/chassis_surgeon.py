from pxr import UsdPhysics, PhysxSchema, UsdGeom, UsdShade, Gf
from utils.constants import OrbitronConfig

def setup_orbitron_physics(stage, root_path):
    print(f" [INFO] Executing Chassis Surgery (Phase 1: Frictionless Ice)")
    mat_path = "/World/Physics_Materials/Rubber"
    stage.DefinePrim("/World/Physics_Materials", "Scope")
    UsdShade.Material.Define(stage, mat_path)
    rubber_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    
    # Ice Floor for Phase 1 Vector Push
    rubber_mat.CreateStaticFrictionAttr(0.0)
    rubber_mat.CreateDynamicFrictionAttr(0.0)
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
        if "physics_sphere" in p_path: continue

        if "visuals" in p_path or "collisions" in p_path:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            continue

        if p_path.endswith("base_link") and prim.IsA(UsdGeom.Xformable):
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(OrbitronConfig.CHASSIS_MASS_KG)
            UsdPhysics.RigidBodyAPI.Apply(prim)

        if ("left" in p_path or "right" in p_path) and prim.IsA(UsdGeom.Xformable):
            if "joints" in p_path: continue
            
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(False)

            sphere_path = f"{p_path}/physics_sphere"
            sphere_geom = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere_geom.CreateRadiusAttr(OrbitronConfig.WHEEL_RADIUS_M) 
            UsdGeom.Imageable(sphere_geom).MakeInvisible()
            
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(sphere_path))
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(sphere_path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(mat_path)), materialPurpose="physics"
            )

            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(OrbitronConfig.WHEEL_MASS_KG)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive_api.CreateStiffnessAttr(0.0)
            drive_api.CreateDampingAttr(1e6)