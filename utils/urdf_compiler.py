import os
import omni.kit.commands

def compile_and_import(urdf_path, usd_output_path):
    print(f" [INFO] Compiling URDF -> USD...")
    if os.path.exists(usd_output_path):
        os.remove(usd_output_path)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False 
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = True

    omni.kit.commands.execute(
        "URDFParseAndImportFile", 
        urdf_path=urdf_path, 
        import_config=import_config, 
        dest_path=usd_output_path
    )
    print(f" [SUCCESS] USD Generated at {usd_output_path}")