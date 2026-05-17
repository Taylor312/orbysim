class OrbitronConfig:
    # --- 1. ENVIRONMENT & ASSETS ---
    TARGET_GPU_UUID = "GPU-6edeed59-5bbf-c940-8e42-5830baef4d84"
    
    # Updated paths pointing to the new orbysim_description directory
    URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/orbysim_description/urdf/orbysim.xacro"
    USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/orbysim_description/urdf/orbysim.usd"
    ROBOT_PRIM_PATH = "/World/TankRover"

    # --- 2. SIMULATION TIMING & LIMITS ---
    PHYSICS_HZ = 200
    RENDER_HZ = 60
    PHYSICS_DT = 1.0 / PHYSICS_HZ
    RENDER_DT = 1.0 / RENDER_HZ
    DEFAULT_RPM_CAP = 1600.0 

    # --- 3. 250LB HEAVYWEIGHT COMBAT PHYSICAL PROPERTIES ---
    # Chassis (Base Link) mass and locked origin Center of Mass
    CHASSIS_MASS_KG = 83.46
    CHASSIS_COM_XYZ = [0.0, 0.0, 0.0] # Locked directly to your balanced visual origin
    
    # 4lb Wheels with 0.025 kg*m^2 Moment of Inertia
    WHEEL_MASS_KG = 1.81
    WHEEL_RADIUS_M = 0.06604
    WHEEL_WIDTH_M = 0.0635
    WHEEL_INERTIA = 0.025 
    
    # THE FIX: Added the missing offset variable (0.0 keeps it perfectly centered on the joint origin)
    WHEEL_OFFSET_M = 0.03175
    
    # 25lb Weapon Spinners with 0.10 kg*m^2 Moment of Inertia
    WEAPON_MASS_KG = 11.34
    WEAPON_INERTIA = 0.10
    
    # Surface Traction & Contact Mechanics
    STATIC_FRICTION = 1.5
    DYNAMIC_FRICTION = 1.1
    RESTITUTION = 0.0
    GEARBOX_DRAG_DAMPING = 0.005
    
    # 1100 lbs of Magnets pinning the bot down
    DOWNFORCE_N = -4893.04
    DOWNFORCE_RAMP_TIME_S = 1.5

    # --- 4. MOTOR SIMULATOR (TP5870 750KV & Trampa VESC) ---
    V_BUS_MAX = 69.6          
    V_BUS_MIN = 51.2          
    MOTOR_KV = 750.0        
    MOTOR_RESISTANCE = 0.0081 
    ESC_CURRENT_LIMIT = 250.0 # Restored to maximum current delivery
    
    GEAR_RATIO = 25.0
    GEARBOX_EFFICIENCY = 0.90
    
    VESC_KP = 0.005   
    VESC_KI = 0.02   

    # --- 4.5 MOTOR THERMAL PROPERTIES ---
    T_AMBIENT = 25.0          
    C_TH_COPPER = 95.5        
    C_TH_BULK = 250.0         
    R_TH_INTERNAL = 0.25      
    R_TH_AMBIENT = 0.60       
    K_HYSTERESIS = 0.0012     
    K_EDDY_CURRENT = 2.5e-7   

    # --- 4.7 BATTERY PACK CONSTANTS (SMC HCL-HV2 16S Series) ---
    BATTERY_CAPACITY_AH = 5.9   
    BATTERY_RESISTANCE = 0.0192 

    # --- 5. HARDWARE ABSTRACTION LAYER (HAL) ---
    INV_FR = 1.0
    INV_BR = -1.0 
    INV_BL = -1.0 
    INV_FL = 1.0