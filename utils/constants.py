class OrbitronConfig:
    # --- 1. ENVIRONMENT & ASSETS ---
    TARGET_GPU_UUID = "GPU-6edeed59-5bbf-c940-8e42-5830baef4d84"
    URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/orbysim_description/urdf/orbysim.xacro"
    USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/orbysim_description/urdf/orbysim.usd"
    ROBOT_PRIM_PATH = "/World/TankRover"

    # --- 2. SIMULATION TIMING & LIMITS ---
    PHYSICS_HZ = 144
    RENDER_HZ = 60
    PHYSICS_DT = 1.0 / PHYSICS_HZ
    RENDER_DT = 1.0 / RENDER_HZ
    DEFAULT_RPM_CAP = 1754.0 

    # --- 3. 250LB HEAVYWEIGHT COMBAT PHYSICAL PROPERTIES ---
    CHASSIS_MASS_KG = 83.46
    CHASSIS_COM_XYZ = [0.0, 0.0, 0.0] 
    
    WHEEL_MASS_KG = 1.81
    WHEEL_RADIUS_M = 0.06604
    WHEEL_WIDTH_M = 0.0635
    # THE STABILITY FIX: Goldilocks inertia prevents solver bouncing
    WHEEL_INERTIA = 0.0025
    WHEEL_OFFSET_M = 0.03175
    
    WEAPON_MASS_KG = 11.34
    WEAPON_INERTIA = 0.10
    
    STATIC_FRICTION = 1.5
    DYNAMIC_FRICTION = 1.1
    RESTITUTION = 0.0
    GEARBOX_DRAG_DAMPING = 0.005 
    
    DOWNFORCE_N = -4893.04 
    DOWNFORCE_RAMP_TIME_S = 1.5

    # --- 4. MOTOR SIMULATOR (TP5870 750KV & Trampa VESC) ---
    V_BUS_MAX = 69.6          
    V_BUS_MIN = 51.2          
    MOTOR_KV = 750.0        
    MOTOR_RESISTANCE = 0.0081 
    ESC_CURRENT_LIMIT = 280.0 
    
    GEAR_RATIO = 28.5
    GEARBOX_EFFICIENCY = 0.85
    
    # THE TORQUE WALL FIX: Nuked the Integral windup. P-Gain only.
    VESC_KP = 0.025   
    VESC_KI = 0.0   

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