class OrbitronConfig:
    # --- 1. ENVIRONMENT & ASSETS ---
    TARGET_GPU_UUID = "GPU-6edeed59-5bbf-c940-8e42-5830baef4d84"
    URDF_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.xacro"
    USD_OUTPUT_PATH = "C:/MasterData/Backable/3d ENG files/issacsimtestfiles/testudrf1_description/urdf/testudrf1.usd"
    ROBOT_PRIM_PATH = "/World/TankRover"

    # --- 2. SIMULATION TIMING & LIMITS ---
    PHYSICS_DT = 1.0 / 120.0
    MAX_TELEOP_RPM = 52200.0 # 69.6V * 750KV LiHV Peak

    # --- 3. CHASSIS KINEMATICS & PHYSICS ---
    CHASSIS_MASS_KG = 200.0
    WHEEL_MASS_KG = 1.5
    WHEEL_RADIUS_M = 0.03
    TRACK_WIDTH_M = 0.508
    WHEELBASE_M = 0.343  
    CG_HEIGHT_M = 0.106  

    # --- 4. MOTOR SIMULATOR (TP5870 750KV & Trampa VESC) ---
    V_BUS_MAX = 69.6          
    V_BUS_MIN = 51.2          
    MOTOR_KV = 750.0        
    MOTOR_RESISTANCE = 0.0081 
    ESC_CURRENT_LIMIT = 250.0 

    GEAR_RATIO = 30.0
    
    # THE FIX: Softened VESC PI Loop Tuning to prevent micro-chatter
    VESC_KP = 0.005   
    VESC_KI = 0.02   

    # --- 4.5 MOTOR THERMAL PROPERTIES ---
    T_AMBIENT = 25.0          
    C_TH_COPPER = 95.5        
    # THE FIX: Reduced effective bulk mass to the immediate stator iron to allow faster localized heating
    C_TH_BULK = 250.0         
    R_TH_INTERNAL = 0.25      
    R_TH_AMBIENT = 0.60       
    
    K_HYSTERESIS = 0.0012     
    # THE FIX: Aggressive high-frequency core loss coefficient
    K_EDDY_CURRENT = 2.5e-7   

    # --- 4.7 BATTERY PACK CONSTANTS (SMC HCL-HV2 16S Series) ---
    BATTERY_CAPACITY_AH = 5.9   
    BATTERY_RESISTANCE = 0.0192 

    # --- 5. HARDWARE ABSTRACTION LAYER (HAL) ---
    IDX_FR = 0 
    IDX_BR = 1 
    IDX_BL = 2 
    IDX_FL = 3 

    INV_FR = 1.0
    INV_BR = -1.0 
    INV_BL = -1.0 
    INV_FL = 1.0