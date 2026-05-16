import numpy as np
from utils.constants import OrbitronConfig

class OrbyBattery:
    def __init__(self):
        self.capacity_ah = OrbitronConfig.BATTERY_CAPACITY_AH
        self.max_capacity_as = self.capacity_ah * 3600.0  
        self.current_capacity_as = self.max_capacity_as
        self.R_internal = OrbitronConfig.BATTERY_RESISTANCE
        
        self.soc = 1.0
        # Expose Open Circuit Voltage for implicit circuit solving
        self.v_oc = OrbitronConfig.V_BUS_MAX
        self.v_bus = self.v_oc

    def step(self, total_bus_current, dt):
        """
        Updates electrochemical discharge capacity state
        """
        self.current_capacity_as -= total_bus_current * dt
        self.current_capacity_as = np.clip(self.current_capacity_as, 0.0, self.max_capacity_as)
        
        self.soc = self.current_capacity_as / self.max_capacity_as
        
        v_cell = OrbitronConfig.V_BUS_MIN/16.0 + (OrbitronConfig.V_BUS_MAX/16.0 - OrbitronConfig.V_BUS_MIN/16.0) * self.soc
        self.v_oc = v_cell * 16.0
        
        self.v_bus = self.v_oc - (total_bus_current * self.R_internal)
        return self.v_bus