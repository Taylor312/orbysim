import os
import sys
import time
import numpy as np
from collections import deque
import pygame
import dearpygui.dearpygui as dpg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import OrbitronConfig
from utils.motor_sim import Apex3Motor
from utils.battery_sim import OrbyBattery

def main():
    pygame.init()
    pygame.joystick.init()
    joystick = None
    
    for i in range(pygame.joystick.get_count()):
        temp_joy = pygame.joystick.Joystick(i)
        name = temp_joy.get_name().lower()
        if "xbox" in name or "controller" in name or "xinput" in name:
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            print(f" [SUCCESS] Locked onto Gamepad: {joystick.get_name()}")
            break

    if joystick is None:
        print("[ERROR] No Xbox controller found! Plug it in.")
        return

    motor = Apex3Motor()
    battery = OrbyBattery()
    
    current_rpm = 0.0
    dt = 1.0 / 60.0  
    
    # THE NEW ANGLE: Sub-stepping variables
    physics_substeps = 50
    sub_dt = dt / physics_substeps
    
    max_theoretical_rpm = OrbitronConfig.V_BUS_MAX * OrbitronConfig.MOTOR_KV 

    dpg.create_context()
    dpg.create_viewport(title="VESC PID, Thermal & Battery Simulator", width=1000, height=1000)
    dpg.setup_dearpygui()

    max_points = 300
    time_data = deque(maxlen=max_points)
    t_rpm_data = deque(maxlen=max_points)
    a_rpm_data = deque(maxlen=max_points)
    torque_data = deque(maxlen=max_points)
    t_copper_data = deque(maxlen=max_points)
    t_bulk_data = deque(maxlen=max_points)
    v_bat_data = deque(maxlen=max_points)

    with dpg.window(label="Test Stand Dashboard", width=980, height=950, no_collapse=True):
        dpg.add_text("Push Left Stick Y to command RPM", color=[0, 255, 0])
        
        with dpg.group(horizontal=True):
            dpg.add_text("Battery Voltage: ")
            dpg.add_text("69.6V", tag="txt_bat_voltage", color=[0, 200, 255])
            dpg.add_text(" | Pack SoC: ")
            dpg.add_text("100.000%", tag="txt_bat_soc", color=[0, 255, 100])
            dpg.add_text(" | Bus Current: ")
            dpg.add_text("0.0A", tag="txt_bus_current", color=[255, 150, 0])
        
        dpg.add_separator()
        dpg.add_text("Tuning Parameters:")
        dpg.add_slider_float(label="Kp (Proportional)", tag="ui_kp", min_value=0.0, max_value=0.1, default_value=OrbitronConfig.VESC_KP, format="%.4f", width=400)
        dpg.add_slider_float(label="Ki (Integral)", tag="ui_ki", min_value=0.0, max_value=0.5, default_value=OrbitronConfig.VESC_KI, format="%.4f", width=400)
        dpg.add_slider_float(label="ESC Current Limit (Amps)", tag="ui_current_limit", min_value=10.0, max_value=300.0, default_value=OrbitronConfig.ESC_CURRENT_LIMIT, format="%.1f", width=400)
        dpg.add_slider_float(label="Flywheel Inertia (kg*m^2)", tag="ui_inertia", min_value=0.00001, max_value=0.5, default_value=0.00006, format="%.5f", width=400)
        dpg.add_slider_float(label="Dyno Friction Load (Nm)", tag="ui_dyno_load", min_value=0.0, max_value=5.0, default_value=0.0, format="%.2f", width=400)
        dpg.add_separator()

        with dpg.plot(label="Velocity Control Loop (RPM)", height=180, width=-1):
            dpg.add_plot_legend()
            x_axis_rpm = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_rpm")
            y_axis_rpm = dpg.add_plot_axis(dpg.mvYAxis, label="RPM", tag="y_axis_rpm")
            dpg.set_axis_limits(y_axis_rpm, -max_theoretical_rpm - 2000, max_theoretical_rpm + 2000)
            dpg.add_line_series([], [], label="Target RPM", parent=y_axis_rpm, tag="plot_t_rpm")
            dpg.add_line_series([], [], label="Actual RPM", parent=y_axis_rpm, tag="plot_a_rpm")

        with dpg.plot(label="Motor Output Torque (Nm)", height=160, width=-1):
            dpg.add_plot_legend()
            x_axis_torque = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_torque")
            y_axis_torque = dpg.add_plot_axis(dpg.mvYAxis, label="Torque (Nm)", tag="y_axis_torque")
            dpg.set_axis_limits(y_axis_torque, -6.0, 6.0) 
            dpg.add_line_series([], [], label="Net Torque", parent=y_axis_torque, tag="plot_torque")

        with dpg.plot(label="Two-Node Thermal Sensor Core (deg C)", height=160, width=-1):
            dpg.add_plot_legend()
            x_axis_temp = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_temp")
            y_axis_temp = dpg.add_plot_axis(dpg.mvYAxis, label="Temperature (C)", tag="y_axis_temp")
            dpg.set_axis_limits(y_axis_temp, 20.0, 120.0)
            dpg.add_drag_line(label="Max Threshold (105C)", color=[255, 0, 0, 255], default_value=105.0, vertical=False)
            dpg.add_line_series([], [], label="Winding Temp (Copper)", parent=y_axis_temp, tag="plot_t_copper")
            dpg.add_line_series([], [], label="Casing Temp (Bulk)", parent=y_axis_temp, tag="plot_t_bulk")
            
        with dpg.plot(label="Loaded Battery Pack DC Rail Voltage (V)", height=160, width=-1):
            dpg.add_plot_legend()
            x_axis_bat = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_bat")
            y_axis_bat = dpg.add_plot_axis(dpg.mvYAxis, label="Voltage (V)", tag="y_axis_bat")
            dpg.set_axis_limits(y_axis_bat, OrbitronConfig.V_BUS_MIN - 2.0, OrbitronConfig.V_BUS_MAX + 5.0)
            dpg.add_line_series([], [], label="Terminal Voltage", parent=y_axis_bat, tag="plot_v_bat")

    dpg.show_viewport()
    start_time = time.time()
    
    while dpg.is_dearpygui_running():
        loop_start = time.time()
        pygame.event.pump()
        
        motor.kp = dpg.get_value("ui_kp")
        motor.ki = dpg.get_value("ui_ki")
        motor.current_limit = dpg.get_value("ui_current_limit")
        flywheel_inertia = max(0.00001, dpg.get_value("ui_inertia")) 
        dyno_load = dpg.get_value("ui_dyno_load")
        
        raw_stick = -joystick.get_axis(1)
        if abs(raw_stick) < 0.05: raw_stick = 0.0 
        target_rpm = raw_stick * max_theoretical_rpm

        # THE NEW ANGLE: 3,000 Hz Physics Micro-Loop
        for _ in range(physics_substeps):
            torque_nm, bus_current = motor.compute_torque(target_rpm, current_rpm, sub_dt, battery_obj=battery)
            battery.step(bus_current, sub_dt)

            if current_rpm > 50:
                mech_load = dyno_load
            elif current_rpm < -50:
                mech_load = -dyno_load
            else:
                mech_load = (current_rpm / 50.0) * dyno_load  

            damping_torque = 0.00001 * (current_rpm * (np.pi / 30.0))  
            net_torque = torque_nm - damping_torque - mech_load
            
            alpha = net_torque / flywheel_inertia
            rads_per_sec = (current_rpm * (np.pi / 30.0)) + (alpha * sub_dt)
            current_rpm = rads_per_sec * (30.0 / np.pi)

        # UI Updates & Graph Logging (only drawn once per 60Hz frame)
        dpg.set_value("txt_bat_voltage", f"{battery.v_bus:.2f}V")
        dpg.set_value("txt_bat_soc", f"{(battery.soc * 100.0):.3f}%")
        dpg.set_value("txt_bus_current", f"{bus_current:.1f}A")

        current_time = time.time() - start_time
        time_data.append(current_time)
        t_rpm_data.append(target_rpm)
        a_rpm_data.append(current_rpm)
        torque_data.append(torque_nm)
        t_copper_data.append(motor.T_copper)
        t_bulk_data.append(motor.T_bulk)
        v_bat_data.append(battery.v_bus)

        t_list = list(time_data)
        dpg.set_value("plot_t_rpm", [t_list, list(t_rpm_data)])
        dpg.set_value("plot_a_rpm", [t_list, list(a_rpm_data)])
        dpg.set_value("plot_torque", [t_list, list(torque_data)])
        dpg.set_value("plot_t_copper", [t_list, list(t_copper_data)])
        dpg.set_value("plot_t_bulk", [t_list, list(t_bulk_data)])
        dpg.set_value("plot_v_bat", [t_list, list(v_bat_data)])
        
        dpg.fit_axis_data("x_axis_rpm")
        dpg.fit_axis_data("x_axis_torque")
        dpg.fit_axis_data("x_axis_temp")
        dpg.fit_axis_data("x_axis_bat")
        
        dpg.render_dearpygui_frame()
        
        sleep_time = dt - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    dpg.destroy_context()

if __name__ == "__main__":
    main()