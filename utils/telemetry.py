import multiprocessing
import time
from collections import deque
import dearpygui.dearpygui as dpg

# ADDED: speed_limit_val pointer for bidirectional memory mapping
def run_telemetry(data_queue, speed_limit_val=None):
    dpg.create_context()
    dpg.create_viewport(title="Orbitron Live Telemetry", width=800, height=650)
    dpg.setup_dearpygui()

    max_points = 200
    time_data = deque(maxlen=max_points)
    target_rpm_fl = deque(maxlen=max_points)
    actual_rpm_fl = deque(maxlen=max_points)
    thrust_data = deque(maxlen=max_points)
    yaw_data = deque(maxlen=max_points)

    with dpg.window(label="Diagnostics", width=780, height=630, no_collapse=True):
        dpg.add_text("STATUS: Waiting for physics...", tag="status_text", color=[255, 200, 0])
        dpg.add_separator()
        
        # --- THE SPEED LIMITER SLIDER ---
        dpg.add_slider_float(label="Max Speed Limit %", tag="ui_max_speed", min_value=1.0, max_value=100.0, default_value=100.0)
        
        dpg.add_slider_float(label="Left Stick Y (Accel)", tag="ui_accel", min_value=-1.0, max_value=1.0, default_value=0.0)
        dpg.add_slider_float(label="Right Stick X (Steer)", tag="ui_steer", min_value=-1.0, max_value=1.0, default_value=0.0)
        dpg.add_separator()

        with dpg.plot(label="Front-Left Wheel (RPM)", height=200, width=-1):
            dpg.add_plot_legend()
            x_axis_rpm = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_rpm")
            y_axis_rpm = dpg.add_plot_axis(dpg.mvYAxis, label="RPM", tag="y_axis_rpm")
            dpg.add_line_series([], [], label="Target RPM", parent=y_axis_rpm, tag="plot_t_rpm")
            dpg.add_line_series([], [], label="Actual RPM", parent=y_axis_rpm, tag="plot_a_rpm")

        with dpg.plot(label="Phase 1 Injected Vectors", height=200, width=-1):
            dpg.add_plot_legend()
            x_axis_force = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_force")
            y_axis_force = dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude", tag="y_axis_force")
            dpg.add_line_series([], [], label="Forward Thrust (N)", parent=y_axis_force, tag="plot_thrust")
            dpg.add_line_series([], [], label="Yaw Torque (Nm)", parent=y_axis_force, tag="plot_yaw")

    dpg.show_viewport()
    start_time = time.time()
    
    while dpg.is_dearpygui_running():
        # WRITE: Push the GUI slider value back to the physics engine memory map
        if speed_limit_val is not None:
            speed_limit_val.value = dpg.get_value("ui_max_speed") / 100.0

        latest_data = None
        while not data_queue.empty():
            try:
                latest_data = data_queue.get_nowait()
            except:
                break
        
        if latest_data is not None:
            current_time = time.time() - start_time
            time_data.append(current_time)
            
            target_rpm_fl.append(latest_data["t_rpm_fl"])
            actual_rpm_fl.append(latest_data["c_rpm_fl"])
            thrust_data.append(latest_data["f_thrust"])
            yaw_data.append(latest_data["t_yaw"])
            
            dpg.set_value("status_text", "STATUS: Live Data Receiving")
            dpg.configure_item("status_text", color=[0, 255, 0])
            
            dpg.set_value("ui_accel", latest_data["accel"])
            dpg.set_value("ui_steer", latest_data["steer"])
            
            t_list = list(time_data)
            dpg.set_value("plot_t_rpm", [t_list, list(target_rpm_fl)])
            dpg.set_value("plot_a_rpm", [t_list, list(actual_rpm_fl)])
            dpg.set_value("plot_thrust", [t_list, list(thrust_data)])
            dpg.set_value("plot_yaw", [t_list, list(yaw_data)])
            
            dpg.fit_axis_data("x_axis_rpm")
            dpg.fit_axis_data("y_axis_rpm")
            dpg.fit_axis_data("x_axis_force")
            dpg.fit_axis_data("y_axis_force")
        
        dpg.render_dearpygui_frame()
    dpg.destroy_context()