import multiprocessing
import time
from collections import deque
import dearpygui.dearpygui as dpg

def run_telemetry(data_queue, cmd_queue=None):
    dpg.create_context()
    dpg.create_viewport(title="Orbitron Live Telemetry", width=900, height=1050)
    dpg.setup_dearpygui()

    max_points = 300
    time_data = deque(maxlen=max_points)
    
    t_wheel_rpm = deque(maxlen=max_points)
    c_wheel_rpm = deque(maxlen=max_points)
    
    torque_fl = deque(maxlen=max_points)
    torque_fr = deque(maxlen=max_points)
    torque_bl = deque(maxlen=max_points)
    torque_br = deque(maxlen=max_points)
    
    g_force_data = deque(maxlen=max_points)

    with dpg.window(label="Diagnostics", tag="primary_window", no_collapse=True):
        dpg.add_text("STATUS: Waiting for physics...", tag="status_text", color=[255, 200, 0])
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="Accel (Y)", tag="ui_accel", min_value=-1.0, max_value=1.0, default_value=0.0, width=150)
            dpg.add_slider_float(label="Steer (X)", tag="ui_steer", min_value=-1.0, max_value=1.0, default_value=0.0, width=150)
            dpg.add_slider_float(label="Max Wheel RPM", tag="ui_max_rpm", min_value=100.0, max_value=3000.0, default_value=1754.0, width=200)
        
        dpg.add_separator()
        
        dpg.add_text("--- LIVE KINEMATICS ---", color=[100, 200, 255])
        with dpg.group(horizontal=True):
            dpg.add_text("Speed: 0.00 MPH", tag="txt_speed")
            dpg.add_spacer(width=20)
            dpg.add_text("Accel: 0.00 G", tag="txt_accel")
            dpg.add_spacer(width=20)
            dpg.add_text("Slip: 0.0 %", tag="txt_slip", color=[255, 100, 100])
            
        with dpg.group(horizontal=True):
            dpg.add_text("Raw Motor RPM: 0.0", tag="txt_raw_rpm")
            dpg.add_spacer(width=20)
            dpg.add_text("Raw ESC Torque: 0.00 Nm", tag="txt_raw_torque")

        dpg.add_separator()

        with dpg.plot(label="Wheel RPM (Target vs Actual)", height=220, width=-1):
            dpg.add_plot_legend()
            x_axis_rpm = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_rpm")
            
            y_axis_rpm = dpg.add_plot_axis(dpg.mvYAxis, label="RPM", tag="y_axis_rpm")
            dpg.set_axis_limits("y_axis_rpm", -1800.0, 1800.0)
            
            dpg.add_line_series([], [], label="Target Wheel RPM", parent=y_axis_rpm, tag="plot_t_rpm")
            dpg.add_line_series([], [], label="Actual Wheel RPM", parent=y_axis_rpm, tag="plot_c_rpm")

        with dpg.plot(label="Wheel Torque (Nm) [Geared]", height=220, width=-1):
            dpg.add_plot_legend()
            x_axis_trq = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_trq")
            
            y_axis_trq = dpg.add_plot_axis(dpg.mvYAxis, label="Torque (Nm)", tag="y_axis_trq")
            dpg.set_axis_limits("y_axis_trq", -110.0, 110.0)
            
            dpg.add_line_series([], [], label="FL Torque", parent=y_axis_trq, tag="plot_t_fl")
            dpg.add_line_series([], [], label="FR Torque", parent=y_axis_trq, tag="plot_t_fr")
            dpg.add_line_series([], [], label="BL Torque", parent=y_axis_trq, tag="plot_t_bl")
            dpg.add_line_series([], [], label="BR Torque", parent=y_axis_trq, tag="plot_t_br")
            
        with dpg.plot(label="Longitudinal Acceleration (G-Force)", height=220, width=-1):
            dpg.add_plot_legend()
            x_axis_g = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis_g")
            
            y_axis_g = dpg.add_plot_axis(dpg.mvYAxis, label="G-Force", tag="y_axis_g")
            dpg.set_axis_limits("y_axis_g", -8.0, 8.0)
            
            dpg.add_line_series([], [], label="G-Force", parent=y_axis_g, tag="plot_g")

    dpg.set_primary_window("primary_window", True)
    dpg.show_viewport()
    start_time = time.time()
    
    while dpg.is_dearpygui_running():
        latest_data = None
        while not data_queue.empty():
            try:
                latest_data = data_queue.get_nowait()
            except:
                break
        
        if latest_data is not None:
            current_time = time.time() - start_time
            time_data.append(current_time)
            
            t_wheel_rpm.append(latest_data["t_wheel_rpm"])
            c_wheel_rpm.append(latest_data["c_wheel_rpm"])
            torque_fl.append(latest_data["t_wheel_fl"])
            torque_fr.append(latest_data["t_wheel_fr"])
            torque_bl.append(latest_data["t_wheel_bl"])
            torque_br.append(latest_data["t_wheel_br"])
            g_force_data.append(latest_data["accel_g"])
            
            dpg.set_value("status_text", "STATUS: Live Data Receiving")
            dpg.configure_item("status_text", color=[0, 255, 0])
            dpg.set_value("ui_accel", latest_data["accel"])
            dpg.set_value("ui_steer", latest_data["steer"])
            
            dpg.set_value("txt_speed", f"Speed: {latest_data['speed_mph']:.2f} MPH")
            dpg.set_value("txt_accel", f"Accel: {latest_data['accel_g']:.2f} G")
            
            if latest_data['slip_pct'] >= 0:
                dpg.set_value("txt_slip", f"Slip: {latest_data['slip_pct']:.1f} %")
            else:
                dpg.set_value("txt_slip", "Slip: --- (Turning)")
                
            dpg.set_value("txt_raw_rpm", f"Raw Motor RPM: {latest_data['c_motor_rpm']:.0f}")
            dpg.set_value("txt_raw_torque", f"Raw ESC Torque: {latest_data['t_motor_fl']:.2f} Nm")
            
            t_list = list(time_data)
            dpg.set_value("plot_t_rpm", [t_list, list(t_wheel_rpm)])
            dpg.set_value("plot_c_rpm", [t_list, list(c_wheel_rpm)])
            
            dpg.set_value("plot_t_fl", [t_list, list(torque_fl)])
            dpg.set_value("plot_t_fr", [t_list, list(torque_fr)])
            dpg.set_value("plot_t_bl", [t_list, list(torque_bl)])
            dpg.set_value("plot_t_br", [t_list, list(torque_br)])
            
            dpg.set_value("plot_g", [t_list, list(g_force_data)])
            
            dpg.fit_axis_data("x_axis_rpm")
            dpg.fit_axis_data("x_axis_trq")
            dpg.fit_axis_data("x_axis_g")

        if cmd_queue is not None:
            try:
                cmd_queue.put_nowait({"max_rpm": dpg.get_value("ui_max_rpm")})
            except:
                pass

        dpg.render_dearpygui_frame()
    dpg.destroy_context()