import pygame
import dearpygui.dearpygui as dpg
import sys
import time

def main():
    # --- 1. PYGAME INIT & HARDWARE HUNT ---
    pygame.init()
    pygame.joystick.init()
    
    joy_count = pygame.joystick.get_count()
    if joy_count == 0:
        print("[ERROR] No joysticks or gamepads detected by Pygame.")
        sys.exit()

    print(f"\n[SCAN] Found {joy_count} devices:")
    target_joystick = None
    
    for i in range(joy_count):
        temp_joy = pygame.joystick.Joystick(i)
        name = temp_joy.get_name()
        print(f"  Device {i}: {name}")
        
        # Auto-select the first thing that looks like an Xbox controller
        if target_joystick is None and any(kw in name.lower() for kw in ["xbox", "xinput", "controller", "gamepad"]):
            target_joystick = pygame.joystick.Joystick(i)
            target_joystick.init()

    # Fallback if no specific "Xbox" name is found, just grab the first one
    if target_joystick is None:
        target_joystick = pygame.joystick.Joystick(0)
        target_joystick.init()

    print(f"\n[LOCKED ON] Using Device: {target_joystick.get_name()}")
    num_axes = target_joystick.get_numaxes()
    print(f"[INFO] Device has {num_axes} axes available.")

    # --- 2. DEARPYGUI SETUP ---
    dpg.create_context()
    dpg.create_viewport(title="Xbox Controller Axis Debugger", width=600, height=400)
    dpg.setup_dearpygui()

    with dpg.window(label="Live Hardware Feed", width=580, height=380, no_collapse=True):
        dpg.add_text(f"Active Device: {target_joystick.get_name()}", color=[0, 255, 0])
        dpg.add_separator()
        dpg.add_text("Wiggle your sticks and pull your triggers.\nNote which Axis number moves!")
        dpg.add_separator()
        
        # Dynamically create a slider for every axis the controller reports
        for i in range(num_axes):
            dpg.add_slider_float(label=f"Axis {i}", tag=f"axis_{i}", min_value=-1.0, max_value=1.0, default_value=0.0)

    dpg.show_viewport()

    # --- 3. THE READ LOOP ---
    while dpg.is_dearpygui_running():
        # THIS IS CRITICAL: Without pump(), Pygame will never update the joystick state
        pygame.event.pump() 
        
        # Read every axis and push the value directly to the GUI sliders
        for i in range(num_axes):
            raw_val = target_joystick.get_axis(i)
            
            # Apply a tiny deadzone so the sliders don't jitter from sensor noise
            if abs(raw_val) < 0.05:
                raw_val = 0.0
                
            dpg.set_value(f"axis_{i}", raw_val)
            
        dpg.render_dearpygui_frame()
        time.sleep(0.01) # ~100Hz refresh rate

    dpg.destroy_context()

if __name__ == "__main__":
    main()


    #python joystick_debug.py