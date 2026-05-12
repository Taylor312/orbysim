import ntcore
import time

# 1. Setup the Client
inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient4("IsaacSim Bridge") 

# 2. Connect to the Server found in your log
# Server = localhost (Your PC)
# Port = 5810 (From your log: "NT4 port 5810")
inst.setServer("localhost", 5810) 

print("Attempting to connect to Java Robot Code...")

# 3. Wait loop
while not inst.isConnected():
    print(f"Connecting... (Is the Java Sim running?)")
    time.sleep(1)

print("\nSUCCESS: Connected to NetworkTables!")

# 4. Prove it works (Check AdvantageScope after this runs)
table = inst.getTable("Sim")
pub = table.getDoubleTopic("HelloIsaac").publish()

while True:
    current_time = time.time()
    pub.set(current_time)
    print(f"Sending Time: {current_time:.2f}")
    time.sleep(1)