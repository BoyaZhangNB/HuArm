from scservo_sdk import *   # Feetech's SDK
import time

PORT = "/dev/cu.usbmodem5B910265991"   # <-- yours
portHandler = PortHandler(PORT)
packetHandler = sms_sts(portHandler)

portHandler.openPort()
portHandler.setBaudRate(1_000_000)

# 1. Scan the bus for servos
for sid in range(1, 20):
    model, result, error = packetHandler.ping(sid)
    if result == COMM_SUCCESS:
        print(f"Found servo ID {sid}, model {model}")

ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_CURRENT = 69
ADDR_MOVING = 66

SERVO_ID = 1

def read_servo_status():
    """Read and display current servo status"""
    # Read position
    position, _, _ = portHandler.read2ByteTxRx(SERVO_ID, ADDR_PRESENT_POSITION)
    degrees = position * 360 / 4096
    
    # Read load (bits 0-9 = magnitude, bit 10 = direction)
    load_raw, _, _ = portHandler.read2ByteTxRx(SERVO_ID, ADDR_PRESENT_LOAD)
    load_magnitude = load_raw & 0x3FF  # Lower 10 bits
    load_direction = "CW" if load_raw & 0x400 else "CCW"  # Bit 10
    load_percent = load_magnitude / 1000 * 100
    
    # Read current draw
    current_raw, _, _ = portHandler.read2ByteTxRx(SERVO_ID, ADDR_PRESENT_CURRENT)
    current_ma = current_raw * 6.5  # Convert to milliamps
    
    # Read moving status
    moving, _, _ = portHandler.read1ByteTxRx(SERVO_ID, ADDR_MOVING)
    
    print(f"Position: {degrees:6.1f}° | Load: {load_percent:5.1f}% {load_direction} | Current: {current_ma:6.1f}mA | Moving: {'Yes' if moving else 'No'}")

# Move to a position and monitor
print("Moving servo and monitoring status...\n")

def write_target_pos(pos):
    packetHandler.WritePosEx(SERVO_ID, pos, 2500, 50)

write_target_pos(1000)
time.sleep(2)
write_target_pos(2000)

# Monitor for 2 seconds
for i in range(20):
    read_servo_status()
    time.sleep(0.1)

print("\nDone!")

# 3. Read back position and speed
pos, speed, result, error = packetHandler.ReadPosSpeed(1)
print("pos:", pos, "speed:", speed)

portHandler.closePort()