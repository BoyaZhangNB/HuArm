from scservo_sdk import *

PORT = "/dev/cu.usbmodem5B910265991"   # <-- yours
portHandler = PortHandler(PORT)
packetHandler = sms_sts(portHandler)

portHandler.openPort()
portHandler.setBaudRate(1_000_000)

print(SMS_STS_ID)

OLD, NEW = 1, 5
packetHandler.unLockEprom(OLD)
packetHandler.write1ByteTxRx(OLD, SMS_STS_ID, NEW)
packetHandler.LockEprom(NEW)