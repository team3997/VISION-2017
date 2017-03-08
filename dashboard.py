import sys
import time
from networktables import NetworkTables
import logging

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]

NetworkTables.initialize(server=ip)

dashboard = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
    try:
        print('DEBUG_FPGATimestamp:', dashboard.getNumber('DEBUG_FPGATimestamp'))
    except KeyError:
        print('DEBUG_FPGATimestamp: N/A')

    dashboard.putNumber('piTime:', i)
    time.sleep(1)
    i += 1
