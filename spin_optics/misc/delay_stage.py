import visa
import re
import signal
import time
import sys

rm = visa.ResourceManager()
ds = rm.open_resource('ASRL23::INSTR')
ds.timeout = 60000
ds.clear() # Make sure the initial help command isn't in the buffer

def signal_handler(signal, frame):
    print("Stopping!")
    ds.close()
    rm.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def move_to(x):
    x = min(46500, x)
    x = max(0, x)
    return int(re.split(' ', ds.query('MT %d' % x).rstrip())[1])

ds.write('SS 1000')
ds.write('SA 10000')
while True:
    a = move_to(43000)
    print("Arrived at " + str(a))
    time.sleep(0.05)
    a = move_to(0)
    print("Arrived at " + str(a))