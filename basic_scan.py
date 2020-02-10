"""
Simple example for scanning of a DOOCS channel and collecting data which was configured

@author: S.Tomin
"""
import pydoocs
import numpy as np
import matplotlib.pyplot as plt
import time
from mint.machine import Machine
from mint.snapshot import Snapshot, SnapshotDB
import basic_config
import time

# DOOCS channel to scan. Fake channel
doocs_ch = "XFEL/DOOCS/CHANNEL/A.SP"

# read initial value of the doocs channel
x_0 = pydoocs.read(doocs_ch)["data"]

# scan range
x_range = np.arange(-30., 30., 2)

# define Snapshot config
snapshot = basic_config.snapshot

# init Machine
machine = Machine(snapshot)
# init DataBase
db = SnapshotDB(filename="simple_scan" + time.strftime("%Y%m%d-%H_%M_%S") + ".pcl")

# get machine snapshot before scan
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

# scanning loop
for x_i in x_range:
    # set new value
    print("set: {} <-- {}".format(doocs_ch, x_i))
    pydoocs.write(doocs_ch, x_i)
    # sleep 1 sec. Here maybe we need to wait a bit longer in some cases. Especially if the images are saved
    time.sleep(1)

    while True:
        # check if machine is online. Looking to snapshot alarm channels
        if machine.is_machine_online():
            break
        else:
            # if is not online sleep for 3 sec
            time.sleep(3)
            print("sleep 3 sec ..")
    # get machine snapshot
    df = machine.get_machine_snapshot()
    # add data to DataBase db
    db.add(df)
    time.sleep(1)

db.save()

pydoocs.write(doocs_ch, x_0)


