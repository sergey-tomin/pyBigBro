import sys
sys.path.append("/Users/tomins/ownCloud/DESY/repository/")
import pandas as pd
import matplotlib.pyplot as plt
from pyBigBro.mint.snapshot import *
from pyBigBro.image_proc.tds_analysis import *

db = SnapshotDB("/Users/tomins/ownCloud/DESY/repository/pyBigBro/simple_scan20200121-16_30_15.pcl")
df2 = db.load()
print(df2['timestamp'])
x, y = db.get_orbits(section_id="I1")
y.T.plot(legend='timestamp')
plt.show()