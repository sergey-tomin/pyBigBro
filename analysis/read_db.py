import sys
sys.path.append("/Users/tomins/ownCloud/DESY/repository/")
import pandas as pd
from pyBigBro.mint.snapshot import *

db = SnapshotDB("/Users/tomins/ownCloud/DESY/repository/pyBigBro/simple_scan20200121-16_30_15.pcl")
df2 = db.load()
print(df2)
db.analyse()