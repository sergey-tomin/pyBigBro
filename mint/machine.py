# -*- coding: utf-8 -*-
"""
S.Tomin

Machine class to get machine snapshot
"""

from mint.xfel_interface import XFELMachineInterface
import pandas as pd 
import numpy as np
import time
import scipy.misc
import os
import pickle

class Machine:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.mi = XFELMachineInterface()
        self.bpm_server = "ORBIT"     # or "BPM"
        self.server = "XFEL"
        self.subtrain = "ALL"
        self.suffix = ""


    def is_machine_online(self):
        """
        method to check if machine is online

        :return: True if online
        """

        alarm_status = []
        for i, alarm in enumerate(self.snapshot.alarm_channels):
            try:
                val = self.mi.get_value(alarm)
            except Exception as e:
                print("id: " + alarm+ " ERROR: " + str(e))
                val = 0
            min_val, max_val = self.snapshot.alarm_bounds[i]
            if min_val < val < max_val:
                alarm_status.append(True)
            else:
                alarm_status.append(False)
        if np.array(alarm_status).any():
            return True
        return False

    def get_orbit(self, data, all_names):
        for sec_id in self.snapshot.orbit_sections:
            try:
                orbit_x = np.array(self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/X." + self.subtrain + self.suffix))

                orbit_y = np.array(self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/Y."  + self.subtrain + self.suffix))
            except Exception as e:
                print("orbit id: " + sec_id+ " ERROR: " + str(e))
                return [], []
            x = orbit_x[:, 1].astype(np.float)
            y = orbit_y[:, 1].astype(np.float)
            xy = np.append(x, y)

            names_x =  [name + ".X" for name in orbit_x[:, 4]]
            names_y =  [name + ".Y" for name in orbit_y[:, 4]]
            names = np.append(names_x, names_y)
            data = np.append(data, xy)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_magnets(self, data, all_names):
        for sec_id in self.snapshot.magnet_sections:
            try:
                magnets = np.array(self.mi.get_value("XFEL.MAGNETS/MAGNET.ML/*." + sec_id + "/KICK_MRAD.SP"))
            except Exception as e:
                print("magnets id: " + sec_id+ " ERROR: " + str(e))
                return [], []
            vals = magnets[:, 1].astype(np.float)

            names = [name for name in magnets[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_channels(self, data, all_names):

        for ch in self.snapshot.channels:
            try:
                val = self.mi.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                val = np.nan
            #print(ch, type(val))
            data = np.append(data, val)
            all_names = np.append(all_names, ch)
        return data, all_names
    
    def get_images(self, data, all_names):
        for i, ch in enumerate(self.snapshot.images):
            folder = self.snapshot.image_folders[i]
            try:
                img = self.mi.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                img = None
            
            cam_name = ch.split("/")[-2]
            
            name = cam_name + "-" + time.strftime("%Y%m%d-%H%M%S") 
            filename = name + ".png"
            path = folder + os.sep + filename
            path_pcl = folder + os.sep + name + ".pcl"
            if img is not None: 
                scipy.misc.imsave(path, img)
                with open(path_pcl, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(img, f)
                
            else:
                path = None
            #print(data)
            data = list(data)
            data = np.append(data, np.array([path], dtype=object))
            all_names = np.append(all_names, ch)
        return data, all_names
    
    def get_machine_snapshot(self):

        if not self.is_machine_online():
            print("machine is not online. wait 3 sec ...")
            time.sleep(2)
            return None


        data = np.array([time.time()], dtype=object)
        all_names = np.array(["timestamp"])
        data, all_names = self.get_orbit(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_magnets(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_channels(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_images(data, all_names)
        if len(data) == 0:
            print("get_images bad")
            return None
        #print(data)
        df = pd.DataFrame(data=data.reshape((1, len(data))), columns=all_names)
        #df = df.apply(pd.to_numeric, errors='ignore')
        return df          
            
    

if __name__ is "__main__":
    df = pd.DataFrame()
    df2 = pd.DataFrame(data = {"a": 2, "b":4}, index=[0])
    df3 = pd.DataFrame(data = {"b": [5], "a":[6], "c": [3]})
    df = df.append(df2)
    print("1", df)
    df = df.append(df3)
    print("2", df)