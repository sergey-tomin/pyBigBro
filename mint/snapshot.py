# -*- coding: utf-8 -*-

"""
S.Tomin

 machine snapshot
"""
import pandas as pd 
import numpy as np
import os

class Snapshot():
    def __init__(self):
        self.sase_sections = ["SA1", "SA2", "SA3"]
        # or list of magnet prefix to check if they are vary
        self.magnet_prefix = None # ["QA.", "CBY.", "CAX.", "Q.", "CY.", "CX.", "CIX.", "CIY.", "QI.", "BL."]
        
        self.orbit_sections = {}
        self.magnet_sections = {}
        self.phase_shifter_sections = {}
        self.undulators = {}
        self.channels = []
        self.channels_tol = []

        # alarm channels
        self.alarm_channels = []
        self.alarm_bounds = []
        self.channels_track = []
        
        # multidim data
        self.images = []
        self.image_folders = []
    
    def add_alarm_channels(self, ch, min, max):
        """
        return scalar channels
        """
        self.alarm_channels.append(ch)
        self.alarm_bounds.append([min, max])
    
    def add_image(self, ch, folder):
        # check if folder exists and create if not
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.images.append(ch)
        self.image_folders.append(folder)
    
    def add_orbit_section(self, sec_id, tol=0.001, track=True):
        self.orbit_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_magnet_section(self, sec_id, tol=0.001, track=True):
        self.magnet_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_phase_shifter_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            self.phase_shifter_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_undulator(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            self.undulators[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_channel(self, channel, tol=None, track=True):
        if channel in self.channels:
            print("WARNING: channel is already added")
            return
        self.channels.append(channel)
        self.channels_tol.append(tol)
        self.channels_track.append(track)
        
    
    def is_diff(self, snap1, snap2):
        diff = False
        for i, ch in enumerate(self.channels):
            print(ch, snap1[ch][0], snap2[ch][0], type(snap1[ch][0]))
            
            if (self.channels_tol[i] is not None) and self.channels_track[i] and np.abs(snap1[ch][0] - snap2[ch][0]) > self.channels_tol[i]:
                print("diff channels: ", ch, np.abs(snap1[ch][0] - snap2[ch][0]), self.channels_tol[i])
                diff = True
                
        for sec_id in self.orbit_sections:
            if self.orbit_sections[sec_id]["tol"] is not None and self.orbit_sections[sec_id]["track"]:
                for name in snap1.columns.values:

                    if "."+sec_id + ".X" in name:
                        #print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.orbit_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0])> tol:
                            print("diff orbit: ", name, np.abs(snap1[name][0] - snap2[name][0]), " tol = ", tol)
                            diff = True
                    if "."+sec_id + ".Y" in name:
                        #print("debug Y: ", name, "." + sec_id + ".X" in name)
                        tol = self.orbit_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0])> tol:
                            print("diff orbit: ", name, np.abs(snap1[name][0] - snap2[name][0]), " tol = ", tol)
                            diff = True

        for sec_id in self.magnet_sections:
            if self.magnet_sections[sec_id]["track"]:
                for name in snap1.columns.values:
                    if self.magnet_prefix is not None and any(x in name for x in self.magnet_prefix) and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.magnet_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        for sec_id in self.phase_shifter_sections:
            if self.phase_shifter_sections[sec_id]["track"]:
                for name in snap1.columns.values:

                    if "BPS." in name and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.phase_shifter_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        for sec_id in self.undulators:
            if self.undulators[sec_id]["track"]:
                for name in snap1.columns.values:

                    if ("U40." in name or "U68." in name) and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.phase_shifter_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        return diff
        
        
class SnapshotDB:
    def __init__(self, filename="test.p"):
        self.df = pd.DataFrame()
        self.filename = filename
    
    def add(self, df):
        self.df = self.df.append(df)
    
    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        self.df.to_pickle(filename)
    
    def load(self):
        return pd.read_pickle(self.filename)
    
    def __str__(self):
        return self.df.__str__()