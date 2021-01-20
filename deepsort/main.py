# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:19:47 2021

@author: trist
"""

from tracker import Tracker
from metrics import NearestNeighbor 


def main():
    
    metric = NearestNeighbor
    tracker = Tracker(metric)