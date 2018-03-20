#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:38:36 2018

@author: nicob
"""
import matplotlib

print([i for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
if 'times' in i.lower()])