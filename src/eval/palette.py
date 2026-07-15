"""Canonical color palette for the 30-class Go2W outdoor taxonomy.

Mirrors NeoVerse's `diffsynth/utils/semantics.CLASS_COLORS` so we don't need
torch on Mac to render label overlays. Values are (R, G, B) uint8 in [0, 255].

LOCKSTEP: if NeoVerse's CLASS_COLORS changes, update this table too.
"""
from __future__ import annotations

import numpy as np


CLASS_COLORS_255 = np.array([
    (  0,   0,   0),   # 0  void
    (200, 225, 245),   # 1  sky
    (139,  90,  43),   # 2  dirt
    (230, 200, 155),   # 3  sand
    ( 75, 190,  80),   # 4  grass
    (180, 155, 100),   # 5  gravel
    (110,  55,  25),   # 6  mulch
    ( 55,  55,  30),   # 7  mud
    ( 50, 120, 200),   # 8  water
    (135, 145, 155),   # 9  rock
    ( 55,  55,  65),   # 10 asphalt
    (225, 220, 190),   # 11 concrete
    (110, 110, 115),   # 12 road
    (180, 180, 180),   # 13 sidewalk
    (255, 250, 235),   # 14 crosswalk
    (170,  75,  60),   # 15 building
    (175, 145, 175),   # 16 wall
    ( 90,  60, 130),   # 17 fence
    ( 75, 155, 175),   # 18 bridge
    ( 40, 105,  55),   # 19 tree
    (170, 200,  55),   # 20 vegetation
    (135, 115,  90),   # 21 log
    (220, 140,  80),   # 22 stairs
    ( 25,  65, 130),   # 23 pole
    (230, 195,  60),   # 24 traffic_sign
    (235,  85,  75),   # 25 traffic_light
    (110, 130, 220),   # 26 vehicle
    (155,  60, 200),   # 27 motorcycle
    (100, 230, 200),   # 28 bicycle
    (205,  70, 145),   # 29 person
], dtype=np.uint8)
