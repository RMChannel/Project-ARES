import struct
import math
from dataclasses import dataclass

@dataclass
class Coordinates:
    x: float; y: float; z: float
    dist: float; id: int; direction: float
    right_bound: float; left_bound: float; angle: float

def get_data(file_path):
    with open(file_path, "rb") as f:
        _, count, _, _ = struct.unpack("<4i", f.read(16))
        ideal = [struct.unpack("<4fi", f.read(20)) for _ in range(count)]
        detail = [struct.unpack("<18f", f.read(72)) for _ in range(count)]

    coords = []
    for i in range(count):
        x, y, z, dist, row_id = ideal[i]
        direction, rb, lb = detail[i][4:7]
        
        px, _, pz, _, _ = ideal[(i - 1) % count]
        angle = math.atan2(pz - z, x - px)
        
        coords.append(Coordinates(x, y, z, dist, row_id, direction, rb, lb, angle))
    return coords

def get3d(file_path, k):
    coords = get_data(file_path)
    bounds = []
    for i in range(len(coords)):
        p1, p2 = coords[i], coords[(i + 1) % len(coords)]
        dx, dz = p2.x - p1.x, p2.z - p1.z
        h = math.sqrt(dx**2 + dz**2)
        if h == 0: continue
        
        bounds.append({
            "c":     (p1.x, p1.y, p1.z),
            "left":  (round(p1.x - (k/h) * dz, 3), p1.y, round(p1.z + (k/h) * dx, 3)),
            "right": (round(p1.x + (k/h) * dz, 3), p1.y, round(p1.z - (k/h) * dx, 3))
        })
    return bounds
