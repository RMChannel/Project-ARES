import struct
import math
from operator import itemgetter


class Cordinates:
    def __init__(self, x, y, z, dist, id, speed, direction, right_bound, left_bound, angle):
        self.x = x
        self.y = y
        self.z = z
        self.dist = dist
        self.id = id
        self.speed = speed  # Velocità target AI (km/h)
        self.direction = direction
        self.right_bound = right_bound
        self.left_bound = left_bound
        self.angle = angle
    def __str__(self):
        return f"({self.x}||{self.y}||{self.z})||{self.direction}||{self.right_bound}||{self.left_bound}||{self.angle}"

# Initialize accumulators
def get_data(nome_file):
    dir_real = 0.0
    lista_coordinate = []
    with open(nome_file, "rb") as buffer:
        # 1. Parse Header
        # Assuming: magic_number, count, unknown1, unknown2
        header, detail_count, u1, u2 = struct.unpack("<4i", buffer.read(16))

        # 2. Parse Ideal Data (x, y, z, dist, id)
        # 4 floats (16 bytes) + 1 int (4 bytes) = 20 bytes per entry
        data_ideal = [struct.unpack("<4fi", buffer.read(20)) for _ in range(detail_count)]

        # 3. Parse Detail Data (18 floats)
        # 18 floats (72 bytes) per entry
        data_detail = [struct.unpack("<18f", buffer.read(72)) for _ in range(detail_count)]

    # 4. Processing Loop
    for i in range(detail_count):
        x, y, z, dist, row_id = data_ideal[i]
        
        

        # Extract specific indices from the 18-float detail block
        # itemgetter per estrarre: speed (idx 3), direction (4), bounds (5,6)
        # L'indice 3 tipicamente contiene la velocità target AI in km/h
        speed, direction, right_bound, left_bound = itemgetter(3, 4, 5, 6)(data_detail[i])

        dir_real += direction

        # Calculate angle to previous node (Pathfinding logic)
        prev_idx = i - 1 if i > 0 else detail_count - 1
        prev_x, _, prev_z, _, _ = data_ideal[prev_idx]

        # Angle calculation: atan2(delta_z, delta_x)
        angle = math.atan2(prev_z - z, x - prev_x)
        lista_coordinate.append(Cordinates(x, y, z, dist, row_id, speed, direction, right_bound, left_bound, angle))
    return lista_coordinate
        

if __name__ == "__main__":
    lista_coordinate = get_data("files_ai/fast_lane.ai")
    for coordinate in lista_coordinate:
        print(coordinate.__str__())