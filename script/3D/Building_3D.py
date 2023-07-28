import numpy as np

class Building:
    def __init__(self, building_index, building_vertex, building_height):
        self.index = building_index
        self.vertex = building_vertex
        self.height = building_height

    def building_3D(self):
        length = len(self.vertex)
        vertex_3D = np.array([])
        for ii in range(length):
            vertex_3D.append((self.vertex[ii], self.height))