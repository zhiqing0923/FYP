import numpy as np

# Threshold values for demographic groups
thresholds = {
    "Malay": {
        "SNA": (81, 87),
        "SNB": (77, 85),
        "ANB": (0, 6),
        "MMPA": (21, 31),
        "LFH": (55, 55),
        "U1A": (108, 120),
        "L1A": (116, 132),
    },
    "Chinese": {
        "SNA": (79, 87),
        "SNB": (76, 84),
        "ANB": (1, 5),
        "MMPA": (21, 31),
        "LFH": (55, 55),
        "U1A": (106, 120),
        "L1A": (90, 100),
    },
    "Indian": {
        "SNA": (79, 85),
        "SNB": (76, 82),
        "ANB": (1, 5),
        "MMPA": (23, 33),
        "LFH": (55, 55),
        "U1A": (103, 115),
        "L1A": (87, 99),
    },
}

class Vector:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
        self.vector = self.end - self.start
        self.magnitude = np.linalg.norm(self.vector)

class Angle:
    def __init__(self, vector1, vector2):
        self.vector1 = vector1
        self.vector2 = vector2

    def theta(self):
        dot_product = np.dot(self.vector1.vector, self.vector2.vector)
        magnitude_product = self.vector1.magnitude * self.vector2.magnitude
        cos_theta = dot_product / magnitude_product
        return np.degrees(np.arccos(cos_theta))

class Distance:
    PIXEL_TO_MM = 0.1  # Conversion factor from pixels to millimeters

    @staticmethod
    def point_to_point(p1, p2):
        return np.linalg.norm(np.array(p2) - np.array(p1)) * Distance.PIXEL_TO_MM

    @staticmethod
    def point_to_vector(point, vector):
        v1 = np.array(point) - vector.start
        cross_product = np.abs(np.cross(v1, vector.vector))
        distance_in_mm = cross_product / vector.magnitude
        return distance_in_mm * Distance.PIXEL_TO_MM

def define_vectors(coords):
    # Define required points and vectors based on given coordinates
    S, N, Or, P, A, B, Pg, Me, Gn, Go, L1, U1, UL, LL, Sn, Pg_prime, PNS, ANS, Ar = coords
    points = {
        "N": N, "S": S, "A": A, "B": B, "Go": Go,
        "Me": Me, "PNS": PNS, "ANS": ANS, "U1": U1, "L1": L1
    }
    vectors = {
        "NS": Vector(N, S), "NA": Vector(N, A), "NB": Vector(N, B),
        "GoMe": Vector(Go, Me), "PNS_ANS": Vector(PNS, ANS),
        "ANS_U1": Vector(ANS, U1), "MeL1": Vector(Me, L1)
    }
    return points, vectors

def calculate_angles_and_distances(normalized_coords):
    points, vectors = define_vectors(normalized_coords)

    SNA = Angle(vectors["NS"], vectors["NA"]).theta()
    SNB = Angle(vectors["NS"], vectors["NB"]).theta()
    ANB = SNA - SNB
    MMPA = Angle(vectors["GoMe"], vectors["PNS_ANS"]).theta()
    
    UFH = Distance.point_to_vector(points["N"], vectors["PNS_ANS"])
    LFH_dist = Distance.point_to_vector(points["Me"], vectors["PNS_ANS"])
    LFH = (LFH_dist / (LFH_dist + UFH)) * 100

    U1A = Angle(vectors["ANS_U1"], vectors["PNS_ANS"]).theta()
    L1A = Angle(vectors["MeL1"], vectors["GoMe"]).theta()
    
    return [SNA, SNB, ANB, MMPA, LFH, U1A, L1A]
