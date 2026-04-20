from __future__ import annotations
from itertools import permutations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# ============================================================
# CONFIG: fill these in from your game
# ============================================================

# Your target / goal vector from the gate puzzle
TARGET = (-0.19045813703346, 0.72617454667624, -0.66060292596786)

# You found the starting face symbol:
START_SYMBOL = 52

# Put the 3 corner probe vectors for face 52 here:
# [52, 60, 60, 60, 60, 60, 60, 60]
FACE_TOP = (
    -9.1767359873081e-08, 0.52573121956222, -0.85065074194854,
)

# [52, 62, 62, 62, 62, 62, 62, 62]
FACE_BOTTOM_LEFT = (
    -9.176736059215e-08, 0.93417230562785, -0.35682222940553,
)

# [52, 63, 63, 63, 63, 63, 63, 63]
FACE_BOTTOM_RIGHT = (
    -0.57735012786692, 0.57735035991235, -0.57735031978958,
)


# ============================================================
# Your real inner symbol -> row-major subtriangle mapping
# ============================================================

INNER_SYMBOL_TO_INDEX: Dict[int, int] = {
    60: 0,

    0: 1, 50: 2, 30: 3,

    25: 4, 54: 5, 41: 6, 40: 7, 57: 8,

    12: 9, 10: 10, 1: 11, 52: 12, 18: 13, 42: 14, 53: 15,

    22: 16, 48: 17, 9: 18, 37: 19, 36: 20, 4: 21, 44: 22, 49: 23, 28: 24,

    5: 25, 33: 26, 38: 27, 13: 28, 2: 29, 61: 30, 17: 31, 31: 32, 26: 33, 19: 34, 35: 35,

    55: 36, 16: 37, 7: 38, 8: 39, 6: 40, 24: 41, 27: 42, 39: 43, 20: 44, 34: 45, 23: 46, 15: 47, 47: 48,

    62: 49, 56: 50, 29: 51, 3: 52, 21: 53, 46: 54, 32: 55, 43: 56, 51: 57, 11: 58, 59: 59, 14: 60, 58: 61, 45: 62, 63: 63,
}


# ============================================================
# Math
# ============================================================

@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return (self.dot(self)) ** 0.5

    def normalized(self) -> "Vec3":
        n = self.length()
        if n == 0:
            raise ValueError("Cannot normalize zero vector.")
        return self / n


@dataclass(frozen=True)
class Triangle2D:
    a: Vec2
    b: Vec2
    c: Vec2

    def centroid(self) -> Vec2:
        return (self.a + self.b + self.c) / 3.0


# ============================================================
# Base inner triangle in local face coordinates
# ============================================================

BASE_TRIANGLE = Triangle2D(
    a=Vec2(0.5, 1.0),  # top
    b=Vec2(0.0, 0.0),  # bottom-left
    c=Vec2(1.0, 0.0),  # bottom-right
)


def barycentric_coords_2d(tri: Triangle2D, p: Vec2) -> Tuple[float, float, float]:
    ax, ay = tri.a.x, tri.a.y
    bx, by = tri.b.x, tri.b.y
    cx, cy = tri.c.x, tri.c.y
    px, py = p.x, p.y

    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-15:
        raise ValueError("Degenerate 2D triangle.")

    u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    w = 1.0 - u - v
    return (u, v, w)


def point_in_triangle_2d(tri: Triangle2D, p: Vec2, eps: float = 1e-12) -> bool:
    try:
        u, v, w = barycentric_coords_2d(tri, p)
    except ValueError:
        return False
    return u >= -eps and v >= -eps and w >= -eps


def bary_to_cartesian(tri: Triangle2D, bary: Tuple[float, float, float]) -> Vec2:
    u, v, w = bary
    return tri.a * u + tri.b * v + tri.c * w


def transform_child_triangle(parent: Triangle2D, child_in_base: Triangle2D) -> Triangle2D:
    a_bary = barycentric_coords_2d(BASE_TRIANGLE, child_in_base.a)
    b_bary = barycentric_coords_2d(BASE_TRIANGLE, child_in_base.b)
    c_bary = barycentric_coords_2d(BASE_TRIANGLE, child_in_base.c)

    return Triangle2D(
        a=bary_to_cartesian(parent, a_bary),
        b=bary_to_cartesian(parent, b_bary),
        c=bary_to_cartesian(parent, c_bary),
    )


# ============================================================
# Build the 64 row-major inner triangles
# ============================================================

def generate_row_major_64_triangles() -> List[Triangle2D]:
    """
    Generate the 64 little triangles in row-major order:
      row sizes = 1, 3, 5, ..., 15

    The big triangle is subdivided into an 8x8 triangular lattice.
    """
    n = 8
    tris: List[Triangle2D] = []

    def bary_to_base(i: int, j: int, k: int) -> Vec2:
        # i + j + k must equal n
        u = i / n
        v = j / n
        w = k / n
        return bary_to_cartesian(BASE_TRIANGLE, (u, v, w))

    for row in range(n):
        # In strip `row`, the top line has i = n-row
        # and the bottom line has i = n-row-1.
        i_top = n - row
        i_bot = n - row - 1

        for j in range(row + 1):
            k = row - j

            # lattice points:
            #   p0 ---- p1
            #    \    /
            #      p2
            p0 = bary_to_base(i_top, j, k)
            p1 = bary_to_base(i_top, j + 1, k - 1) if k > 0 else None
            p2 = bary_to_base(i_bot, j, k + 1)
            p3 = bary_to_base(i_bot, j + 1, k)

            # upward triangle always exists
            tris.append(Triangle2D(
                a=p0,
                b=p2,
                c=p3,
            ))

            # downward triangle exists except at far right edge
            if k > 0 and p1 is not None:
                tris.append(Triangle2D(
                    a=p0,
                    b=p1,
                    c=p3,
                ))

    if len(tris) != 64:
        raise RuntimeError(f"Expected 64 triangles, got {len(tris)}")

    return tris


# ============================================================
# 3D projection helpers
# ============================================================

def project_target_into_face(
    target: Vec3,
    top: Vec3,
    bottom_left: Vec3,
    bottom_right: Vec3,
) -> Tuple[Vec3, Tuple[float, float, float], Vec2]:
    """
    Ray from origin through target intersects the plane of the face.
    Then convert that hit point to barycentric coordinates in the 3D face triangle.
    Then map those barycentric coordinates into local 2D face coordinates.
    """
    target = target.normalized()
    a = top
    b = bottom_left
    c = bottom_right

    ab = b - a
    ac = c - a
    normal = ab.cross(ac)

    denom = normal.dot(target)
    if abs(denom) < 1e-15:
        raise ValueError("Target ray is parallel to the face plane.")

    t = normal.dot(a) / denom
    hit = target * t

    v0 = b - a
    v1 = c - a
    v2 = hit - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    det = d00 * d11 - d01 * d01
    if abs(det) < 1e-15:
        raise ValueError("Degenerate face triangle.")

    v = (d11 * d20 - d01 * d21) / det
    w = (d00 * d21 - d01 * d20) / det
    u = 1.0 - v - w

    # Map barycentric weights onto the local base triangle:
    local = bary_to_cartesian(BASE_TRIANGLE, (u, v, w))
    return hit, (u, v, w), local


# ============================================================
# Solver
# ============================================================

class SEInnerSolver:
    def __init__(self, inner_symbol_to_index: Dict[int, int]) -> None:
        self.inner_symbol_to_index = inner_symbol_to_index
        self.inner_triangles = generate_row_major_64_triangles()
        for idx, tri in enumerate(self.inner_triangles):
            area2 = abs(
                (tri.b.x - tri.a.x) * (tri.c.y - tri.a.y)
                - (tri.b.y - tri.a.y) * (tri.c.x - tri.a.x)
            )
            if area2 < 1e-15:
                raise RuntimeError(f"Degenerate inner triangle at index {idx}: {tri}")
        self.index_to_symbol = {idx: sym for sym, idx in inner_symbol_to_index.items()}

    def greedy_symbols_for_local_target(self, local_target: Vec2, depth: int = 7) -> List[int]:
        """
        Repeatedly choose the child triangle that contains the local target point.
        """
        if not point_in_triangle_2d(BASE_TRIANGLE, local_target):
            raise ValueError(
                f"Local target is outside the base triangle: ({local_target.x}, {local_target.y})"
            )

        result: List[int] = []
        current = BASE_TRIANGLE

        for step in range(depth):
            found_idx = None
            found_triangle = None

            for idx, child_base in enumerate(self.inner_triangles):
                child = transform_child_triangle(current, child_base)
                if point_in_triangle_2d(child, local_target):
                    found_idx = idx
                    found_triangle = child
                    break

            if found_idx is None or found_triangle is None:
                raise RuntimeError(f"Could not locate local target at depth {step + 1}")

            result.append(self.index_to_symbol[found_idx])
            current = found_triangle

        return result

    def resolve_sequence_to_local_triangle(self, inner_symbols: Sequence[int]) -> Triangle2D:
        tri = BASE_TRIANGLE
        for sym in inner_symbols:
            idx = self.inner_symbol_to_index[sym]
            tri = transform_child_triangle(tri, self.inner_triangles[idx])
        return tri


# ============================================================
# Main
# ============================================================

def main() -> None:
    target = Vec3(*TARGET)

    corners = [
        ("probe_60", Vec3(*FACE_TOP)),
        ("probe_62", Vec3(*FACE_BOTTOM_LEFT)),
        ("probe_63", Vec3(*FACE_BOTTOM_RIGHT)),
    ]

    solver = SEInnerSolver(INNER_SYMBOL_TO_INDEX)

    print("Target vector:")
    print(f"  ({target.x:.14f}, {target.y:.14f}, {target.z:.14f})")
    print()

    for perm_num, perm in enumerate(permutations(corners), start=1):
        top_name, top = perm[0]
        left_name, bottom_left = perm[1]
        right_name, bottom_right = perm[2]

        try:
            hit, bary, local = project_target_into_face(
                target=target,
                top=top,
                bottom_left=bottom_left,
                bottom_right=bottom_right,
            )

            u, v, w = bary
            inside = (u >= 0 and v >= 0 and w >= 0)

            print(f"Permutation {perm_num}:")
            print(f"  top         <- {top_name}")
            print(f"  bottom_left <- {left_name}")
            print(f"  bottom_right<- {right_name}")
            print(f"  barycentric = top:{u:.12f}, left:{v:.12f}, right:{w:.12f}")
            print(f"  inside face = {inside}")
            print(f"  local point = ({local.x:.12f}, {local.y:.12f})")

            if inside:
                inner_symbols = solver.greedy_symbols_for_local_target(local, depth=7)
                full_sequence = [START_SYMBOL] + inner_symbols
                print(f"  candidate sequence = {full_sequence}")
            else:
                print("  skipped, target not inside this orientation")

        except Exception as exc:
            print(f"  failed: {exc}")

        print()


if __name__ == "__main__":
    main()