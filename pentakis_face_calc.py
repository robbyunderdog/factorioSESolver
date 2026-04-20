from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import acos, sqrt, pi
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Basic vector math
# ============================================================

@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.length()
        if n == 0:
            raise ValueError("Cannot normalize zero vector.")
        return Vec3(self.x / n, self.y / n, self.z / n)

    def angle_deg_to(self, other: "Vec3") -> float:
        a = self.normalized()
        b = other.normalized()
        c = max(-1.0, min(1.0, a.dot(b)))
        return acos(c) * 180.0 / pi

    def rounded(self, places: int = 14) -> Tuple[float, float, float]:
        return (round(self.x, places), round(self.y, places), round(self.z, places))


# ============================================================
# Your target vector
# ============================================================

TARGET = Vec3(
    -0.19045813703346,
     0.72617454667624,
    -0.66060292596786,
)


# ============================================================
# Your researched face-center vectors from star mapping
# ============================================================

KNOWN_FACE_CENTERS: Dict[int, Vec3] = {
    1:  Vec3(-0.73564182776852, -0.64449044863310,  0.20847820715387),
    2:  Vec3( 0.20847820715387,  0.73564182776852,  0.64449044863310),
    3:  Vec3( 0.33732482508860,  0.39831700267993,  0.85296865578697),
    4:  Vec3( 0.39831700267993,  0.85296865578697,  0.33732482508860),
    5:  Vec3(-0.98181527372170,  0.0,              -0.18983879552606),
    6:  Vec3( 0.33732482508860, -0.39831700267993,  0.85296865578697),
    7:  Vec3( -0.20847820715387, 0.73564182776852,  0.64449044863310),
    8:  Vec3(-0.64449044863310, -0.20847820715387, -0.73564182776852),
    9:  Vec3(-0.20847820715387, -0.73564182776852, -0.6444904486331),
    10: Vec3(-0.3373248250886, 0.39831700267993, -0.85296865578697),
    16: Vec3( 0.0,               0.18983879552606,  0.98181527372170),
    19: Vec3( 0.85296865578697,  0.33732482508860,  0.39831700267993),
    21: Vec3(-0.64449044863310, -0.20847820715387,  0.73564182776852),
    23: Vec3( 0.0,              -0.18983879552606, -0.98181527372170),
    25: Vec3(-0.33732482508860,  0.39831700267993,  0.85296865578697),
    26: Vec3( 0.98181527372170,  0.0,               0.18983879552606),
    28: Vec3( 0.39831700267993, -0.85296865578697,  0.33732482508860),
    30: Vec3( 0.20847820715387, -0.73564182776852, -0.64449044863310),
    33: Vec3(-0.39831700267993,  0.85296865578697,  0.33732482508860),
    34: Vec3( 0.73564182776852, -0.64449044863310, -0.20847820715387),
    35: Vec3( 0.98181527372170,  0.0,              -0.18983879552606),
    37: Vec3( 0.73564182776852, -0.64449044863310,  0.20847820715387),
    38: Vec3(-0.18983879552606, -0.98181527372170,  0.0),
    39: Vec3(-0.73564182776852,  0.64449044863310,  0.20847820715387),
    41: Vec3(-0.98181527372170,  0.0,               0.18983879552606),
    43: Vec3(0, 0.18983879552606, -0.9818152737217),
    45: Vec3( 0.85296865578697, -0.33732482508860, -0.39831700267993),
    46: Vec3( 0.39831700267993, -0.85296865578697, -0.33732482508860),
    50: Vec3( 0.33732482508860, -0.39831700267993, -0.85296865578697),
    52: Vec3( -0.20847820715387, 0.73564182776852, -0.6444904486331),
    57: Vec3( 0.73564182776852,  0.64449044863310, -0.20847820715387),
    58: Vec3(-0.85296865578697, -0.33732482508860,  0.39831700267993),
}


# ============================================================
# 60 outer-face geometry from 3 coordinate families
#
# The 60 face centers appear to come from three families:
#   family A: 12 vectors
#   family B: 24 vectors
#   family C: 24 vectors
#
# We generate them using EVEN permutations and all sign choices.
# This gives exactly 12 + 24 + 24 = 60 unique directions.
# ============================================================

FAMILY_A_BASE = (0.0,               0.18983879552606, 0.98181527372170)  # 12
FAMILY_B_BASE = (0.73564182776852,  0.64449044863310, 0.20847820715387)  # 24
FAMILY_C_BASE = (0.39831700267993,  0.85296865578697, 0.33732482508860)  # 24


def permutation_parity(base: Sequence[float], perm: Sequence[float]) -> int:
    """
    Returns 0 for even, 1 for odd.
    Assumes unique values in base/perm.
    """
    idx = [base.index(x) for x in perm]
    inv = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if idx[i] > idx[j]:
                inv += 1
    return inv % 2


def even_permutations(base: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    out = []
    seen = set()
    for perm in permutations(base):
        if perm in seen:
            continue
        seen.add(perm)
        if permutation_parity(list(base), list(perm)) == 0:
            out.append(perm)
    return out


def generate_family(base: Tuple[float, float, float]) -> List[Vec3]:
    """
    Generate one family using even permutations + all sign choices
    on the non-zero coordinates.
    """
    out = set()

    for perm in even_permutations(base):
        nonzero_positions = [i for i, v in enumerate(perm) if abs(v) > 1e-15]

        for signs in product([1.0, -1.0], repeat=len(nonzero_positions)):
            vals = list(perm)
            for idx, sign in zip(nonzero_positions, signs):
                vals[idx] *= sign
            out.add((round(vals[0], 14), round(vals[1], 14), round(vals[2], 14)))

    return [Vec3(*v) for v in sorted(out)]


def generate_all_60_face_centers() -> Dict[str, List[Vec3]]:
    fam_a = generate_family(FAMILY_A_BASE)
    fam_b = generate_family(FAMILY_B_BASE)
    fam_c = generate_family(FAMILY_C_BASE)

    if len(fam_a) != 12:
        raise RuntimeError(f"Family A expected 12 vectors, got {len(fam_a)}")
    if len(fam_b) != 24:
        raise RuntimeError(f"Family B expected 24 vectors, got {len(fam_b)}")
    if len(fam_c) != 24:
        raise RuntimeError(f"Family C expected 24 vectors, got {len(fam_c)}")

    all_unique = {v.rounded() for v in fam_a + fam_b + fam_c}
    if len(all_unique) != 60:
        raise RuntimeError(f"Expected 60 unique face centers, got {len(all_unique)}")

    return {
        "A": fam_a,
        "B": fam_b,
        "C": fam_c,
    }


# ============================================================
# Ranking helpers
# ============================================================

@dataclass
class RankedFace:
    vector: Vec3
    dot: float
    angle_deg: float
    known_symbol: Optional[int]
    family: str


def rank_all_face_centers(
    target: Vec3,
    families: Dict[str, List[Vec3]],
    known_face_centers: Dict[int, Vec3],
) -> List[RankedFace]:
    target_n = target.normalized()

    known_lookup: Dict[Tuple[float, float, float], int] = {
        vec.rounded(): symbol for symbol, vec in known_face_centers.items()
    }

    ranked: List[RankedFace] = []

    for family_name, vectors in families.items():
        for vec in vectors:
            v = vec.normalized()
            ranked.append(
                RankedFace(
                    vector=v,
                    dot=target_n.dot(v),
                    angle_deg=target_n.angle_deg_to(v),
                    known_symbol=known_lookup.get(v.rounded()),
                    family=family_name,
                )
            )

    ranked.sort(key=lambda r: r.dot, reverse=True)
    return ranked


def print_ranked_faces(ranked: List[RankedFace], top_n: int = 15) -> None:
    print(f"Top {top_n} candidate outer faces:\n")
    for i, item in enumerate(ranked[:top_n], start=1):
        symbol_text = str(item.known_symbol) if item.known_symbol is not None else "unknown"
        print(
            f"{i:2d}. symbol={symbol_text:>7}  "
            f"family={item.family}  "
            f"dot={item.dot:.12f}  "
            f"angle={item.angle_deg:8.4f}°  "
            f"vec=({item.vector.x:.14f}, {item.vector.y:.14f}, {item.vector.z:.14f})"
        )


def print_known_symbol_ranking(ranked: List[RankedFace], top_n: int = 10) -> None:
    known_only = [r for r in ranked if r.known_symbol is not None]
    print(f"\nTop {min(top_n, len(known_only))} researched symbols:\n")
    for i, item in enumerate(known_only[:top_n], start=1):
        print(
            f"{i:2d}. symbol={item.known_symbol:>2}  "
            f"dot={item.dot:.12f}  "
            f"angle={item.angle_deg:8.4f}°  "
            f"vec=({item.vector.x:.14f}, {item.vector.y:.14f}, {item.vector.z:.14f})"
        )


def main() -> None:
    families = generate_all_60_face_centers()
    ranked = rank_all_face_centers(TARGET, families, KNOWN_FACE_CENTERS)

    print("Target vector:")
    print(f"  ({TARGET.x:.14f}, {TARGET.y:.14f}, {TARGET.z:.14f})")
    print(f"  norm={TARGET.length():.14f}\n")

    print("Generated family counts:")
    for name, vectors in families.items():
        print(f"  Family {name}: {len(vectors)}")
    print(f"  Total: {sum(len(v) for v in families.values())}\n")

    print_ranked_faces(ranked, top_n=15)
    print_known_symbol_ranking(ranked, top_n=10)

    best = ranked[0]
    print("\nBest geometric candidate overall:")
    print(
        f"  symbol={'unknown' if best.known_symbol is None else best.known_symbol}\n"
        f"  family={best.family}\n"
        f"  dot={best.dot:.12f}\n"
        f"  angle={best.angle_deg:.6f}°\n"
        f"  vec=({best.vector.x:.14f}, {best.vector.y:.14f}, {best.vector.z:.14f})"
    )


if __name__ == "__main__":
    main()