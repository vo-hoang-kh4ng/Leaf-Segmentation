"""Flavia label resolution.

The published Flavia archive is a flat folder of 1907 JPEGs whose filename number
encodes the species via the ID ranges below. If instead the images are organised one
directory per species, the directory name wins and this table is never consulted.

The table is transcribed from flavia.sourceforge.net and verified against the archive:
the 32 ranges tile exactly the five contiguous ID blocks present on disk
(1001-1616, 2001-2612, 2616-2675, 3001-3563, 3566-3621) with no gaps and no overlap --
see `python -m src.labels` which re-checks that. `resolve` warns about unmapped IDs
instead of silently dropping images, because a shifted range would otherwise show up
only as mysteriously low accuracy.
"""
from __future__ import annotations

ID_RANGES: list[tuple[int, int, str]] = [
    (1001, 1059, "pubescent bamboo"),
    (1060, 1122, "Chinese horse chestnut"),
    (1123, 1194, "Chinese redbud"),
    (1195, 1267, "true indigo"),
    (1268, 1323, "Japanese maple"),
    (1324, 1385, "Nanmu"),
    (1386, 1437, "castor aralia"),
    (1438, 1496, "goldenrain tree"),
    (1497, 1551, "Chinese cinnamon"),
    (1552, 1616, "Anhui Barberry"),
    (2001, 2050, "Big-fruited Holly"),
    (2051, 2113, "Japanese cheesewood"),
    (2114, 2165, "wintersweet"),
    (2166, 2230, "camphortree"),
    (2231, 2290, "Japan Arrowwood"),
    (2291, 2346, "sweet osmanthus"),
    (2347, 2423, "deodar"),
    (2424, 2485, "maidenhair tree"),
    (2486, 2546, "Crape myrtle"),
    (2547, 2612, "oleander"),
    (2616, 2675, "yew plum pine"),
    (3001, 3055, "Japanese Flowering Cherry"),
    (3056, 3110, "Glossy Privet"),
    (3111, 3175, "Chinese Toon"),
    (3176, 3229, "peach"),
    (3230, 3281, "Ford Woodlotus"),
    (3282, 3334, "trident maple"),
    (3335, 3389, "Beale's barberry"),
    (3390, 3446, "southern magnolia"),
    (3447, 3510, "Canadian poplar"),
    (3511, 3563, "Chinese tulip tree"),
    (3566, 3621, "tangerine"),
]


def from_id(image_id: int) -> str | None:
    for low, high, species in ID_RANGES:
        if low <= image_id <= high:
            return species
    return None


def _self_check() -> None:
    """Sanity-check the table itself: 32 species, sorted, non-overlapping."""
    assert len(ID_RANGES) == 32, f"expected 32 species, got {len(ID_RANGES)}"
    assert len({s for _, _, s in ID_RANGES}) == 32, "duplicate species name"
    for (_, prev_high, _), (low, _, _) in zip(ID_RANGES, ID_RANGES[1:]):
        assert low > prev_high, f"overlapping ranges at {low}"
    total = sum(high - low + 1 for low, high, _ in ID_RANGES)
    print(f"{len(ID_RANGES)} species, {total} ids covered")


if __name__ == "__main__":
    _self_check()
