#!/usr/bin/env python3
"""Trace the walkable corridors of a floor plan photo into a layout YAML.

Pipeline: undo the photo's perspective using the building's four outer corners,
scale with the plan's scale bar, threshold the thick wall lines (thin text and
door-swing arcs are removed by a morphological opening), then keep the
not-wall pixels inside hand-placed region rectangles. The rectangle edges sit on
the wall lines, so they close every doorway: rooms stay out of the map. Learning
zones are traced as open space (the LiDAR sees into them) but listed as keep-out.

The output (free-space polygons, keep-out zones and door goals, in meters) is
the source of truth for build_floorplan_world.py. World frame: x east, y north,
origin at the building's outer SW corner.

Usage: trace_floorplan.py PHOTO OUT_YAML [--overlay OUT_PNG]
Needs OpenCV (apt: python3-opencv). Constants below are for docs/cbu_floorplan.jpg,
the TEGR 2nd floor plan.
"""
import argparse
import math

import cv2

import numpy as np

from scipy.ndimage import binary_fill_holes, label

import yaml

# Photo pixels of the building's outer corners, and the world point each maps to.
# Scale from the plan's 0-40 ft bar: 40 ft = 415 px after rectification.
M_PER_PX = 40 * 0.3048 / 415
BUILDING_X = 2086 * M_PER_PX  # east-west extent (m)
BUILDING_Y = 2446 * M_PER_PX  # north-south extent (m)
CORNERS = [  # (photo px), (world m). North points right in the photo.
    ((312, 862), (0.0, 0.0)),                  # SW
    ((2759, 862), (0.0, BUILDING_Y)),          # NW
    ((2737, 2937), (BUILDING_X, BUILDING_Y)),  # NE
    ((293, 2958), (BUILDING_X, 0.0)),          # SE
]
MARGIN = 1.0  # m of photo kept around the building in the rectified raster

# Walkable regions (x0, y0, x1, y1) in meters, edges on wall lines.
CORRIDORS = {
    'south': (8.2, 10.6, 50.4, 13.3),
    'restroom_women_vestibule': (24.5, 13.0, 28.3, 15.25),
    'gallery_2': (8.5, 13.0, 16.45, 20.5),
    'stair_1_landing': (7.3, 13.95, 8.7, 19.55),  # floor in front of Stair 1's flights
    'west': (12.3, 13.0, 15.85, 59.0),
    'west_alcove_228': (11.3, 48.3, 12.5, 54.7),
    'north': (12.3, 58.85, 51.2, 61.15),
    'elevator_2_lobby': (39.3, 56.1, 46.9, 59.0),
}
# Open to the corridor but off limits to the robot. Stair 1 is open to Gallery 2 (no wall or
# door) and one flight goes down, which a 2D LiDAR cannot see.
ZONES = {
    'stair_1': (1.9, 13.95, 7.5, 19.55),
    'learning_zone_2': (31.4, 13.2, 37.45, 20.4),
    'learning_zone_3': (15.8, 33.7, 24.5, 44.6),
    'learning_zone_4': (24.6, 51.2, 35.7, 58.95),
}
# Doors the plan draws swung open into the corridor: clear their leaves (modeled shut).
OPEN_DOOR_LEAVES = [
    (21.6, 12.0, 22.25, 13.1),   # men's restroom
    (41.6, 12.0, 42.25, 13.1),   # stair 1 hall
    (38.8, 58.95, 39.35, 59.9),  # north corridor, south side
]
SEED = (20.0, 12.0)  # a point in the south corridor
# Enclosed dark blobs smaller than this are labels ("D2-6", "GALLERY 2"), not furniture.
MIN_OBSTACLE_AREA = 0.6  # m^2
SIMPLIFY_TOLERANCE = 0.06  # m, polygon simplification

# Goals: corridor centerline in front of each door, facing it. Centerlines from the
# traced wall faces: south corridor y 11.95, west corridor x 14.1, north corridor y 60.0.
S, W, N, E = -math.pi / 2, math.pi, math.pi / 2, 0.0
DOORS = {
    'elevator_1': (9.4, 11.7, W),
    'tegr_201': (9.3, 11.95, S),
    'tegr_202': (14.25, 11.95, S),
    'restroom_men': (21.6, 11.95, N),
    'restroom_women': (26.4, 11.95, N),
    'tegr_203': (29.4, 11.95, S),
    'learning_zone_2': (34.4, 11.95, N),
    'tegr_203b': (39.4, 11.95, S),
    'tegr_204': (45.5, 11.95, S),
    'tegr_205': (49.6, 12.2, E),
    'stair_1': (9.6, 16.8, W),
    'tegr_230': (14.1, 22.4, W),
    'tegr_214': (14.1, 34.5, E),  # doors face Learning Zone 3: goal at the zone's opening
    'learning_zone_3': (14.1, 39.0, E),
    'tegr_229': (14.1, 43.5, W),
    'tegr_216': (14.1, 44.0, E),  # doors face Learning Zone 3: goal at the zone's opening
    'tegr_228b': (14.1, 49.4, W),
    'tegr_228': (14.1, 53.6, W),
    'tegr_226': (14.6, 60.0, N),
    'tegr_225': (26.0, 60.0, N),
    'tegr_224': (28.1, 60.0, N),
    'learning_zone_4': (30.0, 60.0, S),
    'tegr_223': (42.8, 60.0, N),
    'elevator_2': (43.0, 57.4, S),
    'tegr_222': (48.2, 60.0, N),
    'tegr_221': (50.4, 60.0, E),
}


def rectify(photo):
    """Warp the photo into a north-up raster at M_PER_PX; return it and a world->px function."""
    width = int(round((BUILDING_X + 2 * MARGIN) / M_PER_PX))
    height = int(round((BUILDING_Y + 2 * MARGIN) / M_PER_PX))

    def to_px(x, y):
        return (x + MARGIN) / M_PER_PX, (BUILDING_Y + MARGIN - y) / M_PER_PX

    src = np.float32([c[0] for c in CORNERS])
    dst = np.float32([to_px(*c[1]) for c in CORNERS])
    matrix = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(photo, matrix, (width, height), borderValue=(255, 255, 255))
    return image, to_px


def wall_mask(gray):
    """Thick dark lines: local threshold (uneven lighting), then drop thin strokes."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 61, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel) > 0


def rect_mask(shape, rects, to_px):
    mask = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in rects:
        c0, r1 = (int(round(v)) for v in to_px(x0, y0))
        c1, r0 = (int(round(v)) for v in to_px(x1, y1))
        mask[r0:r1, c0:c1] = True
    return mask


def free_space(walls, to_px):
    walls = walls & ~rect_mask(walls.shape, OPEN_DOOR_LEAVES, to_px)
    region = rect_mask(walls.shape, list(CORRIDORS.values()) + list(ZONES.values()), to_px)
    labels, _ = label(region & ~walls)
    col, row = (int(round(v)) for v in to_px(*SEED))
    free = labels == labels[row, col]
    # Fill label-sized holes; keep real obstacles (counters, columns).
    holes, n = label(binary_fill_holes(free) & ~free)
    sizes = np.bincount(holes.ravel(), minlength=n + 1) * M_PER_PX ** 2
    free |= (holes > 0) & (sizes[holes] < MIN_OBSTACLE_AREA)
    # Smooth photo noise off the edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel)
    return free.astype(bool)


def polygons(free):
    """Free-space boundary as (outer, holes) polygons in world meters."""
    contours, hierarchy = cv2.findContours(free.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    eps = SIMPLIFY_TOLERANCE / M_PER_PX
    outer, holes = [], []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        poly = cv2.approxPolyDP(contour, eps, closed=True)[:, 0, :].astype(float)
        if len(poly) < 3:
            continue
        # Contours run through the edge pixels' centers: up to half a pixel (1.5 cm) inside
        # the free area, well under the photo's accuracy.
        pts = [[round(float(c) * M_PER_PX - MARGIN, 3),
                round(BUILDING_Y + MARGIN - float(r) * M_PER_PX, 3)] for c, r in poly]
        (outer if parent < 0 else holes).append(pts)
    if len(outer) != 1:
        raise RuntimeError(f'expected one connected free region, got {len(outer)}')
    return outer[0], holes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('photo')
    parser.add_argument('out_yaml')
    parser.add_argument('--overlay', help='write the traced area over the rectified photo')
    args = parser.parse_args()

    photo = cv2.imread(args.photo)
    if photo is None:
        raise SystemExit(f'cannot read {args.photo}')
    image, to_px = rectify(photo)
    walls = wall_mask(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    free = free_space(walls, to_px)
    outer, holes = polygons(free)

    layout = {
        'description': 'TEGR (CBU Engineering) 2nd floor corridors, traced from '
                       'docs/cbu_floorplan.jpg by src/cash_sim/tools/trace_floorplan.py. '
                       'x east, y north, origin at the outer SW corner; meters.',
        'free_space': {'outer': outer, 'holes': holes},
        'keepout': {k: list(v) for k, v in ZONES.items()},
        'doors': {k: [round(v[0], 2), round(v[1], 2), round(v[2], 4)] for k, v in DOORS.items()},
    }
    with open(args.out_yaml, 'w') as f:
        f.write('# Generated by src/cash_sim/tools/trace_floorplan.py; doors are [x, y, yaw].\n')
        yaml.safe_dump(layout, f, default_flow_style=None, sort_keys=False, width=100)
    area = free.sum() * M_PER_PX ** 2
    print(f'{args.out_yaml}: {area:.0f} m^2 free, outer polygon {len(outer)} vertices, '
          f'{len(holes)} obstacles, {len(DOORS)} doors')

    if args.overlay:
        vis = image.copy()
        vis[free] = (vis[free] * 0.55 + np.array([0, 230, 0]) * 0.45).astype(np.uint8)
        for poly in [outer] + holes:
            pts = np.array([to_px(x, y) for x, y in poly], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        for x0, y0, x1, y1 in ZONES.values():
            p0 = tuple(int(v) for v in to_px(x0, y0))
            p1 = tuple(int(v) for v in to_px(x1, y1))
            cv2.rectangle(vis, p0, p1, (255, 0, 255), 3)
        for name, (x, y, yaw) in DOORS.items():
            c = tuple(int(v) for v in to_px(x, y))
            tip = tuple(int(v) for v in to_px(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw)))
            cv2.circle(vis, c, 9, (255, 80, 0), -1)
            cv2.arrowedLine(vis, c, tip, (255, 80, 0), 4, tipLength=0.4)
            cv2.putText(vis, name.replace('tegr_', ''), (c[0] + 12, c[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 40, 0), 2)
        cv2.imwrite(args.overlay, vis)


if __name__ == '__main__':
    main()
