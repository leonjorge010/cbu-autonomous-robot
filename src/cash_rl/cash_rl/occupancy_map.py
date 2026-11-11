"""SLAM occupancy map loading and geodesic (around-the-walls) goal distances.

ROS-free, shared by HallwayRaycastEnv (raycasting, spawn sampling) and HallwayEnv
(reward). The map frame is aligned with the Gazebo world frame, since SLAM was
started at the spawn pose.
"""
import os

from cash_rl import spec
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import yaml

DEFAULT_MAP_ENV_VAR = 'CASH_MAP_YAML'
DEFAULT_MAP_RELPATH = os.path.join('maps', 'hallway.yaml')
# Cells whose center is at least this far from any non-free cell are traversable
# for geodesic distances. Below the footprint's half-width + collision margin
# (0.295 m), so any collision-free robot center lies on a traversable cell.
TRAVERSABLE_CLEARANCE = 0.25


def resolve_map_yaml(name_or_path):
    """Return the .yaml path for a map path or a name in <workspace>/maps, e.g. 'cbu_floor2'."""
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    maps_dir = os.path.dirname(default_map_yaml_in_workspace())
    candidate = os.path.join(maps_dir, name_or_path + ('' if name_or_path.endswith('.yaml')
                                                       else '.yaml'))
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f'No map {name_or_path!r} (not a file, not in {maps_dir})')


def is_hallway_map(yaml_path):
    """Return True for maps/hallway.yaml, which spec.DOOR_GOALS and junctions assume."""
    return os.path.basename(os.path.realpath(yaml_path)) == os.path.basename(DEFAULT_MAP_RELPATH)


def keepout_yaml_for(yaml_path):
    """Return the <name>_keepout.yaml next to a map (from build_floorplan_world.py), or None."""
    stem, ext = os.path.splitext(yaml_path)
    candidate = f'{stem}_keepout{ext}'
    return candidate if os.path.isfile(candidate) else None


def load_free_mask(yaml_path):
    """Return (meta, free mask with row 0 at the bottom, raw pixels) of a map_server map."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    image_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
    pixels = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
    occupancy = pixels / 255.0 if meta.get('negate', 0) else (255.0 - pixels) / 255.0
    # Flip so row 0 is the bottom (y = origin_y) row, matching world +y up.
    return meta, (occupancy < meta['free_thresh'])[::-1].copy(), pixels[::-1]


def landmarks_for(yaml_path):
    """Return named places {name: (x, y)} to label on a map.

    spec.DOOR_GOALS on the hallway, else the doors in cash_sim's
    worlds/<name>_layout.yaml (from trace_floorplan.py), if there is one.
    """
    if is_hallway_map(yaml_path):
        return dict(spec.DOOR_GOALS)
    maps_dir = os.path.dirname(os.path.realpath(yaml_path))
    stem = os.path.splitext(os.path.basename(yaml_path))[0]
    layout = os.path.join(os.path.dirname(maps_dir), 'src', 'cash_sim', 'worlds',
                          f'{stem}_layout.yaml')
    if not os.path.isfile(layout):
        return {}
    with open(layout) as f:
        doors = yaml.safe_load(f).get('doors', {})
    return {name: (float(v[0]), float(v[1])) for name, v in doors.items()}


def use_map(name_or_path):
    """Resolve a map name/path and make it the default (also for subprocesses); return it."""
    path = resolve_map_yaml(name_or_path)
    os.environ[DEFAULT_MAP_ENV_VAR] = path
    return path


def default_map_yaml():
    """Return $CASH_MAP_YAML, else maps/hallway.yaml found above this source file.

    Works with `colcon build --symlink-install` (the module resolves back into
    src/, under the workspace that holds maps/).
    """
    if os.environ.get(DEFAULT_MAP_ENV_VAR):
        return os.environ[DEFAULT_MAP_ENV_VAR]
    return default_map_yaml_in_workspace()


def default_map_yaml_in_workspace():
    """Return maps/hallway.yaml found above this source file (ignores $CASH_MAP_YAML)."""
    directory = os.path.dirname(os.path.realpath(__file__))
    while True:
        candidate = os.path.join(directory, DEFAULT_MAP_RELPATH)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            raise FileNotFoundError(
                f'Could not find {DEFAULT_MAP_RELPATH} above {__file__}; pass map_yaml '
                f'or set ${DEFAULT_MAP_ENV_VAR}')
        directory = parent


class DistanceField:
    """Geodesic distance (m) to one point, sampled at map cell centers."""

    def __init__(self, grid, occupancy_map):
        self.grid = grid
        self._map = occupancy_map

    def __call__(self, x, y):
        """Bilinearly interpolate at (x, y), ignoring unreachable cells; NaN if none."""
        m = self._map
        # Continuous index where cell centers sit on integers.
        fc = (x - m.origin[0]) / m.resolution - 0.5
        fr = (y - m.origin[1]) / m.resolution - 0.5
        c0, r0 = int(np.floor(fc)), int(np.floor(fr))
        tc, tr = fc - c0, fr - r0
        total = weight_sum = 0.0
        for dr, wr in ((0, 1 - tr), (1, tr)):
            for dc, wc in ((0, 1 - tc), (1, tc)):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < m.height and 0 <= c < m.width and np.isfinite(self.grid[r, c]):
                    total += wr * wc * self.grid[r, c]
                    weight_sum += wr * wc
        return total / weight_sum if weight_sum > 1e-9 else float('nan')


class OccupancyMap:
    """map_server-format map (.yaml + image); anything not free is an obstacle."""

    def __init__(self, yaml_path=None):
        yaml_path = yaml_path or default_map_yaml()
        self.yaml_path = os.path.abspath(yaml_path)
        meta, self.free, pixels = load_free_mask(yaml_path)

        origin_x, origin_y, origin_yaw = meta['origin']
        if origin_yaw != 0:
            raise ValueError(f'{yaml_path}: rotated map origins are not supported')
        self.resolution = float(meta['resolution'])
        self.origin = np.array([origin_x, origin_y], dtype=np.float64)
        self.pixels = pixels.astype(np.uint8)
        self.height, self.width = self.free.shape
        self.free_cells = np.argwhere(self.free)
        # Distance from each cell center to the nearest non-free cell center (m).
        self.clearance = distance_transform_edt(self.free) * self.resolution
        self._build_graph()

    def to_cell(self, x, y):
        col = np.floor((x - self.origin[0]) / self.resolution).astype(np.int64)
        row = np.floor((y - self.origin[1]) / self.resolution).astype(np.int64)
        return row, col

    def cell_center(self, row, col):
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (row + 0.5) * self.resolution)

    def is_free(self, x, y):
        """Vectorized: True where (x, y) is a free cell; off-map counts as blocked."""
        row, col = self.to_cell(np.asarray(x), np.asarray(y))
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        free = np.zeros(row.shape, dtype=bool)
        free[inside] = self.free[row[inside], col[inside]]
        return free

    def _build_graph(self):
        """8-connected graph over traversable cells, edge weights in meters.

        Each edge's length is scaled by the mean of its two cells' wall-cost
        factors (spec.SAFE_CLEARANCE, spec.WALL_COST).
        """
        self.traversable = self.clearance >= TRAVERSABLE_CLEARANCE
        nearness = np.clip(1.0 - self.clearance / spec.SAFE_CLEARANCE, 0.0, None)
        cost = 1.0 + spec.WALL_COST * nearness ** 2
        self._traversable_cells = np.argwhere(self.traversable)
        node_of = np.full(self.free.shape, -1, dtype=np.int64)
        node_of[self.traversable] = np.arange(len(self._traversable_cells))
        rows, cols = self._traversable_cells.T
        src, dst, weight = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r2, c2 = rows + dr, cols + dc
            ok = (r2 >= 0) & (r2 < self.height) & (c2 >= 0) & (c2 < self.width)
            ok[ok] = self.traversable[r2[ok], c2[ok]]
            src.append(node_of[rows[ok], cols[ok]])
            dst.append(node_of[r2[ok], c2[ok]])
            factor = (cost[rows[ok], cols[ok]] + cost[r2[ok], c2[ok]]) / 2
            weight.append(factor * np.hypot(dr, dc) * self.resolution)
        n = len(self._traversable_cells)
        self._graph = csr_matrix(
            (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))), shape=(n, n))
        self._node_of = node_of

    def distance_field(self, x, y, limit=np.inf):
        """Wall-cost-weighted geodesic distance from every traversable cell to (x, y).

        inf on cells that are not traversable, or (to save time on big maps) farther
        than `limit`; cells within it get the same distances as without one.
        """
        row, col = self.to_cell(np.asarray(x), np.asarray(y))
        if (0 <= row < self.height and 0 <= col < self.width
                and self.traversable[row, col]):
            source = self._node_of[row, col]
        else:  # Snap to the nearest traversable cell.
            centers = np.stack(self.cell_center(*self._traversable_cells.T), axis=1)
            source = int(np.argmin(np.hypot(centers[:, 0] - x, centers[:, 1] - y)))
        source_xy = self.cell_center(*self._traversable_cells[source])
        distances = dijkstra(self._graph, directed=False, indices=source, limit=limit)
        grid = np.full(self.free.shape, np.inf)
        grid[self.traversable] = distances + np.hypot(source_xy[0] - x, source_xy[1] - y)
        return DistanceField(grid, self)
