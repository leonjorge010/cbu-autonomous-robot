"""Occupancy map loading, keep-out zones, named goals and geodesic goal distances.

ROS-free, shared by HallwayRaycastEnv (raycasting, spawn sampling) and HallwayEnv
(reward). The map frame is aligned with the Gazebo world frame.

A map MAP.yaml (map_server format) may have two sidecar files next to it:
  MAP_keepout.yaml  Nav2 keepout-filter mask on the same grid: occupied cells are
                    open floor the robot must not enter (learning zones, stairs).
  MAP_goals.yaml    spawn: [x, y, yaw], the default start pose (the world's spawn);
                    goals: {name: [x, y] or [x, y, yaw]}, the named evaluation goals;
                    branch_min_y: optional, for junction_prob (cells at or above it
                    count as the side branch; hallway only).
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


def default_map_yaml():
    """Return $CASH_MAP_YAML, else maps/hallway.yaml found above this source file.

    Works with `colcon build --symlink-install` (the module resolves back into
    src/, under the workspace that holds maps/).
    """
    if os.environ.get(DEFAULT_MAP_ENV_VAR):
        return os.environ[DEFAULT_MAP_ENV_VAR]
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

    def __init__(self, grid, occupancy_map, goal):
        self.grid = grid
        self._map = occupancy_map
        self.goal = np.asarray(goal, dtype=np.float64)

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

    def waypoint(self, x, y, lookahead):
        """Point about `lookahead` m (path length) along the cheapest route from (x, y).

        Follows the field downhill cell by cell, so the route rounds corners as the
        reward's does. Returns the goal itself when it is closer than that, or when
        (x, y) is off the traversable cells.
        """
        m = self._map
        row, col = (int(v) for v in m.to_cell(x, y))
        if not (0 <= row < m.height and 0 <= col < m.width) \
                or not np.isfinite(self.grid[row, col]):
            return self.goal.copy()
        downhill = self._downhill()
        index, length = row * m.width + col, 0.0
        while length < lookahead:
            next_index = int(downhill[index])
            if next_index == index:
                return self.goal.copy()  # reached the goal's cell
            step = abs(next_index - index)
            length += m.resolution * (1.0 if step in (1, m.width) else np.sqrt(2.0))
            index = next_index
        return np.array(m.cell_center(*divmod(index, m.width)))

    def _downhill(self):
        """Flat index of each cell's lowest 8-neighbor (itself if none is lower); cached."""
        if getattr(self, '_downhill_cache', None) is None:
            h, w = self.grid.shape
            padded = np.full((h + 2, w + 2), np.inf)
            padded[1:-1, 1:-1] = self.grid
            best = self.grid.copy()
            index = np.arange(h * w).reshape(h, w)
            downhill = index.copy()
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbor = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                lower = neighbor < best
                best = np.where(lower, neighbor, best)
                downhill = np.where(lower, index + dr * w + dc, downhill)
            self._downhill_cache = downhill.ravel()
        return self._downhill_cache


def _load_grid(yaml_path):
    """(meta, pixels, occupancy) of a map_server map; rows top-down as in the image."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    image_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
    pixels = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
    occupancy = pixels / 255.0 if meta.get('negate', 0) else (255.0 - pixels) / 255.0
    if meta['origin'][2] != 0:
        raise ValueError(f'{yaml_path}: rotated map origins are not supported')
    return meta, pixels, occupancy


def sidecar_path(yaml_path, suffix):
    """MAP.yaml -> MAP_<suffix>.yaml."""
    return f'{os.path.splitext(yaml_path)[0]}_{suffix}.yaml'


class OccupancyMap:
    """map_server-format map (.yaml + image); anything not free is an obstacle.

    Also loads the optional keep-out and goals sidecars (see the module docstring):
    keepout (bool grid), allowed (free and not keep-out), spawn ((x, y, yaw) or
    None), goals ({name: (x, y)}), goal_yaws ({name: yaw or None}), branch_min_y.
    """

    def __init__(self, yaml_path=None):
        yaml_path = yaml_path or default_map_yaml()
        self.yaml_path = os.path.abspath(yaml_path)
        self.name = os.path.splitext(os.path.basename(yaml_path))[0]
        meta, pixels, occupancy = _load_grid(yaml_path)
        self.resolution = float(meta['resolution'])
        self.origin = np.array(meta['origin'][:2], dtype=np.float64)
        # Flip so row 0 is the bottom (y = origin_y) row, matching world +y up.
        self.free = (occupancy < meta['free_thresh'])[::-1].copy()
        self.pixels = pixels[::-1].astype(np.uint8)
        self.height, self.width = self.free.shape
        self.free_cells = np.argwhere(self.free)

        self.keepout = np.zeros_like(self.free)
        keepout_yaml = sidecar_path(yaml_path, 'keepout')
        if os.path.isfile(keepout_yaml):
            k_meta, _, k_occupancy = _load_grid(keepout_yaml)
            if (k_occupancy.shape != self.free.shape
                    or float(k_meta['resolution']) != self.resolution
                    or not np.allclose(k_meta['origin'][:2], self.origin)):
                raise ValueError(f'{keepout_yaml} is not on the same grid as {yaml_path}')
            self.keepout = (k_occupancy > k_meta['occupied_thresh'])[::-1].copy()
        self.allowed = self.free & ~self.keepout
        self.allowed_cells = np.argwhere(self.allowed)

        self.spawn, self.goals, self.goal_yaws, self.branch_min_y = None, {}, {}, None
        goals_yaml = sidecar_path(yaml_path, 'goals')
        if os.path.isfile(goals_yaml):
            with open(goals_yaml) as f:
                goals = yaml.safe_load(f) or {}
            if goals.get('spawn') is not None:
                self.spawn = tuple(float(v) for v in goals['spawn'])
            for name, pose in (goals.get('goals') or {}).items():
                self.goals[name] = (float(pose[0]), float(pose[1]))
                self.goal_yaws[name] = float(pose[2]) if len(pose) > 2 else None
            self.branch_min_y = goals.get('branch_min_y')
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

    def in_keepout(self, x, y):
        """Vectorized: True where (x, y) is in a keep-out cell."""
        row, col = self.to_cell(np.asarray(x), np.asarray(y))
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        keepout = np.zeros(row.shape, dtype=bool)
        keepout[inside] = self.keepout[row[inside], col[inside]]
        return keepout

    def _build_graph(self):
        """8-connected graph over traversable cells, edge weights in meters.

        Keep-out cells are not traversable, so routes go around keep-out zones.
        Each edge's length is scaled by the mean of its two cells' wall-cost
        factors (spec.SAFE_CLEARANCE, spec.WALL_COST).
        """
        self.traversable = (self.clearance >= TRAVERSABLE_CLEARANCE) & ~self.keepout
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

    def distance_field(self, x, y):
        """Wall-cost-weighted geodesic distance from every traversable cell to (x, y).

        inf on cells that are not traversable.
        """
        row, col = self.to_cell(np.asarray(x), np.asarray(y))
        if (0 <= row < self.height and 0 <= col < self.width
                and self.traversable[row, col]):
            source = self._node_of[row, col]
        else:  # Snap to the nearest traversable cell.
            centers = np.stack(self.cell_center(*self._traversable_cells.T), axis=1)
            source = int(np.argmin(np.hypot(centers[:, 0] - x, centers[:, 1] - y)))
        source_xy = self.cell_center(*self._traversable_cells[source])
        distances = dijkstra(self._graph, directed=False, indices=source)
        grid = np.full(self.free.shape, np.inf)
        grid[self.traversable] = distances + np.hypot(source_xy[0] - x, source_xy[1] - y)
        return DistanceField(grid, self, (x, y))
