"""Find the workspace's maps/ directory and a world's default spawn pose."""
import os

import yaml


def maps_dir():
    """<workspace>/maps, found above this file (works with --symlink-install)."""
    directory = os.path.dirname(os.path.realpath(__file__))
    while directory != os.path.dirname(directory):
        if os.path.isdir(os.path.join(directory, 'maps')):
            return os.path.join(directory, 'maps')
        directory = os.path.dirname(directory)
    raise FileNotFoundError(f'No maps/ directory above {__file__}; build with --symlink-install')


def default_spawn(world):
    """(x, y, yaw) from maps/<world>_goals.yaml's spawn, else (0, 0, 0)."""
    try:
        path = os.path.join(maps_dir(), f'{world}_goals.yaml')
        with open(path) as f:
            spawn = (yaml.safe_load(f) or {}).get('spawn')
    except FileNotFoundError:
        spawn = None
    return tuple(float(v) for v in spawn) if spawn else (0.0, 0.0, 0.0)
