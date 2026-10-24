"""Portable project dependencies; importing never invokes a historical driver."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    path = ROOT / 'config' / 'local.json'
    data = json.loads(path.read_text()) if path.is_file() else {}
    project = Path(os.environ.get('MAGTHERMO_PROJECT_ROOT', data.get('project_root', ROOT.parent)))
    return {**data, 'project_root': project.resolve()}


def base_root():
    config = configuration()
    return config['project_root'] / config.get('base_relative', 'STAGE6_E1_WP0_WP1_2026-10-22')
