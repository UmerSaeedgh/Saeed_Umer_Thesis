"""Portable paths for this project bundle; does not modify raw inputs."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / 'data'
CORPUS = BASE / 'corpus'
OUT_DIR = PROJECT_ROOT / 'distribution_analysis_v1'
