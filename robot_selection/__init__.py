# robot_selection module
"""
Robot Selection Module for NL2TDL Framework

This module implements Contribution 2: Optimal Robot Selection
to prevent over-specification (과소비 방지) by selecting the most
appropriate robot based on task requirements.

Key Features:
- Gaussian-based payload scoring to find optimal match
- Reach scoring with exponential growth
- DoF scoring with over-specification penalty
- Weighted scoring system for multi-criteria optimization

Usage:
    from NL2TDL.robot_selection import select_best_robot

    best_robot_id, score, all_scores = select_best_robot(tdl_content)
"""

from .robot_selector import (
    select_best_robot,
    calculate_payload_score,
    calculate_reach_score,
    calculate_dof_score,
    parse_requirements_from_tdl,
    print_selection_report
)

__all__ = [
    'select_best_robot',
    'calculate_payload_score',
    'calculate_reach_score',
    'calculate_dof_score',
    'parse_requirements_from_tdl',
    'print_selection_report'
]

__version__ = '1.0.0'
__author__ = 'NL2TDL Team'
