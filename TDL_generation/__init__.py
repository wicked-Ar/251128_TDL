# __init__.py
"""
NL2TDL TDL Generation Module

This module provides Natural Language to TDL conversion capabilities.
"""

from .tdl_knowledge_base import TDLKnowledgeBase
from .nl2tdl_converter import NL2TDLConverter

__all__ = ['TDLKnowledgeBase', 'NL2TDLConverter']

__version__ = '1.0.0'
__author__ = 'xz robotics project'
__description__ = 'Natural Language to Task Description Language converter'
