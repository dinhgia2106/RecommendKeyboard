"""
Core modules cho bàn phím AI tiếng Việt
AI-powered Vietnamese Keyboard System - Minimal Core
"""

# Core module for Vietnamese Keyboard AI - Production Ready
# Minimal imports after cleanup

from .text_processor import TextProcessor
from .ai_recommender import AIRecommender, get_ai_recommender

__all__ = [
    'TextProcessor',
    'AIRecommender',
    'get_ai_recommender'
]
