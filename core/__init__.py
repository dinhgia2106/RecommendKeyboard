"""
Core modules cho bàn phím AI tiếng Việt
AI-powered Vietnamese Keyboard System
"""

# Core module for Vietnamese Keyboard AI
# AI-powered components only

from .text_processor import TextProcessor
from .ai_recommender import AIRecommender, get_ai_recommender

__all__ = [
    'TextProcessor',
    'AIRecommender',
    'get_ai_recommender'
] 