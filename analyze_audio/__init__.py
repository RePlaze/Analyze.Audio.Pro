"""
Apple-grade audio analysis pipeline
Acceptance Checklist compliant: deterministic, verifiable, fail-fast
"""

__version__ = "1.0.0"
__author__ = "Voice Analysis Pipeline"

from .core import AudioAnalyzer
from .cli import main
from .schema import ValidationError, AnalysisResult

__all__ = ['AudioAnalyzer', 'main', 'ValidationError', 'AnalysisResult']
