"""
Centralized settings for audio analysis pipeline
All thresholds and parameters in one place - Apple-grade discipline
"""

import os
from pathlib import Path

# ============================================================================
# AUDIO PROCESSING SETTINGS (immutable)
# ============================================================================
SAMPLE_RATE = 48000
PITCH_FLOOR = 60
PITCH_CEILING = 400
WINDOW_LENGTH = 0.025  # 25ms
HOP_LENGTH = 0.005     # 5ms
FFT_SIZE = 2048
WINDOW_TYPE = 'hann'
DYNAMIC_RANGE = 50     # dB
PRE_EMPHASIS = 6       # dB
FORMANT_MAX = 5000     # Hz

# ============================================================================
# VALIDATION THRESHOLDS (fail-fast boundaries)
# ============================================================================
MIN_DURATION = 0.1     # seconds
MAX_DURATION = 3600    # 1 hour
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000

# Pitch validation
MIN_VOICED_FRACTION = 0.05  # 5% minimum for speech
MAX_CLIPPING_PERCENT = 1.0  # 1% maximum
MAX_PEAK_DBFS = 0.0        # No clipping allowed
MIN_PITCH_CONFIDENCE = 0.3  # 30% minimum
MAX_NAN_RATIO = 0.5        # 50% maximum NaN in data

# Formant validation
MIN_FORMANT_COVERAGE = 0.2  # 20% minimum when voiced > 60%

# ============================================================================
# FILE PATHS AND STRUCTURE
# ============================================================================
PRAAT_APP_PATH = "/Applications/Praat.app/Contents/MacOS/Praat"

# Output structure (Acceptance Checklist compliant)
OUTPUT_STRUCTURE = {
    'report.png': 'Main visual report',
    'report.pdf': 'PDF version of report', 
    'report.html': 'Interactive HTML report',
    'manifest.json': 'Analysis metadata and validation results',
    'data/pitch.csv': 'Pitch tracking results',
    'data/formants.csv': 'Formant analysis results',
    'data/ltas.csv': 'Long-term average spectrum',
    'data/metrics.json': 'Computed metrics',
    'data/spectrogram.npz': 'Spectrogram data for verification',
    'logs/ffmpeg.log': 'Audio preprocessing log',
    'logs/praat.log': 'Praat analysis log',
    'logs/pipeline.log': 'Pipeline execution log'
}

# Required CSV columns (contracts)
PITCH_COLUMNS = ['time_s', 'f0_hz', 'voiced', 'method', 'confidence']
FORMANTS_COLUMNS = ['time_s', 'f1_hz', 'f2_hz', 'f3_hz']
LTAS_COLUMNS = ['freq_hz', 'db']

# Required metrics (contracts)
REQUIRED_METRICS = [
    'duration_sec', 'sample_rate_hz', 'rms_dbfs', 'peak_dbfs',
    'clipping_percent', 'voiced_fraction', 'f0_median_hz',
    'spectral_centroid_hz_mean', 'spectral_flatness_mean',
    'ltas_body_db', 'ltas_presence_db', 'ltas_sibilance_db',
    'data_quality_score'
]

# ============================================================================
# BAND DEFINITIONS (frequency analysis)
# ============================================================================
FREQUENCY_BANDS = {
    'body': (120, 250),        # Body/warmth
    'presence': (2000, 6000),  # Presence/clarity  
    'sibilance': (6000, 9000), # Sibilance/harshness
    'air': (10000, 20000)      # Air/brilliance
}

# ============================================================================
# QUALITY THRESHOLDS (Apple-grade standards)
# ============================================================================
QUALITY_THRESHOLDS = {
    'excellent': {
        'voiced_fraction': 0.7,
        'pitch_confidence': 0.8,
        'clipping_percent': 0.1,
        'snr_estimate': 20
    },
    'good': {
        'voiced_fraction': 0.5,
        'pitch_confidence': 0.6,
        'clipping_percent': 0.5,
        'snr_estimate': 15
    },
    'acceptable': {
        'voiced_fraction': 0.3,
        'pitch_confidence': 0.4,
        'clipping_percent': 1.0,
        'snr_estimate': 10
    }
}

# ============================================================================
# SELFTEST PARAMETERS
# ============================================================================
SELFTEST_SIGNALS = {
    'sine_100hz': {
        'frequency': 100,
        'duration': 1.0,
        'expected_f0': (95, 105),
        'expected_voiced': 0.9
    },
    'harmonics': {
        'fundamental': 150,
        'harmonics': [150, 300, 450, 600],
        'duration': 1.0,
        'expected_f0': (145, 155)
    },
    'silence': {
        'duration': 1.0,
        'expected_voiced': 0.0,
        'expected_rms': (-60, -40)  # dBFS
    },
    'white_noise': {
        'duration': 1.0,
        'expected_voiced': 0.0,
        'expected_flatness': 0.2
    }
}

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================
def get_praat_path():
    """Find Praat executable with fallbacks"""
    candidates = [
        PRAAT_APP_PATH,
        "/usr/local/bin/praat",
        "/opt/homebrew/bin/praat"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    return None

def validate_dependencies():
    """Validate all required dependencies"""
    issues = []
    
    # Check Praat
    if not get_praat_path():
        issues.append("Praat not found in standard locations")
    
    # Check ffmpeg
    if os.system("which ffmpeg > /dev/null 2>&1") != 0:
        issues.append("ffmpeg not found in PATH")
    
    # Check Python packages
    try:
        import numpy, scipy, soundfile, matplotlib, pandas
    except ImportError as e:
        issues.append(f"Missing Python package: {e}")
    
    return issues
