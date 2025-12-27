"""
Схемы данных: валидация метрик, manifest.json структура
Apple-grade: fail fast, never lie about data quality
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from src.config.settings import (
    PITCH_COLUMNS, FORMANTS_COLUMNS, LTAS_COLUMNS, REQUIRED_METRICS,
    MIN_DURATION, MAX_DURATION, MIN_VOICED_FRACTION, MAX_CLIPPING_PERCENT,
    MAX_PEAK_DBFS, MIN_PITCH_CONFIDENCE, MAX_NAN_RATIO, MIN_FORMANT_COVERAGE,
    PITCH_FLOOR, PITCH_CEILING
)


class ValidationError(Exception):
    """Raised when validation fails - Apple-grade: fail fast"""
    pass


@dataclass
class Warning:
    """Structured warning with evidence"""
    code: str
    severity: str  # 'INFO', 'WARN', 'ERROR'
    message: str
    evidence: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class InputValidation:
    """Input file validation results"""
    file_exists: bool
    file_size: int
    sha256: str
    container_type: str
    duration_sec: float
    sample_rate: int
    channels: int
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'InputValidation':
        """Create InputValidation from file analysis"""
        import soundfile as sf
        import os
        
        if not file_path.exists():
            return cls(
                file_exists=False, file_size=0, sha256="", container_type="",
                duration_sec=0, sample_rate=0, channels=0
            )
        
        file_size = os.path.getsize(file_path)
        sha256 = compute_file_hash(file_path)
        container_type = file_path.suffix.lower()
        
        try:
            # Try to read audio metadata
            info = sf.info(file_path)
            duration_sec = info.duration
            sample_rate = info.samplerate
            channels = info.channels
        except Exception:
            # If soundfile can't read it, assume it's a video file
            duration_sec = 0  # Will be determined after ffmpeg conversion
            sample_rate = 0
            channels = 0
        
        return cls(
            file_exists=True,
            file_size=file_size,
            sha256=sha256,
            container_type=container_type,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels
        )
    
    def validate(self) -> List[Warning]:
        """Validate input file properties"""
        warnings = []
        
        if not self.file_exists:
            raise ValidationError("Input file does not exist")
        
        if self.duration_sec < MIN_DURATION:
            raise ValidationError(f"Duration too short: {self.duration_sec:.3f}s < {MIN_DURATION}s")
        
        if self.duration_sec > MAX_DURATION:
            warnings.append(Warning(
                code="DURATION_LONG",
                severity="WARN", 
                message=f"Very long duration: {self.duration_sec:.1f}s",
                evidence={"duration": self.duration_sec, "max_recommended": 300}
            ))
        
        if self.channels > 1:
            warnings.append(Warning(
                code="MULTICHANNEL",
                severity="INFO",
                message=f"Multi-channel input will be downmixed: {self.channels} channels",
                evidence={"channels": self.channels}
            ))
        
        return warnings


@dataclass 
class PitchValidation:
    """Pitch data validation"""
    csv_exists: bool
    row_count: int
    columns: List[str]
    time_monotonic: bool
    nan_ratio: float
    voiced_fraction: float
    f0_median: float
    f0_range: tuple
    confidence_mean: float
    
    @classmethod
    def from_csv(cls, csv_path: Path) -> 'PitchValidation':
        """Create PitchValidation from CSV file analysis"""
        if not csv_path.exists():
            return cls(
                csv_exists=False, row_count=0, columns=[], time_monotonic=False,
                nan_ratio=1.0, voiced_fraction=0.0, f0_median=0.0, 
                f0_range=(0, 0), confidence_mean=0.0
            )
        
        try:
            df = pd.read_csv(csv_path, sep='\t')
            
            if df.empty:
                return cls(
                    csv_exists=True, row_count=0, columns=list(df.columns),
                    time_monotonic=False, nan_ratio=1.0, voiced_fraction=0.0,
                    f0_median=0.0, f0_range=(0, 0), confidence_mean=0.0
                )
            
            # Check time monotonicity
            time_monotonic = True
            if 'time_s' in df.columns:
                time_monotonic = df['time_s'].is_monotonic_increasing
            
            # Calculate NaN ratio
            nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            # Calculate voiced fraction
            voiced_fraction = 0.0
            if 'voiced' in df.columns:
                voiced_fraction = df['voiced'].mean()
            elif 'f0_hz' in df.columns:
                voiced_fraction = (df['f0_hz'] > 0).mean()
            
            # Calculate F0 statistics
            f0_median = 0.0
            f0_range = (0, 0)
            if 'f0_hz' in df.columns:
                f0_values = df['f0_hz'][df['f0_hz'] > 0]
                if len(f0_values) > 0:
                    f0_median = f0_values.median()
                    f0_range = (f0_values.min(), f0_values.max())
            
            # Calculate confidence
            confidence_mean = 0.0
            if 'confidence' in df.columns:
                confidence_mean = df['confidence'].mean()
            
            return cls(
                csv_exists=True,
                row_count=len(df),
                columns=list(df.columns),
                time_monotonic=time_monotonic,
                nan_ratio=nan_ratio,
                voiced_fraction=voiced_fraction,
                f0_median=f0_median,
                f0_range=f0_range,
                confidence_mean=confidence_mean
            )
            
        except Exception:
            return cls(
                csv_exists=True, row_count=0, columns=[], time_monotonic=False,
                nan_ratio=1.0, voiced_fraction=0.0, f0_median=0.0,
                f0_range=(0, 0), confidence_mean=0.0
            )
    
    def validate(self) -> List[Warning]:
        """Validate pitch CSV data"""
        warnings = []
        
        if not self.csv_exists:
            raise ValidationError("pitch.csv missing")
        
        if self.row_count < 10:
            raise ValidationError(f"Insufficient pitch data: {self.row_count} rows")
        
        # Check required columns
        missing_cols = set(PITCH_COLUMNS) - set(self.columns)
        if missing_cols:
            raise ValidationError(f"Missing pitch columns: {missing_cols}")
        
        if not self.time_monotonic:
            raise ValidationError("Pitch time values not monotonic")
        
        if self.nan_ratio > MAX_NAN_RATIO:
            raise ValidationError(f"Too many NaN values: {self.nan_ratio:.1%} > {MAX_NAN_RATIO:.1%}")
        
        if self.voiced_fraction < MIN_VOICED_FRACTION:
            warnings.append(Warning(
                code="LOW_VOICED_FRACTION",
                severity="WARN",
                message=f"Low voiced fraction: {self.voiced_fraction:.1%}",
                evidence={"voiced_fraction": self.voiced_fraction, "min_expected": MIN_VOICED_FRACTION}
            ))
        
        # Only raise error for zero F0 if we have significant voiced content
        # This allows silence and noise signals to have zero F0 without failing
        if self.f0_median == 0 and self.voiced_fraction > 0.1:
            raise ValidationError("F0 median is zero despite voiced content - pitch tracking failed")
        
        if not (PITCH_FLOOR <= self.f0_median <= PITCH_CEILING):
            warnings.append(Warning(
                code="F0_OUT_OF_RANGE", 
                severity="WARN",
                message=f"F0 median outside expected range: {self.f0_median:.1f} Hz",
                evidence={"f0_median": self.f0_median, "expected_range": [PITCH_FLOOR, PITCH_CEILING]}
            ))
        
        if self.confidence_mean < MIN_PITCH_CONFIDENCE:
            warnings.append(Warning(
                code="LOW_PITCH_CONFIDENCE",
                severity="WARN", 
                message=f"Low pitch confidence: {self.confidence_mean:.2f}",
                evidence={"confidence": self.confidence_mean, "min_expected": MIN_PITCH_CONFIDENCE}
            ))
        
        return warnings


@dataclass
class FormantsValidation:
    """Formants data validation"""
    csv_exists: bool
    row_count: int
    columns: List[str]
    formant_coverage: float
    
    @classmethod
    def from_csv(cls, csv_path: Path) -> 'FormantsValidation':
        """Create FormantsValidation from CSV file analysis"""
        if not csv_path.exists():
            return cls(csv_exists=False, row_count=0, columns=[], formant_coverage=0.0)
        
        try:
            df = pd.read_csv(csv_path, sep='\t')
            
            if df.empty:
                return cls(csv_exists=True, row_count=0, columns=list(df.columns), formant_coverage=0.0)
            
            # Calculate formant coverage (rows with at least one non-zero formant)
            formant_coverage = 0.0
            if all(col in df.columns for col in ['f1_hz', 'f2_hz', 'f3_hz']):
                has_formants = (df['f1_hz'] > 0) | (df['f2_hz'] > 0) | (df['f3_hz'] > 0)
                formant_coverage = has_formants.mean()
            
            return cls(
                csv_exists=True,
                row_count=len(df),
                columns=list(df.columns),
                formant_coverage=formant_coverage
            )
            
        except Exception:
            return cls(csv_exists=True, row_count=0, columns=[], formant_coverage=0.0)
    
    def validate(self) -> List[Warning]:
        """Validate formants CSV data"""
        warnings = []
        
        if not self.csv_exists:
            warnings.append(Warning(
                code="FORMANTS_MISSING",
                severity="WARN",
                message="formants.csv missing - formant analysis skipped",
                evidence={}
            ))
            return warnings
        
        if self.row_count < 10:
            warnings.append(Warning(
                code="INSUFFICIENT_FORMANT_DATA",
                severity="WARN",
                message=f"Insufficient formant data: {self.row_count} rows",
                evidence={"row_count": self.row_count}
            ))
        
        # Check columns
        missing_cols = set(FORMANTS_COLUMNS) - set(self.columns)
        if missing_cols:
            warnings.append(Warning(
                code="MISSING_FORMANT_COLUMNS",
                severity="WARN",
                message=f"Missing formant columns: {missing_cols}",
                evidence={"missing": list(missing_cols)}
            ))
        
        if self.formant_coverage < MIN_FORMANT_COVERAGE:
            warnings.append(Warning(
                code="LOW_FORMANT_COVERAGE",
                severity="WARN",
                message=f"Low formant coverage: {self.formant_coverage:.1%}",
                evidence={"coverage": self.formant_coverage, "min_expected": MIN_FORMANT_COVERAGE}
            ))
        
        return warnings


@dataclass
class LTASValidation:
    """LTAS data validation"""
    csv_exists: bool
    row_count: int
    columns: List[str]
    freq_monotonic: bool
    freq_max: float
    
    @classmethod
    def from_csv(cls, csv_path: Path) -> 'LTASValidation':
        """Create LTASValidation from CSV file analysis"""
        if not csv_path.exists():
            return cls(csv_exists=False, row_count=0, columns=[], freq_monotonic=False, freq_max=0.0)
        
        try:
            df = pd.read_csv(csv_path, sep='\t')
            
            if df.empty:
                return cls(csv_exists=True, row_count=0, columns=list(df.columns), freq_monotonic=False, freq_max=0.0)
            
            # Check frequency monotonicity
            freq_monotonic = True
            freq_max = 0.0
            if 'freq_hz' in df.columns:
                freq_monotonic = df['freq_hz'].is_monotonic_increasing
                freq_max = df['freq_hz'].max()
            
            return cls(
                csv_exists=True,
                row_count=len(df),
                columns=list(df.columns),
                freq_monotonic=freq_monotonic,
                freq_max=freq_max
            )
            
        except Exception:
            return cls(csv_exists=True, row_count=0, columns=[], freq_monotonic=False, freq_max=0.0)
    
    def validate(self) -> List[Warning]:
        """Validate LTAS CSV data"""
        warnings = []
        
        if not self.csv_exists:
            raise ValidationError("ltas.csv missing")
        
        if self.row_count < 10:
            raise ValidationError(f"Insufficient LTAS data: {self.row_count} rows")
        
        # Check columns
        missing_cols = set(LTAS_COLUMNS) - set(self.columns)
        if missing_cols:
            raise ValidationError(f"Missing LTAS columns: {missing_cols}")
        
        if not self.freq_monotonic:
            raise ValidationError("LTAS frequencies not monotonic")
        
        if self.freq_max < 8000:
            warnings.append(Warning(
                code="LIMITED_BANDWIDTH",
                severity="WARN",
                message=f"Limited frequency range: max {self.freq_max:.0f} Hz",
                evidence={"freq_max": self.freq_max, "expected_min": 8000}
            ))
        
        return warnings


@dataclass
class MetricsValidation:
    """Metrics JSON validation"""
    json_exists: bool
    metrics: Dict[str, Any]
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'MetricsValidation':
        """Create MetricsValidation from JSON file analysis"""
        if not json_path.exists():
            return cls(json_exists=False, metrics={})
        
        try:
            with open(json_path, 'r') as f:
                metrics = json.load(f)
            
            return cls(json_exists=True, metrics=metrics)
            
        except Exception:
            return cls(json_exists=True, metrics={})
    
    def validate(self) -> List[Warning]:
        """Validate metrics JSON"""
        warnings = []
        
        if not self.json_exists:
            raise ValidationError("metrics.json missing")
        
        # Check required metrics
        missing_metrics = set(REQUIRED_METRICS) - set(self.metrics.keys())
        if missing_metrics:
            raise ValidationError(f"Missing required metrics: {missing_metrics}")
        
        # Validate specific metrics
        duration = self.metrics.get('duration_sec', 0)
        if duration <= 0:
            raise ValidationError(f"Invalid duration: {duration}")
        
        peak_dbfs = self.metrics.get('peak_dbfs', 0)
        if peak_dbfs > MAX_PEAK_DBFS:
            warnings.append(Warning(
                code="CLIPPING_DETECTED",
                severity="WARN",
                message=f"Peak above 0 dBFS: {peak_dbfs:.1f} dB",
                evidence={"peak_dbfs": peak_dbfs}
            ))
        
        clipping_percent = self.metrics.get('clipping_percent', 0)
        if clipping_percent > MAX_CLIPPING_PERCENT:
            warnings.append(Warning(
                code="EXCESSIVE_CLIPPING",
                severity="WARN", 
                message=f"Clipping detected: {clipping_percent:.1f}%",
                evidence={"clipping_percent": clipping_percent}
            ))
        
        return warnings


@dataclass
class SpectrogramValidation:
    """Spectrogram NPZ validation"""
    npz_exists: bool
    magnitude_shape: tuple
    time_length: int
    freq_length: int
    has_nan_inf: bool
    reference_db: float
    dynamic_range: float
    
    @classmethod
    def from_npz(cls, npz_path: Path) -> 'SpectrogramValidation':
        """Create SpectrogramValidation from NPZ file analysis"""
        if not npz_path.exists():
            return cls(
                npz_exists=False, magnitude_shape=(0, 0), time_length=0, 
                freq_length=0, has_nan_inf=True, reference_db=np.nan, dynamic_range=0
            )
        
        try:
            data = np.load(npz_path)
            
            magnitude = data['magnitude_db']
            time = data['time']
            freq = data['freq']
            reference_db = float(data['reference_db'])
            dynamic_range = float(data['dynamic_range'])
            
            has_nan_inf = np.any(np.isnan(magnitude)) or np.any(np.isinf(magnitude))
            
            return cls(
                npz_exists=True,
                magnitude_shape=magnitude.shape,
                time_length=len(time),
                freq_length=len(freq),
                has_nan_inf=has_nan_inf,
                reference_db=reference_db,
                dynamic_range=dynamic_range
            )
            
        except Exception:
            return cls(
                npz_exists=True, magnitude_shape=(0, 0), time_length=0,
                freq_length=0, has_nan_inf=True, reference_db=np.nan, dynamic_range=0
            )
    
    def validate(self) -> List[Warning]:
        """Validate spectrogram NPZ"""
        warnings = []
        
        if not self.npz_exists:
            raise ValidationError("spectrogram.npz missing")
        
        if self.has_nan_inf:
            raise ValidationError("Spectrogram contains NaN/Inf values")
        
        if len(self.magnitude_shape) != 2:
            raise ValidationError(f"Invalid spectrogram shape: {self.magnitude_shape}")
        
        if self.magnitude_shape[1] != self.time_length:
            raise ValidationError("Spectrogram time dimension mismatch")
        
        if self.magnitude_shape[0] != self.freq_length:
            raise ValidationError("Spectrogram frequency dimension mismatch")
        
        if np.isnan(self.reference_db):
            raise ValidationError("Invalid reference_db")
        
        return warnings


@dataclass
class AnalysisResult:
    """Complete analysis result with validation"""
    status: str  # 'OK', 'WARN', 'FAIL'
    input_validation: InputValidation
    pitch_validation: PitchValidation
    formants_validation: FormantsValidation
    ltas_validation: LTASValidation
    metrics_validation: MetricsValidation
    spectrogram_validation: SpectrogramValidation
    warnings: List[Warning]
    execution_time: float
    
    def to_manifest(self) -> Dict[str, Any]:
        """Convert to manifest.json format"""
        return {
            "status": self.status,
            "input": asdict(self.input_validation),
            "validation": {
                "pitch": asdict(self.pitch_validation),
                "formants": asdict(self.formants_validation), 
                "ltas": asdict(self.ltas_validation),
                "metrics": asdict(self.metrics_validation),
                "spectrogram": asdict(self.spectrogram_validation)
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "execution_time": self.execution_time,
            "data_quality_score": self.compute_quality_score()
        }
    
    def compute_quality_score(self) -> int:
        """Compute overall data quality score 0-100"""
        score = 100
        
        # Deduct for warnings
        for warning in self.warnings:
            if warning.severity == 'ERROR':
                score -= 30
            elif warning.severity == 'WARN':
                score -= 10
            elif warning.severity == 'INFO':
                score -= 2
        
        # Deduct for missing data
        if not self.pitch_validation.csv_exists:
            score -= 50
        if not self.formants_validation.csv_exists:
            score -= 20
        if not self.ltas_validation.csv_exists:
            score -= 20
        
        return max(0, score)


def validate_csv_structure(csv_path: Path, expected_columns: List[str]) -> tuple:
    """Validate CSV file structure and return basic stats"""
    if not csv_path.exists():
        return False, 0, [], False, 0.0
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if time column exists and is monotonic
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        time_monotonic = True
        if time_cols:
            time_data = df[time_cols[0]].dropna()
            time_monotonic = time_data.is_monotonic_increasing
        
        # Calculate NaN ratio for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_ratio = df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols))
        else:
            nan_ratio = 0.0
        
        return True, len(df), list(df.columns), time_monotonic, nan_ratio
        
    except Exception:
        return False, 0, [], False, 1.0


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def validate_npz_structure(npz_path: Path) -> SpectrogramValidation:
    """Validate NPZ spectrogram file"""
    if not npz_path.exists():
        return SpectrogramValidation(
            npz_exists=False, magnitude_shape=(0, 0), time_length=0, 
            freq_length=0, has_nan_inf=True, reference_db=np.nan, dynamic_range=0
        )
    
    try:
        data = np.load(npz_path)
        
        magnitude = data['magnitude_db']
        time = data['time']
        freq = data['freq']
        reference_db = float(data['reference_db'])
        dynamic_range = float(data['dynamic_range'])
        
        has_nan_inf = np.any(np.isnan(magnitude)) or np.any(np.isinf(magnitude))
        
        return SpectrogramValidation(
            npz_exists=True,
            magnitude_shape=magnitude.shape,
            time_length=len(time),
            freq_length=len(freq),
            has_nan_inf=has_nan_inf,
            reference_db=reference_db,
            dynamic_range=dynamic_range
        )
        
    except Exception:
        return SpectrogramValidation(
            npz_exists=True, magnitude_shape=(0, 0), time_length=0,
            freq_length=0, has_nan_inf=True, reference_db=np.nan, dynamic_range=0
        )
