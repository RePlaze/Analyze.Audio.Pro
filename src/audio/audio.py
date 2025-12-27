"""
DSP операции: STFT, спектрограммы, интерполяция, фильтрация
Apple-grade: vectorized, validated, deterministic
"""

import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import subprocess
import tempfile

from src.settings.settings import (
    SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, FFT_SIZE, WINDOW_TYPE,
    DYNAMIC_RANGE, FREQUENCY_BANDS
)
from src.settings.validation import ValidationError


class AudioProcessor:
    """Handles audio preprocessing and spectral analysis"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.window_length = WINDOW_LENGTH
        self.hop_length = HOP_LENGTH
        self.fft_size = FFT_SIZE
        self.window_type = WINDOW_TYPE
        self.dynamic_range = DYNAMIC_RANGE
    
    def prepare_audio(self, input_file: Path, output_dir: Path) -> Path:
        """Prepare audio file: decode, resample, convert to mono"""
        
        output_file = output_dir / 'data' / 'audio.wav'
        output_file.parent.mkdir(exist_ok=True)
        
        # Log file for ffmpeg
        ffmpeg_log = output_dir / 'logs' / 'ffmpeg.log'
        ffmpeg_log.parent.mkdir(exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', str(input_file),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',  # Mono
            str(output_file)
        ]
        
        try:
            # Run ffmpeg with logging
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log ffmpeg output
            with open(ffmpeg_log, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            if result.returncode != 0:
                raise ValidationError(f"ffmpeg failed: {result.stderr}")
            
            # Validate output file
            if not output_file.exists():
                raise ValidationError("ffmpeg did not create output file")
            
            # Verify audio properties
            y, sr = sf.read(output_file)
            if sr != self.sample_rate:
                raise ValidationError(f"Sample rate mismatch: {sr} != {self.sample_rate}")
            
            if len(y) == 0:
                raise ValidationError("Empty audio file after processing")
            
            return output_file
            
        except subprocess.TimeoutExpired:
            raise ValidationError("ffmpeg timeout - file too large or corrupted")
        except Exception as e:
            raise ValidationError(f"Audio preparation failed: {e}")
    
    def compute_spectrogram(self, audio_file: Path, output_dir: Path, suffix: str = '') -> None:
        """Compute and save spectrogram data"""
        
        # Load audio
        y, sr = sf.read(audio_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)  # Convert to mono
        
        # Compute STFT
        nperseg = int(self.window_length * sr)
        noverlap = int(nperseg - self.hop_length * sr)
        
        f, t, Zxx = signal.stft(
            y, fs=sr,
            window=self.window_type.lower(),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=self.fft_size,
            boundary=None,
            padded=False
        )
        
        # Convert to magnitude in dB
        magnitude = np.abs(Zxx)
        reference_db = 20 * np.log10(magnitude.max() + 1e-12)
        magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
        
        # Save spectrogram data
        spectrogram_file = output_dir / 'data' / f'spectrogram{suffix}.npz'
        np.savez(
            spectrogram_file,
            magnitude_db=magnitude_db,
            time=t,
            freq=f,
            reference_db=reference_db,
            dynamic_range=self.dynamic_range
        )
    
    def compute_spectrogram_compare(self, audio_files: list, output_dir: Path) -> None:
        """Compute spectrograms for comparison with shared reference"""
        
        # Load both audio files
        audio_data = []
        for audio_file in audio_files:
            y, sr = sf.read(audio_file)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            audio_data.append(y)
        
        # Compute spectrograms
        spectrograms = []
        for i, y in enumerate(audio_data):
            nperseg = int(self.window_length * sr)
            noverlap = int(nperseg - self.hop_length * sr)
            
            f, t, Zxx = signal.stft(
                y, fs=sr,
                window=self.window_type.lower(),
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=self.fft_size,
                boundary=None,
                padded=False
            )
            
            magnitude = np.abs(Zxx)
            spectrograms.append((f, t, magnitude))
        
        # Find global reference for consistent scaling
        global_max = max(spec[2].max() for spec in spectrograms)
        reference_db = 20 * np.log10(global_max + 1e-12)
        
        # Save both spectrograms with shared reference
        suffixes = ['_raw', '_processed']
        for i, (f, t, magnitude) in enumerate(spectrograms):
            magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
            
            spectrogram_file = output_dir / 'data' / f'spectrogram{suffixes[i]}.npz'
            np.savez(
                spectrogram_file,
                magnitude_db=magnitude_db,
                time=t,
                freq=f,
                reference_db=reference_db,
                dynamic_range=self.dynamic_range
            )
    
    def compute_stft(self, audio_file: Path, reference_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute STFT with proper windowing and scaling"""
        
        # Load audio
        y, sr = sf.read(audio_file)
        if sr != self.sample_rate:
            raise ValidationError(f"Sample rate mismatch: {sr} != {self.sample_rate}")
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Validate audio data
        if len(y) == 0:
            raise ValidationError("Empty audio data")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValidationError("Audio contains NaN/Inf values")
        
        # Compute STFT parameters
        nperseg = int(self.window_length * sr)
        noverlap = int((self.window_length - self.hop_length) * sr)
        
        # Compute STFT
        f, t, D = signal.stft(
            y, 
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=self.fft_size,
            window=self.window_type,
            boundary=None,
            padded=False
        )
        
        # Convert to magnitude
        magnitude = np.abs(D)
        
        # Validate STFT output
        if np.any(np.isnan(magnitude)) or np.any(np.isinf(magnitude)):
            raise ValidationError("STFT contains NaN/Inf values")
        
        if magnitude.max() == 0:
            raise ValidationError("STFT magnitude is all zeros")
        
        # Compute reference dB
        if reference_db is None:
            reference_db = 20 * np.log10(magnitude.max() + 1e-12)
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
        
        return f, t, magnitude_db, reference_db
    
    def compute_spectral_features(self, audio_file: Path) -> Dict[str, float]:
        """Compute spectral features from audio"""
        
        # Load audio
        y, sr = sf.read(audio_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Compute spectrum
        f, D = signal.periodogram(y, fs=sr, nfft=self.fft_size, window=self.window_type)
        
        # Spectral centroid
        magnitude = np.abs(D)
        spectral_centroid = np.sum(f * magnitude) / np.sum(magnitude)
        
        # Spectral flatness (geometric mean / arithmetic mean)
        # Avoid log(0) by adding small epsilon
        magnitude_safe = magnitude + 1e-12
        geometric_mean = np.exp(np.mean(np.log(magnitude_safe)))
        arithmetic_mean = np.mean(magnitude_safe)
        spectral_flatness = geometric_mean / arithmetic_mean
        
        # Spectral rolloff (95% of energy)
        cumsum_magnitude = np.cumsum(magnitude)
        total_energy = cumsum_magnitude[-1]
        rolloff_idx = np.where(cumsum_magnitude >= 0.95 * total_energy)[0]
        spectral_rolloff = f[rolloff_idx[0]] if len(rolloff_idx) > 0 else f[-1]
        
        return {
            'spectral_centroid_hz_mean': float(spectral_centroid),
            'spectral_flatness_mean': float(spectral_flatness),
            'spectral_rolloff_hz': float(spectral_rolloff)
        }
    
    def compute_ltas_bands(self, ltas_df) -> Dict[str, float]:
        """Compute energy in frequency bands from LTAS data"""
        
        if len(ltas_df) == 0:
            return {f'ltas_{band}_db': 0.0 for band in FREQUENCY_BANDS.keys()}
        
        results = {}
        
        for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
            # Find frequencies in band
            mask = (ltas_df['freq_hz'] >= f_low) & (ltas_df['freq_hz'] <= f_high)
            
            if np.any(mask):
                # Compute mean energy in band
                band_energy = ltas_df.loc[mask, 'db'].mean()
            else:
                band_energy = 0.0
            
            results[f'ltas_{band_name}_db'] = float(band_energy)
        
        return results
    
    def estimate_snr(self, audio_file: Path, silence_threshold_db: float = -40) -> float:
        """Estimate Signal-to-Noise Ratio"""
        
        # Load audio
        y, sr = sf.read(audio_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Convert to dB
        y_db = 20 * np.log10(np.abs(y) + 1e-12)
        
        # Find silence regions (below threshold)
        silence_mask = y_db < silence_threshold_db
        
        if np.sum(silence_mask) < len(y) * 0.1:  # Need at least 10% silence
            return 0.0  # Cannot estimate SNR
        
        # Estimate noise level from silence regions
        noise_level = np.mean(y_db[silence_mask])
        
        # Estimate signal level from non-silence regions
        signal_mask = ~silence_mask
        if np.sum(signal_mask) == 0:
            return 0.0
        
        signal_level = np.mean(y_db[signal_mask])
        
        # SNR = signal - noise
        snr = signal_level - noise_level
        
        return float(snr)


def create_test_signals(output_dir: Path) -> Dict[str, Path]:
    """Create test signals for selftest mode"""
    
    from src.settings.settings import SELFTEST_SIGNALS
    
    output_dir.mkdir(exist_ok=True)
    test_files = {}
    
    for signal_name, params in SELFTEST_SIGNALS.items():
        
        duration = params['duration']
        n_samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        if signal_name == 'sine_100hz':
            # Pure sine wave
            y = 0.5 * np.sin(2 * np.pi * params['frequency'] * t)
            
        elif signal_name == 'harmonics':
            # Harmonic series
            y = np.zeros_like(t)
            for i, freq in enumerate(params['harmonics']):
                amplitude = 0.3 / (i + 1)  # Decreasing amplitude
                y += amplitude * np.sin(2 * np.pi * freq * t)
            
        elif signal_name == 'silence':
            # Pure silence
            y = np.zeros_like(t)
            
        elif signal_name == 'white_noise':
            # White noise
            y = 0.1 * np.random.randn(n_samples)
        
        else:
            continue
        
        # Save as WAV file
        output_file = output_dir / f'{signal_name}.wav'
        sf.write(output_file, y, SAMPLE_RATE)
        test_files[signal_name] = output_file
    
    return test_files
