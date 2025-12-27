"""
Основной пайплайн анализа: извлечение аудио, Praat, визуализация
Apple-grade execution engine
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.settings.settings import validate_dependencies, OUTPUT_STRUCTURE
from src.settings.validation import (
    ValidationError, Warning, AnalysisResult, InputValidation, 
    PitchValidation, FormantsValidation, LTASValidation, 
    MetricsValidation, SpectrogramValidation,
    validate_csv_structure, compute_file_hash, validate_npz_structure
)
from src.metrics.praat import extract_audio_data
from src.audio.audio import AudioProcessor
from src.output.reports import ReportGenerator


class AudioAnalyzer:
    """Main analysis pipeline with Apple-grade validation"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.report_generator = ReportGenerator()
        self.warnings: List[Warning] = []
        
        # Validate dependencies on initialization
        issues = validate_dependencies()
        if issues:
            raise ValidationError(f"Dependency issues: {issues}")
    
    def analyze_single(self, input_file: Path, output_dir: Path) -> AnalysisResult:
        """Analyze single audio file with full validation"""
        
        start_time = time.time()
        output_dir = Path(output_dir)
        
        try:
            # Stage 1: Input validation (basic file check)
            input_validation = self._validate_input_file(input_file)
            # Don't validate duration yet - will be done after audio preparation
            
            # Stage 2: Audio preprocessing  
            prepared_audio = self._prepare_audio_stage(input_file, output_dir)
            
            # Stage 2b: Update input validation with actual audio properties
            input_validation = self._update_input_validation_with_audio(input_validation, prepared_audio)
            self.warnings.extend(input_validation.validate())
            
            # Stage 3: Praat-based analysis
            self._praat_analysis_stage(prepared_audio, output_dir)
            
            # Stage 4: Spectral analysis
            self._spectral_analysis_stage(prepared_audio, output_dir)
            
            # Stage 5: Data validation
            validations = self._validate_analysis_outputs(output_dir)
            
            # Stage 6: Report generation
            self._generate_reports_stage([prepared_audio], ['audio'], output_dir, 'single')
            
            # Stage 7: Create manifest
            execution_time = time.time() - start_time
            result = self._create_analysis_result(
                input_validation, validations, execution_time
            )
            
            self._write_manifest(output_dir, result)
            
            return result
            
        except Exception as e:
            # Log pipeline failure
            self._log_pipeline_failure(output_dir, str(e), time.time() - start_time)
            raise
    
    def analyze_compare(self, raw_file: Path, processed_file: Path, output_dir: Path) -> AnalysisResult:
        """Compare two audio files with shared reference"""
        
        start_time = time.time()
        output_dir = Path(output_dir)
        
        try:
            # Validate both input files
            raw_validation = self._validate_input_file(raw_file)
            processed_validation = self._validate_input_file(processed_file)
            
            self.warnings.extend(raw_validation.validate())
            self.warnings.extend(processed_validation.validate())
            
            # Prepare both audio files
            raw_audio = self._prepare_audio_stage(raw_file, output_dir, suffix='_raw')
            processed_audio = self._prepare_audio_stage(processed_file, output_dir, suffix='_processed')
            
            # Analyze both files
            self._praat_analysis_stage(raw_audio, output_dir, suffix='_raw')
            self._praat_analysis_stage(processed_audio, output_dir, suffix='_processed')
            
            # Spectral analysis with shared reference
            self._spectral_analysis_stage_compare([raw_audio, processed_audio], output_dir)
            
            # Validation (use raw file as primary)
            validations = self._validate_analysis_outputs(output_dir, suffix='_raw')
            
            # Generate comparison report
            self._generate_reports_stage(
                [raw_audio, processed_audio], 
                ['raw', 'processed'], 
                output_dir, 
                'compare'
            )
            
            # Create result
            execution_time = time.time() - start_time
            result = self._create_analysis_result(
                raw_validation, validations, execution_time
            )
            
            self._write_manifest(output_dir, result)
            
            return result
            
        except Exception as e:
            self._log_pipeline_failure(output_dir, str(e), time.time() - start_time)
            raise
    
    def selftest(self, output_dir: Path) -> Dict[str, bool]:
        """Run comprehensive selftest with synthetic signals"""
        
        from .dsp import create_test_signals
        from .settings import SELFTEST_SIGNALS
        
        output_dir = Path(output_dir)
        test_dir = output_dir / 'selftest'
        test_dir.mkdir(exist_ok=True)
        
        # Create test signals
        test_files = create_test_signals(test_dir)
        
        results = {}
        
        for signal_name, test_file in test_files.items():
            try:
                # Analyze test signal
                signal_output = test_dir / signal_name
                result = self.analyze_single(test_file, signal_output)
                
                # Validate against expected values
                expected = SELFTEST_SIGNALS[signal_name]
                
                if signal_name == 'sine_100hz':
                    # Check F0 is close to 100 Hz
                    metrics_file = signal_output / 'data' / 'metrics.json'
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    f0_median = metrics.get('f0_median_hz', 0)
                    f0_ok = expected['expected_f0'][0] <= f0_median <= expected['expected_f0'][1]
                    
                    voiced_fraction = metrics.get('voiced_fraction', 0)
                    voiced_ok = voiced_fraction >= expected['expected_voiced']
                    
                    results[signal_name] = f0_ok and voiced_ok
                
                elif signal_name == 'silence':
                    # Check voiced fraction is near zero
                    metrics_file = signal_output / 'data' / 'metrics.json'
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    voiced_fraction = metrics.get('voiced_fraction', 1)
                    results[signal_name] = voiced_fraction <= expected['expected_voiced']
                
                elif signal_name == 'white_noise':
                    # Check spectral flatness is high
                    metrics_file = signal_output / 'data' / 'metrics.json'
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    flatness = metrics.get('spectral_flatness_mean', 0)
                    results[signal_name] = flatness >= expected['expected_flatness']
                
                else:
                    # Default: just check that analysis completed
                    results[signal_name] = result.status in ['OK', 'WARN']
                
            except Exception as e:
                results[signal_name] = False
                print(f"Selftest failed for {signal_name}: {e}")
        
        return results
    
    def _validate_input_file(self, input_file: Path) -> InputValidation:
        """Validate input file properties"""
        
        if not input_file.exists():
            raise ValidationError(f"Input file does not exist: {input_file}")
        
        file_size = input_file.stat().st_size
        sha256 = compute_file_hash(input_file)
        
        # Detect container type
        suffix = input_file.suffix.lower()
        if suffix in ['.mp4', '.m4a', '.mov']:
            container_type = 'video'
        elif suffix in ['.wav', '.aiff', '.flac']:
            container_type = 'audio'
        else:
            container_type = 'unknown'
        
        # Get basic audio properties using ffprobe
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', str(input_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Find audio stream
                audio_stream = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream:
                    duration = float(audio_stream.get('duration', 0))
                    sample_rate = int(audio_stream.get('sample_rate', 0))
                    channels = int(audio_stream.get('channels', 0))
                else:
                    duration = float(info.get('format', {}).get('duration', 0))
                    sample_rate = 0
                    channels = 0
            else:
                duration = 0
                sample_rate = 0
                channels = 0
                
        except Exception:
            duration = 0
            sample_rate = 0
            channels = 0
        
        return InputValidation(
            file_exists=True,
            file_size=file_size,
            sha256=sha256,
            container_type=container_type,
            duration_sec=duration,
            sample_rate=sample_rate,
            channels=channels
        )
    
    def _prepare_audio_stage(self, input_file: Path, output_dir: Path, suffix: str = '') -> Path:
        """Audio preparation stage with validation"""
        
        self._log_stage(output_dir, f"Audio preparation{suffix}")
        
        # Create output structure
        self._create_output_structure(output_dir)
        
        # Prepare audio
        prepared_audio = self.audio_processor.prepare_audio(input_file, output_dir)
        
        # Validate prepared audio
        import soundfile as sf
        y, sr = sf.read(prepared_audio)
        
        if len(y) == 0:
            raise ValidationError("Prepared audio is empty")
        
        if sr != self.audio_processor.sample_rate:
            raise ValidationError(f"Sample rate mismatch after preparation: {sr}")
        
        # Rename if suffix provided
        if suffix:
            new_name = prepared_audio.parent / f'audio{suffix}.wav'
            prepared_audio.rename(new_name)
            prepared_audio = new_name
        
        return prepared_audio
    
    def _praat_analysis_stage(self, audio_file: Path, output_dir: Path, suffix: str = '') -> None:
        """Praat analysis stage with validation"""
        
        self._log_stage(output_dir, f"Praat analysis{suffix}")
        
        # Run Praat analysis
        praat_result = extract_audio_data(audio_file, output_dir)
        
        # Rename outputs if suffix provided
        if suffix:
            data_dir = output_dir / 'data'
            for csv_name in ['pitch.csv', 'formants.csv', 'ltas.csv']:
                csv_file = data_dir / csv_name
                if csv_file.exists():
                    new_name = data_dir / f'{csv_name.replace(".csv", f"{suffix}.csv")}'
                    csv_file.rename(new_name)
            
            metrics_file = data_dir / 'metrics.json'
            if metrics_file.exists():
                new_name = data_dir / f'metrics{suffix}.json'
                metrics_file.rename(new_name)
    
    def _spectral_analysis_stage(self, audio_file: Path, output_dir: Path) -> None:
        """Spectral analysis stage"""
        
        self._log_stage(output_dir, "Spectral analysis")
        
        # Compute STFT and save spectrogram
        f, t, magnitude_db, reference_db = self.audio_processor.compute_stft(audio_file)
        
        # Save spectrogram data
        spectrogram_file = output_dir / 'data' / 'spectrogram.npz'
        np.savez(
            spectrogram_file,
            magnitude_db=magnitude_db,
            time=t,
            freq=f,
            reference_db=reference_db,
            dynamic_range=self.audio_processor.dynamic_range
        )
        
        # Compute additional spectral features
        spectral_features = self.audio_processor.compute_spectral_features(audio_file)
        
        # Update metrics.json with spectral features
        metrics_file = output_dir / 'data' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            metrics.update(spectral_features)
            
            # Add SNR estimate
            snr = self.audio_processor.estimate_snr(audio_file)
            metrics['snr_estimate_db'] = snr
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def _spectral_analysis_stage_compare(self, audio_files: List[Path], output_dir: Path) -> None:
        """Spectral analysis for comparison mode with shared reference"""
        
        self._log_stage(output_dir, "Spectral analysis (compare)")
        
        # Compute STFT for both files to find global reference
        stft_results = []
        for audio_file in audio_files:
            f, t, magnitude_db, _ = self.audio_processor.compute_stft(audio_file)
            stft_results.append((f, t, magnitude_db))
        
        # Find global maximum for shared reference
        global_max = max(
            np.power(10, (mag_db.max() + ref_db) / 20) 
            for f, t, mag_db in stft_results
            for ref_db in [20 * np.log10(np.power(10, mag_db / 20).max())]
        )
        shared_reference_db = 20 * np.log10(global_max + 1e-12)
        
        # Save spectrograms with shared reference
        for i, (audio_file, (f, t, magnitude_db)) in enumerate(zip(audio_files, stft_results)):
            suffix = ['_raw', '_processed'][i]
            
            # Recalculate with shared reference
            magnitude_linear = np.power(10, magnitude_db / 20)
            magnitude_db_shared = 20 * np.log10(magnitude_linear + 1e-12) - shared_reference_db
            
            spectrogram_file = output_dir / 'data' / f'spectrogram{suffix}.npz'
            np.savez(
                spectrogram_file,
                magnitude_db=magnitude_db_shared,
                time=t,
                freq=f,
                reference_db=shared_reference_db,
                dynamic_range=self.audio_processor.dynamic_range
            )
            
            # Update metrics
            spectral_features = self.audio_processor.compute_spectral_features(audio_file)
            snr = self.audio_processor.estimate_snr(audio_file)
            
            metrics_file = output_dir / 'data' / f'metrics{suffix}.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                metrics.update(spectral_features)
                metrics['snr_estimate_db'] = snr
                
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
    
    
    def _generate_reports_stage(self, audio_files: List[Path], file_names: List[str], 
                               output_dir: Path, mode: str) -> None:
        """Generate visual reports"""
        
        self._log_stage(output_dir, f"Report generation ({mode})")
        
        # Generate reports
        self.report_generator.create_report(
            audio_files, file_names, output_dir, mode
        )
    
    def _create_analysis_result(self, input_validation: InputValidation, 
                               validations: Dict[str, Any], execution_time: float) -> AnalysisResult:
        """Create final analysis result"""
        
        # Determine overall status
        error_warnings = [w for w in self.warnings if w.severity == 'ERROR']
        warn_warnings = [w for w in self.warnings if w.severity == 'WARN']
        
        if error_warnings:
            status = 'FAIL'
        elif warn_warnings:
            status = 'WARN'
        else:
            status = 'OK'
        
        return AnalysisResult(
            status=status,
            input_validation=input_validation,
            pitch_validation=validations['pitch'],
            formants_validation=validations['formants'],
            ltas_validation=validations['ltas'],
            metrics_validation=validations['metrics'],
            spectrogram_validation=validations['spectrogram'],
            warnings=self.warnings,
            execution_time=execution_time
        )
    
    def _create_output_structure(self, output_dir: Path) -> None:
        """Create required output directory structure"""
        
        output_dir.mkdir(exist_ok=True)
        (output_dir / 'data').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
    
    def _write_manifest(self, output_dir: Path, result: AnalysisResult) -> None:
        """Write manifest.json with complete analysis metadata"""
        
        manifest_file = output_dir / 'manifest.json'
        
        manifest_data = result.to_manifest()
        
        # Add pipeline metadata
        manifest_data.update({
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'pipeline_version': '1.0.0',
            'settings': {
                'sample_rate': self.audio_processor.sample_rate,
                'window_length': self.audio_processor.window_length,
                'hop_length': self.audio_processor.hop_length,
                'fft_size': self.audio_processor.fft_size,
                'dynamic_range': self.audio_processor.dynamic_range
            }
        })
        
        # Convert numpy types to native Python types for JSON serialization
        manifest_data = self._convert_numpy_types(manifest_data)
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(v) for v in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _log_stage(self, output_dir: Path, stage_name: str) -> None:
        """Log pipeline stage execution"""
        
        log_file = output_dir / 'logs' / 'pipeline.log'
        log_file.parent.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {stage_name}\n")
    
    def _log_pipeline_failure(self, output_dir: Path, error: str, execution_time: float) -> None:
        """Log pipeline failure with details"""
        
        log_file = output_dir / 'logs' / 'pipeline.log'
        log_file.parent.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] PIPELINE FAILURE: {error}\n")
            f.write(f"[{timestamp}] Execution time: {execution_time:.2f}s\n")
            f.write(f"[{timestamp}] Warnings count: {len(self.warnings)}\n")
    
    def _validate_input_file(self, input_file: Path) -> InputValidation:
        """Validate input file exists and is readable"""
        return InputValidation.from_file(input_file)
    
    def _prepare_audio_stage(self, input_file: Path, output_dir: Path, suffix: str = '') -> Path:
        """Prepare audio file for analysis"""
        self._create_output_structure(output_dir)
        self._log_stage(output_dir, "Audio preparation")
        return self.audio_processor.prepare_audio(input_file, output_dir)
    
    def _update_input_validation_with_audio(self, input_validation: InputValidation, prepared_audio: Path) -> InputValidation:
        """Update input validation with actual audio properties after preparation"""
        import soundfile as sf
        
        try:
            info = sf.info(prepared_audio)
            # Create updated validation with actual audio properties
            return InputValidation(
                file_exists=input_validation.file_exists,
                file_size=input_validation.file_size,
                sha256=input_validation.sha256,
                container_type=input_validation.container_type,
                duration_sec=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels
            )
        except Exception:
            # If we can't read the prepared audio, something went wrong
            return input_validation
    
    
    def _spectral_analysis_stage_compare(self, audio_files: List[Path], output_dir: Path) -> None:
        """Run spectral analysis for comparison mode"""
        self._log_stage(output_dir, "Spectral analysis (compare)")
        self.audio_processor.compute_spectrogram_compare(audio_files, output_dir)
    
    def _validate_analysis_outputs(self, output_dir: Path, suffix: str = '') -> Dict[str, Any]:
        """Validate all analysis outputs with Apple-grade rigor"""
        
        data_dir = output_dir / 'data'
        
        # Validate pitch data
        pitch_csv = data_dir / f'pitch{suffix}.csv'
        pitch_validation = PitchValidation.from_csv(pitch_csv)
        self.warnings.extend(pitch_validation.validate())
        
        # Validate formants data
        formants_csv = data_dir / f'formants{suffix}.csv'
        formants_validation = FormantsValidation.from_csv(formants_csv)
        self.warnings.extend(formants_validation.validate())
        
        # Validate LTAS data
        ltas_csv = data_dir / f'ltas{suffix}.csv'
        ltas_validation = LTASValidation.from_csv(ltas_csv)
        self.warnings.extend(ltas_validation.validate())
        
        # Validate metrics data
        metrics_json = data_dir / f'metrics{suffix}.json'
        metrics_validation = MetricsValidation.from_json(metrics_json)
        self.warnings.extend(metrics_validation.validate())
        
        # Validate spectrogram data
        spectrogram_npz = data_dir / f'spectrogram{suffix}.npz'
        spectrogram_validation = SpectrogramValidation.from_npz(spectrogram_npz)
        self.warnings.extend(spectrogram_validation.validate())
        
        return {
            'pitch': pitch_validation,
            'formants': formants_validation,
            'ltas': ltas_validation,
            'metrics': metrics_validation,
            'spectrogram': spectrogram_validation
        }
    
    def _generate_reports_stage(self, audio_files: List[Path], file_names: List[str], output_dir: Path, mode: str) -> None:
        """Generate analysis reports"""
        self._log_stage(output_dir, f"Report generation ({mode})")
        self.report_generator.create_report(audio_files, file_names, output_dir, mode)
    
    def _create_analysis_result(
        self, 
        input_validation: InputValidation, 
        validations: Dict[str, Any], 
        execution_time: float
    ) -> AnalysisResult:
        """Create final analysis result with status determination"""
        
        # Determine overall status based on warnings
        error_warnings = [w for w in self.warnings if w.severity == 'ERROR']
        warn_warnings = [w for w in self.warnings if w.severity == 'WARN']
        
        if error_warnings:
            status = 'FAIL'
        elif warn_warnings:
            status = 'WARN'
        else:
            status = 'OK'
        
        return AnalysisResult(
            status=status,
            input_validation=input_validation,
            pitch_validation=validations['pitch'],
            formants_validation=validations['formants'],
            ltas_validation=validations['ltas'],
            metrics_validation=validations['metrics'],
            spectrogram_validation=validations['spectrogram'],
            warnings=self.warnings,
            execution_time=execution_time
        )


def analyze_audio(input_file: Path, output_dir: Path, mode: str = 'single', baseline_file: Optional[Path] = None) -> AnalysisResult:
    """Main entry point for audio analysis"""
    
    analyzer = AudioAnalyzer()
    
    if mode == 'compare' and baseline_file:
        return analyzer.analyze_compare(baseline_file, input_file, output_dir)
    else:
        return analyzer.analyze_single(input_file, output_dir)
