"""
Генерация отчетов: HTML, PDF, PNG, manifest.json
Apple-grade visual output
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy import signal
from scipy.interpolate import interp1d
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.settings.settings import (
    SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, FFT_SIZE, WINDOW_TYPE, DYNAMIC_RANGE
)


# Apple-style visual configuration
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'sans-serif']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 0.5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['grid.alpha'] = 0.2
rcParams['grid.linewidth'] = 0.5
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'


class ReportGenerator:
    """Generates visual reports with Apple-grade quality standards"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.window_length = WINDOW_LENGTH
        self.hop_length = HOP_LENGTH
        self.fft_size = FFT_SIZE
        self.window_type = WINDOW_TYPE
        self.dynamic_range = DYNAMIC_RANGE
    
    def create_report(self, audio_files: List[Path], file_names: List[str], 
                     output_dir: Path, mode: str = 'single') -> None:
        """Create complete report: PNG, PDF, HTML"""
        
        output_dir = Path(output_dir)
        
        if mode == 'compare' and len(audio_files) >= 2:
            self._create_comparison_report(audio_files, file_names, output_dir)
        else:
            self._create_single_report(audio_files[0], file_names[0], output_dir)
        
        # Always create HTML report
        self._create_html_report(output_dir, file_names, mode)
    
    def _create_single_report(self, audio_file: Path, file_name: str, output_dir: Path) -> None:
        """Create single-file analysis report"""
        
        # Load data
        data = self._load_analysis_data(output_dir, file_name)
        
        # Load audio for waveform
        y, sr = sf.read(audio_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        duration = len(y) / sr
        time = np.arange(len(y)) / sr
        
        # Create figure with Apple-grade layout
        fig = plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3,
                              height_ratios=[0.8, 0.5, 2.5, 2.5, 1.5, 1])
        
        # Get colormap
        try:
            magma = plt.colormaps['magma']
        except (AttributeError, KeyError):
            magma = plt.cm.get_cmap('magma')
        
        # ====================================================================
        # SUMMARY CARDS (top row)
        # ====================================================================
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')
        
        self._create_summary_cards(ax_summary, data['metrics'], data['warnings'])
        
        # ====================================================================
        # WAVEFORM + MICRO ZOOM
        # ====================================================================
        ax_wf = fig.add_subplot(gs[1, :])
        ax_wf.plot(time, y, 'k-', linewidth=0.3)
        ax_wf.set_xlim(0, duration)
        ax_wf.set_ylabel('Amplitude', fontsize=10)
        ax_wf.set_title('Waveform', fontsize=12, fontweight='bold')
        ax_wf.grid(True, alpha=0.2)
        
        # Add micro-zoom inset
        self._add_micro_zoom(ax_wf, y, time, data['pitch_df'])
        
        # ====================================================================
        # SPECTROGRAM 0-5 kHz (Formants)
        # ====================================================================
        ax_spec1 = fig.add_subplot(gs[2, :])
        
        if data['spectrogram'] is not None:
            self._plot_spectrogram_with_overlays(
                ax_spec1, data['spectrogram'], data['pitch_df'], data['formants_df'],
                freq_max=5000, title='Spectrogram 0-5 kHz (Formants) + F0/F1/F2/F3',
                magma=magma
            )
        
        # ====================================================================
        # SPECTROGRAM 0-10 kHz (Sibilance)
        # ====================================================================
        ax_spec2 = fig.add_subplot(gs[3, :])
        
        if data['spectrogram'] is not None:
            self._plot_spectrogram_with_overlays(
                ax_spec2, data['spectrogram'], data['pitch_df'], None,
                freq_max=10000, title='Spectrogram 0-10 kHz (Sibilance)',
                magma=magma
            )
        
        # ====================================================================
        # LTAS + METRICS TABLE
        # ====================================================================
        ax_ltas = fig.add_subplot(gs[4, 0])
        self._plot_ltas(ax_ltas, data['ltas_df'])
        
        ax_metrics = fig.add_subplot(gs[4, 1])
        self._create_metrics_table(ax_metrics, data['metrics'])
        
        # ====================================================================
        # QUALITY INDICATORS
        # ====================================================================
        ax_quality = fig.add_subplot(gs[5, :])
        self._create_quality_indicators(ax_quality, data['metrics'], data['warnings'])
        
        # Overall title
        fig.suptitle('Voice Analysis Report', fontsize=16, fontweight='bold', y=0.98)
        
        # Save outputs
        self._save_report_files(fig, output_dir)
    
    def _create_comparison_report(self, audio_files: List[Path], file_names: List[str], 
                                 output_dir: Path) -> None:
        """Create comparison report for two files"""
        
        # Load data for both files
        data1 = self._load_analysis_data(output_dir, file_names[0], suffix='_raw')
        data2 = self._load_analysis_data(output_dir, file_names[1], suffix='_processed')
        
        # Load audio files
        y1, sr1 = sf.read(audio_files[0])
        y2, sr2 = sf.read(audio_files[1])
        
        if len(y1.shape) > 1: y1 = np.mean(y1, axis=1)
        if len(y2.shape) > 1: y2 = np.mean(y2, axis=1)
        
        # Create comparison figure
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25,
                              height_ratios=[0.5, 3, 1.5, 1])
        
        try:
            magma = plt.colormaps['magma']
        except (AttributeError, KeyError):
            magma = plt.cm.get_cmap('magma')
        
        # ====================================================================
        # WAVEFORMS (side by side)
        # ====================================================================
        for i, (y, sr, name) in enumerate([(y1, sr1, file_names[0]), (y2, sr2, file_names[1])]):
            ax = fig.add_subplot(gs[0, i])
            time = np.arange(len(y)) / sr
            ax.plot(time, y, 'k-', linewidth=0.3)
            ax.set_xlim(0, len(y) / sr)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.set_title(f'{name.upper()}: Waveform', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2)
        
        # ====================================================================
        # SPECTROGRAMS (side by side, shared scale)
        # ====================================================================
        for i, (data, name) in enumerate([(data1, file_names[0]), (data2, file_names[1])]):
            ax = fig.add_subplot(gs[1, i])
            
            if data['spectrogram'] is not None:
                self._plot_spectrogram_with_overlays(
                    ax, data['spectrogram'], data['pitch_df'], data['formants_df'],
                    freq_max=10000, title=f'{name.upper()}: Spectrogram 0-10 kHz',
                    magma=magma
                )
        
        # ====================================================================
        # LTAS COMPARISON
        # ====================================================================
        ax_ltas = fig.add_subplot(gs[2, :])
        self._plot_ltas_comparison(ax_ltas, data1['ltas_df'], data2['ltas_df'], file_names)
        
        # ====================================================================
        # METRICS COMPARISON TABLE
        # ====================================================================
        ax_comparison = fig.add_subplot(gs[3, :])
        self._create_comparison_table(ax_comparison, data1['metrics'], data2['metrics'], file_names)
        
        # Overall title
        fig.suptitle('Comparison: RAW vs PROCESSED', fontsize=16, fontweight='bold', y=0.98)
        
        # Save outputs
        self._save_report_files(fig, output_dir)
    
    def _load_analysis_data(self, output_dir: Path, file_name: str, suffix: str = '') -> Dict[str, Any]:
        """Load all analysis data for a file"""
        
        data_dir = output_dir / 'data'
        
        # Load CSV files
        pitch_df = self._load_csv_safe(data_dir / f'pitch{suffix}.csv')
        formants_df = self._load_csv_safe(data_dir / f'formants{suffix}.csv')
        ltas_df = self._load_csv_safe(data_dir / f'ltas{suffix}.csv')
        
        # Load metrics
        metrics = {}
        metrics_file = data_dir / f'metrics{suffix}.json'
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception:
                pass
        
        # Load spectrogram
        spectrogram = None
        spec_file = data_dir / f'spectrogram{suffix}.npz'
        if spec_file.exists():
            try:
                spectrogram = np.load(spec_file)
            except Exception:
                pass
        
        # Load warnings from manifest
        warnings = []
        manifest_file = output_dir / 'manifest.json'
        if manifest_file.exists():
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    warnings = manifest.get('warnings', [])
            except Exception:
                pass
        
        return {
            'pitch_df': pitch_df,
            'formants_df': formants_df,
            'ltas_df': ltas_df,
            'metrics': metrics,
            'spectrogram': spectrogram,
            'warnings': warnings
        }
    
    def _load_csv_safe(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV with error handling"""
        if not csv_path.exists():
            return pd.DataFrame()
        
        try:
            # Try tab-separated first (our format), then comma-separated
            return pd.read_csv(csv_path, sep='\t')
        except Exception:
            try:
                return pd.read_csv(csv_path, sep=',')
            except Exception:
                return pd.DataFrame()
    
    def _create_summary_cards(self, ax, metrics: Dict[str, Any], warnings: List[Dict]) -> None:
        """Create summary cards at top of report"""
        
        ax.axis('off')
        
        # Helper function to format numeric values, handling "--undefined--" strings
        def format_numeric_safe(value, format_spec, unit=""):
            if isinstance(value, str) and value == "--undefined--":
                return "N/A"
            try:
                return f"{float(value):{format_spec}} {unit}".strip()
            except (ValueError, TypeError):
                return "N/A"

        # Define cards data
        cards_data = [
            ('Duration', f"{metrics.get('duration_sec', 0):.2f} s"),
            ('Sample Rate', f"{int(metrics.get('sample_rate_hz', 0))} Hz"),
            ('F0 Median', format_numeric_safe(metrics.get('f0_median_hz', 0), '.0f', 'Hz')),
            ('F0 Range', f"{format_numeric_safe(metrics.get('f0_q10_hz', 0), '.0f')}-{format_numeric_safe(metrics.get('f0_q90_hz', 0), '.0f')} Hz"),
            ('Voiced', f"{metrics.get('voiced_fraction', 0)*100:.0f}%"),
            ('RMS', f"{metrics.get('rms_dbfs', 0):.1f} dBFS"),
            ('Peak', f"{metrics.get('peak_dbfs', 0):.1f} dBFS"),
            ('Body', f"{metrics.get('ltas_body_db', 0):.1f} dB"),
            ('Presence', f"{metrics.get('ltas_presence_db', 0):.1f} dB"),
            ('Warnings', f"{len(warnings)}")
        ]
        
        # Create cards
        n_cards = len(cards_data)
        card_width = 1.0 / n_cards
        
        for i, (label, value) in enumerate(cards_data):
            x_pos = i * card_width
            
            # Card background
            rect = plt.Rectangle((x_pos + 0.01, 0.1), card_width - 0.02, 0.8, 
                               facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=0.5)
            ax.add_patch(rect)
            
            # Label
            ax.text(x_pos + card_width/2, 0.7, label, ha='center', va='center',
                   fontsize=9, color='#6c757d', weight='normal')
            
            # Value
            color = '#dc3545' if label == 'Warnings' and len(warnings) > 0 else '#212529'
            ax.text(x_pos + card_width/2, 0.3, value, ha='center', va='center',
                   fontsize=12, color=color, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _add_micro_zoom(self, ax_main, y: np.ndarray, time: np.ndarray, pitch_df: pd.DataFrame) -> None:
        """Add micro-zoom inset to waveform"""
        
        # Find loudest voiced segment
        zoom_start = self._find_loudest_voiced_segment(y, time, pitch_df)
        zoom_end = zoom_start + 0.2  # 200ms
        
        zoom_mask = (time >= zoom_start) & (time <= zoom_end)
        if not np.any(zoom_mask):
            return
        
        # Create inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_zoom = inset_axes(ax_main, width="30%", height="40%", loc='upper right')
        
        ax_zoom.plot(time[zoom_mask], y[zoom_mask], 'k-', linewidth=1)
        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_title('Detail: 200 ms', fontsize=8)
        ax_zoom.tick_params(labelsize=7)
        ax_zoom.grid(True, alpha=0.3)
    
    def _find_loudest_voiced_segment(self, y: np.ndarray, time: np.ndarray, 
                                    pitch_df: pd.DataFrame) -> float:
        """Find the loudest voiced segment for micro-zoom"""
        
        if len(pitch_df) == 0:
            return len(time) / (2 * self.sample_rate)  # Default to middle
        
        window_samples = int(0.2 * self.sample_rate)
        max_energy = 0
        best_start = len(time) / (2 * self.sample_rate)
        
        for start_sec in np.arange(0, len(time) / self.sample_rate - 0.2, 0.1):
            start_idx = int(start_sec * self.sample_rate)
            end_idx = start_idx + window_samples
            
            if end_idx >= len(y):
                break
            
            segment = y[start_idx:end_idx]
            energy = np.sum(segment ** 2)
            
            # Check if this segment is voiced
            segment_mid = start_sec + 0.1
            
            if 'time_s' in pitch_df.columns and 'f0_hz' in pitch_df.columns:
                pitch_at_mid = pitch_df[
                    pitch_df['time_s'].between(segment_mid - 0.05, segment_mid + 0.05)
                ]
                
                if len(pitch_at_mid) > 0 and pitch_at_mid['f0_hz'].max() > 0:
                    if energy > max_energy:
                        max_energy = energy
                        best_start = start_sec
        
        return best_start
    
    def _plot_spectrogram_with_overlays(self, ax, spectrogram_data, pitch_df: pd.DataFrame, 
                                       formants_df: Optional[pd.DataFrame], freq_max: int,
                                       title: str, magma) -> None:
        """Plot spectrogram with pitch and formant overlays"""
        
        if spectrogram_data is None:
            ax.text(0.5, 0.5, 'Spectrogram data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        
        # Extract spectrogram data
        magnitude_db = spectrogram_data['magnitude_db']
        time_stft = spectrogram_data['time']
        freq = spectrogram_data['freq']
        
        # Filter frequency range
        freq_mask = freq <= freq_max
        freqs_filtered = freq[freq_mask]
        mag_filtered = magnitude_db[freq_mask, :]
        
        # Log-frequency interpolation for better visualization
        freq_log = np.logspace(np.log10(50), np.log10(freq_max), 300)
        
        try:
            interp_func = interp1d(freqs_filtered, mag_filtered, kind='linear', axis=0,
                                 bounds_error=False, fill_value='extrapolate')
            magnitude_log = interp_func(freq_log)
        except Exception:
            # Fallback to original data
            freq_log = freqs_filtered
            magnitude_log = mag_filtered
        
        # Plot spectrogram
        vmin = -self.dynamic_range
        vmax = 0
        
        F, T = np.meshgrid(freq_log, time_stft, indexing='ij')
        im = ax.pcolormesh(T, F, magnitude_log, cmap=magma, vmin=vmin, vmax=vmax, shading='gouraud')
        
        # Add pitch overlay
        if len(pitch_df) > 0 and 'f0_hz' in pitch_df.columns and 'time_s' in pitch_df.columns:
            pitch_valid = pitch_df[pitch_df['f0_hz'] > 0]
            
            if len(pitch_valid) > 5:
                # Smooth pitch for better visualization
                f0_smooth = signal.medfilt(pitch_valid['f0_hz'].values, kernel_size=5)
                ax.plot(pitch_valid['time_s'], f0_smooth, 'w-', linewidth=2, alpha=0.9, label='F0')
        
        # Add formant overlays
        if formants_df is not None and len(formants_df) > 0:
            for formant_num, color in [(1, 'cyan'), (2, 'yellow'), (3, 'magenta')]:
                f_col = f'f{formant_num}_hz'
                
                if f_col in formants_df.columns and 'time_s' in formants_df.columns:
                    formant_valid = formants_df[formants_df[f_col] > 0]
                    
                    if len(formant_valid) > 0:
                        ax.plot(formant_valid['time_s'], formant_valid[f_col],
                               color=color, linewidth=1.5, alpha=0.7, 
                               linestyle='--', label=f'F{formant_num}')
        
        # Configure axes
        ax.set_yscale('log')
        ax.set_ylim(50, freq_max)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='dB', format='%d', fraction=0.046)
        
        # Add legend if overlays exist
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=8)
    
    def _plot_ltas(self, ax, ltas_df: pd.DataFrame) -> None:
        """Plot Long-Term Average Spectrum"""
        
        if len(ltas_df) == 0:
            ax.text(0.5, 0.5, 'LTAS data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LTAS', fontsize=12, fontweight='bold')
            return
        
        # Plot LTAS
        freq_col = 'freq_hz' if 'freq_hz' in ltas_df.columns else 'frequency'
        db_col = 'db' if 'db' in ltas_df.columns else 'dB'
        
        ax.semilogx(ltas_df[freq_col], ltas_df[db_col], 'k-', linewidth=1.5)
        
        # Add frequency band markers
        ax.axvline(200, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(4000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(8000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Labels
        y_pos = ax.get_ylim()[1] * 0.9
        ax.text(200, y_pos, 'Body', fontsize=8, ha='center', alpha=0.7)
        ax.text(4000, y_pos, 'Presence', fontsize=8, ha='center', alpha=0.7)
        ax.text(8000, y_pos, 'Sibilance', fontsize=8, ha='center', alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('dB', fontsize=10)
        ax.set_title('LTAS', fontsize=12, fontweight='bold')
        ax.set_xlim(50, 20000)
        ax.grid(True, alpha=0.2)
    
    def _plot_ltas_comparison(self, ax, ltas1: pd.DataFrame, ltas2: pd.DataFrame, 
                             file_names: List[str]) -> None:
        """Plot LTAS comparison"""
        
        freq_col = 'freq_hz' if 'freq_hz' in ltas1.columns else 'frequency'
        db_col = 'db' if 'db' in ltas1.columns else 'dB'
        
        if len(ltas1) > 0:
            ax.semilogx(ltas1[freq_col], ltas1[db_col], 'b-', linewidth=1.5, 
                       label=file_names[0].upper(), alpha=0.8)
        
        if len(ltas2) > 0:
            ax.semilogx(ltas2[freq_col], ltas2[db_col], 'r-', linewidth=1.5, 
                       label=file_names[1].upper(), alpha=0.8)
        
        # Add frequency band markers
        ax.axvline(200, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(4000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(8000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('dB', fontsize=10)
        ax.set_title('LTAS Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim(50, 20000)
        ax.grid(True, alpha=0.2)
        ax.legend()
    
    def _format_numeric_html(self, value, format_spec):
        """Helper to format numeric values for HTML, handling "--undefined--" strings"""
        if isinstance(value, str) and value == "--undefined--":
            return "N/A"
        try:
            return f"{float(value):{format_spec}}"
        except (ValueError, TypeError):
            return "N/A"
    
    def _format_numeric_diff_html(self, value1, value2, format_spec):
        """Helper to format numeric differences for HTML, handling "--undefined--" strings"""
        if (isinstance(value1, str) and value1 == "--undefined--") or \
           (isinstance(value2, str) and value2 == "--undefined--"):
            return "N/A"
        try:
            diff = float(value1) - float(value2)
            return f"{diff:{format_spec}}"
        except (ValueError, TypeError):
            return "N/A"

    def _create_metrics_table(self, ax, metrics: Dict[str, Any]) -> None:
        """Create metrics table"""
        
        ax.axis('off')
        
        # Helper function to format numeric values, handling "--undefined--" strings
        def format_numeric(value, format_spec, unit=""):
            if isinstance(value, str) and value == "--undefined--":
                return "N/A"
            try:
                return f"{float(value):{format_spec}} {unit}".strip()
            except (ValueError, TypeError):
                return "N/A"
        
        table_data = [
            ['Metric', 'Value'],
            ['Duration', f"{metrics.get('duration_sec', 0):.2f} s"],
            ['Sample Rate', f"{int(metrics.get('sample_rate_hz', 0))} Hz"],
            ['F0 Median', format_numeric(metrics.get('f0_median_hz', 0), '.1f', 'Hz')],
            ['F0 Range', f"{format_numeric(metrics.get('f0_q10_hz', 0), '.0f')}-{format_numeric(metrics.get('f0_q90_hz', 0), '.0f')} Hz"],
            ['Voiced Fraction', f"{metrics.get('voiced_fraction', 0)*100:.1f}%"],
            ['RMS (dBFS)', f"{metrics.get('rms_dbfs', 0):.1f} dB"],
            ['Peak (dBFS)', f"{metrics.get('peak_dbfs', 0):.1f} dB"],
            ['Crest Factor', f"{metrics.get('crest_factor', 0):.2f}"],
            ['SNR Estimate', f"{metrics.get('snr_estimate_db', 0):.1f} dB"],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='left', loc='center', 
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#f8f9fa')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('Metrics', fontsize=12, fontweight='bold', pad=20)
    
    def _create_comparison_table(self, ax, metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                                file_names: List[str]) -> None:
        """Create comparison metrics table"""
        
        ax.axis('off')
        
        table_data = [
            ['Metric', file_names[0].upper(), file_names[1].upper(), 'Δ'],
            ['F0 Median (Hz)', 
             f"{metrics1.get('f0_median_hz', 0):.1f}", 
             f"{metrics2.get('f0_median_hz', 0):.1f}",
             f"{metrics2.get('f0_median_hz', 0) - metrics1.get('f0_median_hz', 0):+.1f}"],
            ['Voiced Fraction', 
             f"{metrics1.get('voiced_fraction', 0)*100:.1f}%", 
             f"{metrics2.get('voiced_fraction', 0)*100:.1f}%",
             f"{(metrics2.get('voiced_fraction', 0) - metrics1.get('voiced_fraction', 0))*100:+.1f}%"],
            ['RMS (dBFS)', 
             f"{metrics1.get('rms_dbfs', 0):.1f}", 
             f"{metrics2.get('rms_dbfs', 0):.1f}",
             f"{metrics2.get('rms_dbfs', 0) - metrics1.get('rms_dbfs', 0):+.1f}"],
            ['Peak (dBFS)', 
             f"{metrics1.get('peak_dbfs', 0):.1f}", 
             f"{metrics2.get('peak_dbfs', 0):.1f}",
             f"{metrics2.get('peak_dbfs', 0) - metrics1.get('peak_dbfs', 0):+.1f}"],
            ['SNR Estimate', 
             f"{metrics1.get('snr_estimate_db', 0):.1f} dB", 
             f"{metrics2.get('snr_estimate_db', 0):.1f} dB",
             f"{metrics2.get('snr_estimate_db', 0) - metrics1.get('snr_estimate_db', 0):+.1f} dB"],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center', 
                        colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#f8f9fa')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('Comparison Metrics', fontsize=12, fontweight='bold', pad=20)
    
    def _create_quality_indicators(self, ax, metrics: Dict[str, Any], warnings: List[Dict]) -> None:
        """Create quality indicators section"""
        
        ax.axis('off')
        
        # Quality score
        quality_score = metrics.get('data_quality_score', 0)
        
        # Determine quality level
        if quality_score >= 80:
            quality_level = 'Excellent'
            quality_color = '#28a745'
        elif quality_score >= 60:
            quality_level = 'Good'
            quality_color = '#ffc107'
        else:
            quality_level = 'Needs Attention'
            quality_color = '#dc3545'
        
        # Quality score display
        ax.text(0.1, 0.7, f'Data Quality Score: {quality_score}/100', 
               fontsize=14, fontweight='bold')
        ax.text(0.1, 0.5, f'Level: {quality_level}', 
               fontsize=12, color=quality_color, fontweight='bold')
        
        # Warnings summary
        if warnings:
            warning_text = f"Warnings ({len(warnings)}):\n"
            for i, warning in enumerate(warnings[:3]):  # Show first 3 warnings
                warning_text += f"• {warning.get('message', 'Unknown warning')}\n"
            
            if len(warnings) > 3:
                warning_text += f"• ... and {len(warnings) - 3} more"
            
            ax.text(0.5, 0.7, warning_text, fontsize=10, 
                   verticalalignment='top', color='#dc3545')
        else:
            ax.text(0.5, 0.6, 'No warnings detected', 
                   fontsize=12, color='#28a745')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _save_report_files(self, fig, output_dir: Path) -> None:
        """Save report in multiple formats"""
        
        # Save PNG (high resolution)
        png_file = output_dir / 'report.png'
        fig.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save PDF (vector format)
        pdf_file = output_dir / 'report.pdf'
        fig.savefig(pdf_file, bbox_inches='tight', facecolor='white')
        
        plt.close(fig)
        
        print(f"✅ Report saved: {png_file}")
    
    def _create_html_report(self, output_dir: Path, file_names: List[str], mode: str) -> None:
        """Create interactive HTML report"""
        
        # Load data for HTML
        if mode == 'compare':
            data1 = self._load_analysis_data(output_dir, file_names[0], suffix='_raw')
            data2 = self._load_analysis_data(output_dir, file_names[1], suffix='_processed')
            primary_data = data1
            comparison_data = data2
        else:
            primary_data = self._load_analysis_data(output_dir, file_names[0])
            comparison_data = None
        
        html_content = self._generate_html_content(
            primary_data, comparison_data, file_names, mode
        )
        
        html_file = output_dir / 'report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report: {html_file}")
    
    def _generate_html_content(self, primary_data: Dict[str, Any], 
                              comparison_data: Optional[Dict[str, Any]], 
                              file_names: List[str], mode: str) -> str:
        """Generate HTML report content"""
        
        import datetime
        
        metrics = primary_data['metrics']
        warnings = primary_data['warnings']
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #212529;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: 600;
            color: #212529;
        }}
        .report-image {{
            text-align: center;
            margin: 40px 0;
        }}
        .report-image img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .warnings {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }}
        .warnings h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .warning-item {{
            margin: 10px 0;
            padding: 8px 12px;
            background: rgba(255, 193, 7, 0.1);
            border-left: 3px solid #ffc107;
        }}
        .quality-score {{
            text-align: center;
            margin: 30px 0;
        }}
        .quality-score .score {{
            font-size: 3em;
            font-weight: bold;
            color: #28a745;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .comparison-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Voice Analysis Report</h1>
            <p>Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        <div class="content">
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="label">Duration</div>
                    <div class="value">{metrics.get('duration_sec', 0):.2f} s</div>
                </div>
                <div class="summary-card">
                    <div class="label">F0 Median</div>
                    <div class="value">{self._format_numeric_html(metrics.get('f0_median_hz', 0), '.0f')} Hz</div>
                </div>
                <div class="summary-card">
                    <div class="label">Voiced</div>
                    <div class="value">{metrics.get('voiced_fraction', 0)*100:.0f}%</div>
                </div>
                <div class="summary-card">
                    <div class="label">RMS Level</div>
                    <div class="value">{metrics.get('rms_dbfs', 0):.1f} dBFS</div>
                </div>
                <div class="summary-card">
                    <div class="label">Peak Level</div>
                    <div class="value">{metrics.get('peak_dbfs', 0):.1f} dBFS</div>
                </div>
                <div class="summary-card">
                    <div class="label">Quality Score</div>
                    <div class="value">{metrics.get('data_quality_score', 0)}/100</div>
                </div>
            </div>
            
            <div class="report-image">
                <img src="report.png" alt="Voice Analysis Report">
            </div>
"""
        
        # Add warnings section if any
        if warnings:
            html += f"""
            <div class="warnings">
                <h3>Analysis Warnings ({len(warnings)})</h3>
"""
            for warning in warnings:
                html += f"""
                <div class="warning-item">
                    <strong>{warning.get('code', 'UNKNOWN')}:</strong> {warning.get('message', 'No message')}
                </div>
"""
            html += "</div>"
        
        # Add comparison table if in compare mode
        if comparison_data:
            comp_metrics = comparison_data['metrics']
            html += f"""
            <h2>Comparison Results</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>{file_names[0].upper()}</th>
                        <th>{file_names[1].upper()}</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>F0 Median (Hz)</td>
                        <td>{self._format_numeric_html(metrics.get('f0_median_hz', 0), '.1f')}</td>
                        <td>{self._format_numeric_html(comp_metrics.get('f0_median_hz', 0), '.1f')}</td>
                        <td>{self._format_numeric_diff_html(comp_metrics.get('f0_median_hz', 0), metrics.get('f0_median_hz', 0), '+.1f')}</td>
                    </tr>
                    <tr>
                        <td>RMS (dBFS)</td>
                        <td>{metrics.get('rms_dbfs', 0):.1f}</td>
                        <td>{comp_metrics.get('rms_dbfs', 0):.1f}</td>
                        <td>{comp_metrics.get('rms_dbfs', 0) - metrics.get('rms_dbfs', 0):+.1f}</td>
                    </tr>
                    <tr>
                        <td>Voiced Fraction</td>
                        <td>{metrics.get('voiced_fraction', 0)*100:.1f}%</td>
                        <td>{comp_metrics.get('voiced_fraction', 0)*100:.1f}%</td>
                        <td>{(comp_metrics.get('voiced_fraction', 0) - metrics.get('voiced_fraction', 0))*100:+.1f}%</td>
                    </tr>
                </tbody>
            </table>
"""
        
        html += """
        </div>
        
        <div class="footer">
            Generated by Apple-grade Voice Analysis Pipeline v1.0.0
        </div>
    </div>
</body>
</html>
"""
        
        return html
