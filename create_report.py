#!/usr/bin/env python3
"""
Apple-grade отчет для voice-analyze
Acceptance Checklist compliant: report.png/pdf/html, spectrogram.npz, manifest.json updates
"""

import sys
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

# Apple-style визуальный стиль
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

# Параметры (должны совпадать с bash скриптом)
SAMPLE_RATE = 48000
WINDOW_LENGTH = 0.025
HOP_LENGTH = 0.005
FFT_SIZE = 2048
DYNAMIC_RANGE = 50
WINDOW_TYPE = 'hann'
PITCH_FLOOR = 60
PITCH_CEILING = 400

def normalize_pitch(df):
    """Нормализует колонки pitch DataFrame (time_s, f0_hz, voiced)"""
    if df.empty:
        return pd.DataFrame({'time_s': [], 'f0_hz': [], 'voiced': []})
    
    cols = {c.lower().strip(): c for c in df.columns}
    
    # Ищем time колонку
    time_col = None
    for key in ['time_s', 'time', 'time (s)', 't']:
        if key in cols:
            time_col = cols[key]
            break
    
    # Ищем f0 колонку
    f0_col = None
    for key in ['f0_hz', 'f0', 'f0 (hz)', 'frequency', 'f0(Hz)']:
        if key in cols:
            f0_col = cols[key]
            break
    
    if time_col is None or f0_col is None:
        raise ValueError(f"Pitch columns not found. Available: {list(df.columns)}")
    
    out = df[[time_col, f0_col]].copy()
    out = out.rename(columns={time_col: 'time_s', f0_col: 'f0_hz'})
    out['f0_hz'] = pd.to_numeric(out['f0_hz'], errors='coerce').fillna(0.0)
    
    # Добавляем voiced если нет
    if 'voiced' not in df.columns:
        out['voiced'] = (out['f0_hz'] > 0).astype(int)
    else:
        out['voiced'] = df['voiced']
    
    return out[['time_s', 'f0_hz', 'voiced']]

def normalize_formants(df):
    """Нормализует колонки formants DataFrame (time_s, f1_hz, f2_hz, f3_hz, voiced)"""
    if df.empty:
        return pd.DataFrame({'time_s': [], 'f1_hz': [], 'f2_hz': [], 'f3_hz': [], 'voiced': []})
    
    cols = {c.lower().strip(): c for c in df.columns}
    
    time_col = None
    for key in ['time_s', 'time', 'time (s)', 't']:
        if key in cols:
            time_col = cols[key]
            break
    
    f1_col = None
    for key in ['f1_hz', 'f1', 'f1 (hz)', 'f1(Hz)']:
        if key in cols:
            f1_col = cols[key]
            break
    
    f2_col = None
    for key in ['f2_hz', 'f2', 'f2 (hz)', 'f2(Hz)']:
        if key in cols:
            f2_col = cols[key]
            break
    
    f3_col = None
    for key in ['f3_hz', 'f3', 'f3 (hz)', 'f3(Hz)']:
        if key in cols:
            f3_col = cols[key]
            break
    
    if not all([time_col, f1_col, f2_col, f3_col]):
        return pd.DataFrame({'time_s': [], 'f1_hz': [], 'f2_hz': [], 'f3_hz': [], 'voiced': []})
    
    out = df[[time_col, f1_col, f2_col, f3_col]].copy()
    out = out.rename(columns={time_col: 'time_s', f1_col: 'f1_hz', f2_col: 'f2_hz', f3_col: 'f3_hz'})
    out['f1_hz'] = pd.to_numeric(out['f1_hz'], errors='coerce').fillna(0.0)
    out['f2_hz'] = pd.to_numeric(out['f2_hz'], errors='coerce').fillna(0.0)
    out['f3_hz'] = pd.to_numeric(out['f3_hz'], errors='coerce').fillna(0.0)
    out['voiced'] = ((out['f1_hz'] > 0) | (out['f2_hz'] > 0) | (out['f3_hz'] > 0)).astype(int)
    
    return out[['time_s', 'f1_hz', 'f2_hz', 'f3_hz', 'voiced']]

def normalize_ltas(df):
    """Нормализует колонки LTAS DataFrame (freq_hz, db)"""
    if df.empty:
        return pd.DataFrame({'freq_hz': [], 'db': []})
    
    cols = {c.lower().strip(): c for c in df.columns}
    
    freq_col = None
    for key in ['freq_hz', 'frequency', 'freq', 'f']:
        if key in cols:
            freq_col = cols[key]
            break
    
    db_col = None
    for key in ['db', 'db (re 1 pa)', 'db(re 1 pa)', 'amplitude']:
        if key in cols:
            db_col = cols[key]
            break
    
    if not all([freq_col, db_col]):
        return pd.DataFrame({'freq_hz': [], 'db': []})
    
    out = df[[freq_col, db_col]].copy()
    out = out.rename(columns={freq_col: 'freq_hz', db_col: 'db'})
    out['freq_hz'] = pd.to_numeric(out['freq_hz'], errors='coerce')
    out['db'] = pd.to_numeric(out['db'], errors='coerce')
    
    return out[['freq_hz', 'db']].dropna()

def load_data(results_dir, file_name):
    """Загружает данные из data/: pitch.csv, formants.csv, ltas.csv, metrics.json"""
    data_dir = Path(results_dir) / 'data'
    
    # Pitch CSV
    pitch_csv = data_dir / 'pitch.csv'
    if pitch_csv.exists():
        try:
            pitch_df = pd.read_csv(pitch_csv)
            pitch_df = normalize_pitch(pitch_df)
        except Exception as e:
            print(f"Warning: Failed to load pitch: {e}", file=sys.stderr)
            pitch_df = pd.DataFrame({'time_s': [], 'f0_hz': [], 'voiced': []})
    else:
        pitch_df = pd.DataFrame({'time_s': [], 'f0_hz': [], 'voiced': []})
    
    # Formants CSV
    formants_csv = data_dir / 'formants.csv'
    if formants_csv.exists():
        try:
            formants_df = pd.read_csv(formants_csv)
            formants_df = normalize_formants(formants_df)
        except Exception as e:
            print(f"Warning: Failed to load formants: {e}", file=sys.stderr)
            formants_df = pd.DataFrame({'time_s': [], 'f1_hz': [], 'f2_hz': [], 'f3_hz': [], 'voiced': []})
    else:
        formants_df = pd.DataFrame({'time_s': [], 'f1_hz': [], 'f2_hz': [], 'f3_hz': [], 'voiced': []})
    
    # LTAS CSV
    ltas_csv = data_dir / 'ltas.csv'
    if ltas_csv.exists():
        try:
            ltas_df = pd.read_csv(ltas_csv)
            ltas_df = normalize_ltas(ltas_df)
        except Exception as e:
            print(f"Warning: Failed to load LTAS: {e}", file=sys.stderr)
            ltas_df = pd.DataFrame({'freq_hz': [], 'db': []})
    else:
        ltas_df = pd.DataFrame({'freq_hz': [], 'db': []})
    
    # Metrics JSON
    metrics_json = data_dir / 'metrics.json'
    if metrics_json.exists():
        try:
            with open(metrics_json, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metrics: {e}", file=sys.stderr)
            metrics = {}
    else:
        metrics = {}
    
    return pitch_df, formants_df, ltas_df, metrics

def find_loudest_voiced_segment(y, sr, pitch_df, duration):
    """Находит самый громкий voiced участок для micro-zoom"""
    window_samples = int(0.2 * sr)  # 200 мс
    max_energy = 0
    best_start = duration / 2
    
    for start in np.arange(0, duration - 0.2, 0.1):
        start_idx = int(start * sr)
        end_idx = start_idx + window_samples
        if end_idx > len(y):
            break
        
        segment = y[start_idx:end_idx]
        energy = np.sum(segment ** 2)
        
        # Проверяем что это voiced участок
        segment_mid = start + 0.1
        if 'time_s' in pitch_df.columns:
            pitch_at_mid = pitch_df[pitch_df['time_s'].between(segment_mid - 0.05, segment_mid + 0.05)]
            f0_col = 'f0_hz' if 'f0_hz' in pitch_df.columns else 'f0'
            if len(pitch_at_mid) > 0 and pitch_at_mid[f0_col].max() > 0:
                if energy > max_energy:
                    max_energy = energy
                    best_start = start
    
    return best_start

def create_report(audio_files, file_names, results_dir, mode='single', baseline_file=None):
    """Создает отчет: report.png/pdf/html + spectrogram.npz"""
    
    results_dir = Path(results_dir)
    warnings = []
    
    # Режим compare: загружаем оба файла
    if mode == 'compare' and len(audio_files) >= 2:
        y1, sr1 = sf.read(audio_files[0])
        y2, sr2 = sf.read(audio_files[1])
        if len(y1.shape) > 1:
            y1 = np.mean(y1, axis=1)
        if len(y2.shape) > 1:
            y2 = np.mean(y2, axis=1)
        
        # Загружаем данные для обоих
        pitch1, formants1, ltas1, metrics1 = load_data(results_dir, file_names[0])
        pitch2, formants2, ltas2, metrics2 = load_data(results_dir, file_names[1])
        
        # Используем первый файл как основной для визуализации
        y, sr = y1, sr1
        pitch_df, formants_df, ltas_df, metrics = pitch1, formants1, ltas1, metrics1
    else:
        # Single mode
        y, sr = sf.read(audio_files[0])
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        pitch_df, formants_df, ltas_df, metrics = load_data(results_dir, file_names[0])
    
    # Источник истины: WAV файл
    duration = len(y) / sr
    time = np.arange(len(y)) / sr
    
    # Обновляем metrics из WAV (источник истины)
    metrics.setdefault('duration', duration)
    metrics.setdefault('sample_rate', sr)
    
    # Sanity checks (Acceptance Checklist compliant) - с градациями через elif
    if duration <= 0:
        warnings.append("Duration <= 0")
    if sr != SAMPLE_RATE:
        warnings.append(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")
    if metrics.get('peak_dbfs', -100) > 0:
        warnings.append(f"Peak > 0 dBFS (clipping): {metrics.get('peak_dbfs', 0):.1f} dBFS")
    if metrics.get('clipping_percent', 0) > 1:
        warnings.append(f"Clipping detected: {metrics.get('clipping_percent', 0):.1f}%")
    
    voiced_fraction = metrics.get('voiced_fraction', 0)
    if voiced_fraction < 0.1:
        warnings.append(f"Very low voiced fraction: {voiced_fraction*100:.1f}%")
    elif voiced_fraction < 0.3:
        warnings.append(f"Low voiced fraction: {voiced_fraction*100:.1f}%")
    
    f0_median = metrics.get('f0_median', 0)
    if f0_median == 0:
        warnings.append("Pitch median = 0 (possible silence or tracking failure)")
    elif f0_median < PITCH_FLOOR or f0_median > PITCH_CEILING:
        warnings.append(f"F0 median out of range: {f0_median:.1f} Hz")
    
    # Проверка NaN/undefined
    if 'f0_hz' in pitch_df.columns and len(pitch_df) > 0:
        nan_count = pitch_df['f0_hz'].isna().sum()
        if nan_count > len(pitch_df) * 0.5:
            warnings.append(f"High NaN ratio in pitch: {nan_count}/{len(pitch_df)}")
    
    # STFT
    f, t_stft, D = signal.stft(y, fs=sr, nperseg=int(WINDOW_LENGTH * sr),
                              noverlap=int((WINDOW_LENGTH - HOP_LENGTH) * sr),
                              nfft=FFT_SIZE, window=WINDOW_TYPE, boundary=None, padded=False)
    
    magnitude = np.abs(D)
    
    # Reference dB (честное сравнение)
    if mode == 'compare' and len(audio_files) >= 2:
        # Compare mode: общий reference от обоих файлов
        y1, sr1 = sf.read(audio_files[0])
        y2, sr2 = sf.read(audio_files[1])
        if len(y1.shape) > 1:
            y1 = np.mean(y1, axis=1)
        if len(y2.shape) > 1:
            y2 = np.mean(y2, axis=1)
        
        f1, t1, D1 = signal.stft(y1, fs=sr1, nperseg=int(WINDOW_LENGTH * sr1),
                                 noverlap=int((WINDOW_LENGTH - HOP_LENGTH) * sr1),
                                 nfft=FFT_SIZE, window=WINDOW_TYPE, boundary=None, padded=False)
        f2, t2, D2 = signal.stft(y2, fs=sr2, nperseg=int(WINDOW_LENGTH * sr2),
                                 noverlap=int((WINDOW_LENGTH - HOP_LENGTH) * sr2),
                                 nfft=FFT_SIZE, window=WINDOW_TYPE, boundary=None, padded=False)
        
        magnitude1 = np.abs(D1)
        magnitude2 = np.abs(D2)
        global_max = max(magnitude1.max(), magnitude2.max())
        reference_db = 20 * np.log10(global_max + 1e-12)
        
        # Пересчитываем magnitude_db для текущего файла с общим reference
        magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
    elif mode == 'compare' and baseline_file:
        # Baseline mode: reference от baseline файла
        y_baseline, sr_baseline = sf.read(baseline_file)
        if len(y_baseline.shape) > 1:
            y_baseline = np.mean(y_baseline, axis=1)
        f_b, t_b, D_b = signal.stft(y_baseline, fs=sr_baseline, 
                                    nperseg=int(WINDOW_LENGTH * sr_baseline),
                                    noverlap=int((WINDOW_LENGTH - HOP_LENGTH) * sr_baseline),
                                    nfft=FFT_SIZE, window=WINDOW_TYPE, boundary=None, padded=False)
        magnitude_b = np.abs(D_b)
        global_max = max(magnitude.max(), magnitude_b.max())
        reference_db = 20 * np.log10(global_max + 1e-12)
        magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
    else:
        # Single mode: ref = max этого файла
        reference_db = 20 * np.log10(magnitude.max() + 1e-12)
        magnitude_db = 20 * np.log10(magnitude + 1e-12) - reference_db
    
    vmin = -DYNAMIC_RANGE
    vmax = 0
    
    # Сохраняем spectrogram.npz для верификации
    spectrogram_npz = results_dir / 'data' / 'spectrogram.npz'
    np.savez(spectrogram_npz,
             magnitude_db=magnitude_db,
             time=t_stft,
             freq=f,
             reference_db=reference_db,
             dynamic_range=DYNAMIC_RANGE)
    
    # Создаем фигуру
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[0.3, 0.5, 2, 2, 1])
    
    try:
        magma = plt.colormaps['magma']
    except (AttributeError, KeyError):
        magma = plt.cm.get_cmap('magma')
    
    # ========================================================================
    # SUMMARY BAR (верх) - 5-7 карточек
    # ========================================================================
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')
    
    # LTAS bands для summary
    ltas_body = metrics.get('ltas_body_db', 0)
    ltas_presence = metrics.get('ltas_presence_db', 0)
    ltas_sibilance = metrics.get('ltas_sibilance_db', 0)
    
    # Используем duration/sr из WAV (источник истины)
    duration_display = duration
    sr_display = sr
    f0_median_display = metrics.get('f0_median', 0)
    f0_q10_display = metrics.get('f0_q10', 0)
    f0_q90_display = metrics.get('f0_q90', 0)
    voiced_fraction_display = metrics.get('voiced_fraction', 0)
    rms_dbfs_display = metrics.get('rms_dbfs', 0)
    peak_dbfs_display = metrics.get('peak_dbfs', 0)
    
    summary_data = [
        ['Duration', f"{duration_display:.2f} s"],
        ['Sample Rate', f"{int(sr_display)} Hz"],
        ['F0', f"{f0_median_display:.0f} Hz [{f0_q10_display:.0f}-{f0_q90_display:.0f}]"],
        ['Voiced', f"{voiced_fraction_display*100:.0f}%"],
        ['RMS', f"{rms_dbfs_display:.1f} dBFS"],
        ['Peak', f"{peak_dbfs_display:.1f} dBFS"],
        ['Body', f"{ltas_body:.1f} dB"],
        ['Presence', f"{ltas_presence:.1f} dB"],
        ['Sibilance', f"{ltas_sibilance:.1f} dB" if ltas_sibilance != 0 else "N/A"],
        ['Warnings', f"{len(warnings)}"]
    ]
    
    # Создаем карточки
    n_cards = len(summary_data)
    card_width = 1.0 / n_cards
    for i, (label, value) in enumerate(summary_data):
        x_pos = i * card_width
        ax_summary.text(x_pos + card_width/2, 0.7, label, ha='center', va='bottom',
                       fontsize=9, color='#666', weight='normal')
        ax_summary.text(x_pos + card_width/2, 0.3, value, ha='center', va='top',
                       fontsize=11, color='#000', weight='bold')
        if i < n_cards - 1:
            ax_summary.axvline(x=(i+1)*card_width, color='#ddd', linewidth=0.5)
    
    # ========================================================================
    # WAVEFORM (полный + micro-zoom)
    # ========================================================================
    ax_wf = fig.add_subplot(gs[1, :])
    ax_wf.plot(time, y, 'k-', linewidth=0.3)
    ax_wf.set_xlim(0, duration)
    ax_wf.set_ylabel('Amplitude', fontsize=9)
    ax_wf.set_title('Waveform', fontsize=10, fontweight='bold')
    ax_wf.tick_params(labelsize=8)
    ax_wf.grid(True, alpha=0.2)
    
    # Micro-zoom (200 ms на самом громком voiced участке)
    zoom_start = find_loudest_voiced_segment(y, sr, pitch_df, duration)
    zoom_end = zoom_start + 0.2
    zoom_mask = (time >= zoom_start) & (time <= zoom_end)
    if np.any(zoom_mask):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_zoom = inset_axes(ax_wf, width="30%", height="40%", loc='upper right')
        ax_zoom.plot(time[zoom_mask], y[zoom_mask], 'k-', linewidth=1)
        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_title('Detail: 200 ms', fontsize=7)
        ax_zoom.tick_params(labelsize=6)
        ax_zoom.grid(True, alpha=0.3)
    
    # ========================================================================
    # СПЕКТРОГРАММА 0-5k (форманты) + F0/F1/F2/F3 overlay
    # ========================================================================
    ax_main = fig.add_subplot(gs[2, 0])
    
    # Лог-шкала с vectorized interpolation (0-5k для форматов)
    freq_mask = f <= 5000
    freqs = f[freq_mask]
    mag_db = magnitude_db[freq_mask, :]
    
    freq_log = np.logspace(np.log10(50), np.log10(5000), 300)
    interp_func = interp1d(freqs, mag_db, kind='linear', axis=0,
                          bounds_error=False, fill_value='extrapolate')
    magnitude_log = interp_func(freq_log)
    
    F, T = np.meshgrid(freq_log, t_stft, indexing='ij')
    im = ax_main.pcolormesh(T, F, magnitude_log, cmap=magma, vmin=vmin, vmax=vmax, shading='gouraud')
    
    # F0 overlay
    pitch_valid = None
    if 'f0_hz' in pitch_df.columns and len(pitch_df) > 0:
        pitch_valid = pitch_df[pitch_df['f0_hz'] > 0]
        if len(pitch_valid) > 0:
            f0_vals = pitch_valid['f0_hz'].values
            if len(f0_vals) > 1:
                diff = np.abs(np.diff(f0_vals))
                valid_mask = np.concatenate([[True], diff < 50])
                pitch_filtered = pitch_valid[valid_mask]
                if len(pitch_filtered) > 5:
                    pitch_smooth = signal.medfilt(pitch_filtered['f0_hz'].values, kernel_size=5)
                    ax_main.plot(pitch_filtered['time_s'], pitch_smooth, 'w-', linewidth=2, alpha=0.9, label='F0')
    
    # Formants overlay (F1, F2, F3)
    if 'f1_hz' in formants_df.columns and len(formants_df) > 0:
        for formant_num, color in [(1, 'cyan'), (2, 'yellow'), (3, 'magenta')]:
            formant_col = f'f{formant_num}_hz'
            if formant_col in formants_df.columns:
                formant_valid = formants_df[formants_df[formant_col] > 0]
                if len(formant_valid) > 0:
                    ax_main.plot(formant_valid['time_s'], formant_valid[formant_col],
                               color=color, linewidth=1.5, alpha=0.7, linestyle='--', label=f'F{formant_num}')
    
    ax_main.set_yscale('log')
    ax_main.set_ylim(50, 5000)
    ax_main.set_xlabel('Time (s)', fontsize=10)
    ax_main.set_ylabel('Frequency (Hz)', fontsize=10)
    ax_main.set_title('Spectrogram 0-5 kHz (Formants) + F0/F1/F2/F3', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_main, label='dB', format='%d')
    if (pitch_valid is not None and len(pitch_valid) > 0) or ('f1_hz' in formants_df.columns and len(formants_df) > 0):
        ax_main.legend(loc='upper right', fontsize=7)
    
    # ========================================================================
    # СПЕКТРОГРАММА 0-10k (сибилянты)
    # ========================================================================
    ax_sibil = fig.add_subplot(gs[3, 0])
    
    freq_mask = f <= 10000
    freqs = f[freq_mask]
    mag_db = magnitude_db[freq_mask, :]
    
    freq_log = np.logspace(np.log10(50), np.log10(10000), 300)
    interp_func = interp1d(freqs, mag_db, kind='linear', axis=0,
                          bounds_error=False, fill_value='extrapolate')
    magnitude_log = interp_func(freq_log)
    
    F, T = np.meshgrid(freq_log, t_stft, indexing='ij')
    im2 = ax_sibil.pcolormesh(T, F, magnitude_log, cmap=magma, vmin=vmin, vmax=vmax, shading='gouraud')
    
    ax_sibil.set_yscale('log')
    ax_sibil.set_ylim(50, 10000)
    ax_sibil.set_xlabel('Time (s)', fontsize=10)
    ax_sibil.set_ylabel('Frequency (Hz)', fontsize=10)
    ax_sibil.set_title('Spectrogram 0-10 kHz (Sibilance)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax_sibil, label='dB', format='%d')
    
    if len(warnings) > 0:
        ax_main.text(0.02, 0.98, f'Warnings: {len(warnings)}', transform=ax_main.transAxes,
                    fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # ========================================================================
    # LTAS + METRICS TABLE
    # ========================================================================
    ax_ltas = fig.add_subplot(gs[4, :])
    if len(ltas_df) > 0:
        freq_col = 'freq_hz' if 'freq_hz' in ltas_df.columns else 'frequency'
        db_col = 'db' if 'db' in ltas_df.columns else 'dB'
        ax_ltas.semilogx(ltas_df[freq_col], ltas_df[db_col], 'k-', linewidth=1.5)
        ax_ltas.axvline(200, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_ltas.axvline(4000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_ltas.text(200, ax_ltas.get_ylim()[1] * 0.9, 'Body', fontsize=7, ha='center')
        ax_ltas.text(4000, ax_ltas.get_ylim()[1] * 0.9, 'Presence', fontsize=7, ha='center')
    ax_ltas.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_ltas.set_ylabel('dB', fontsize=9)
    ax_ltas.set_title('LTAS', fontsize=10)
    ax_ltas.set_xlim(50, 10000)
    ax_ltas.grid(True, alpha=0.2)
    
    # Metrics table
    ax_metrics = fig.add_subplot(gs[4, 1])
    ax_metrics.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Duration', f"{metrics.get('duration', 0):.2f} s"],
        ['Sample rate', f"{int(metrics.get('sample_rate', 0))} Hz"],
        ['F0 median', f"{metrics.get('f0_median', 0):.1f} Hz"],
        ['F0 range', f"{metrics.get('f0_q10', 0):.0f}-{metrics.get('f0_q90', 0):.0f} Hz"],
        ['Voiced fraction', f"{metrics.get('voiced_fraction', 0)*100:.1f}%"],
        ['RMS (dBFS)', f"{metrics.get('rms_dbfs', 0):.1f} dB"],
        ['Peak (dBFS)', f"{metrics.get('peak_dbfs', 0):.1f} dB"],
        ['Crest Factor', f"{metrics.get('crest_factor', 0):.2f}"],
        ['LTAS Body', f"{metrics.get('ltas_body_db', 0):.1f} dB"],
        ['LTAS Presence', f"{metrics.get('ltas_presence_db', 0):.1f} dB"],
    ]
    
    table = ax_metrics.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#F5F5F5')
        table[(0, i)].set_text_props(weight='bold')
    
    ax_metrics.set_title('Metrics', fontsize=10, fontweight='bold', pad=10)
    
    # Общий заголовок
    fig.suptitle('Voice Analysis Report', fontsize=14, fontweight='bold', y=0.995)
    
    # Сохраняем
    output_png = results_dir / 'report.png'
    output_pdf = results_dir / 'report.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✅ Report: {output_png}")
    plt.close()
    
    # HTML отчет
    create_html_report(results_dir, file_names[0], metrics, warnings, len(warnings))
    
    # Обновляем manifest.json с warnings
    update_manifest_warnings(results_dir, warnings)
    
    return warnings

def create_html_report(results_dir, file_name, metrics, warnings, warnings_count):
    """Создает интерактивный HTML отчет"""
    html_file = results_dir / 'report.html'
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Voice Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif; 
               margin: 40px; background: #fafafa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }}
        h1 {{ color: #1d1d1f; font-weight: 600; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 30px 0; }}
        .card {{ background: #f5f5f7; padding: 15px; border-radius: 6px; }}
        .card-label {{ font-size: 12px; color: #86868b; text-transform: uppercase; }}
        .card-value {{ font-size: 24px; font-weight: 600; color: #1d1d1f; margin-top: 5px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 4px; }}
        .warnings {{ background: #fff3cd; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        .warnings ul {{ margin: 10px 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Analysis Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="card">
                <div class="card-label">Duration</div>
                <div class="card-value">{metrics.get('duration', 0):.2f} s</div>
            </div>
            <div class="card">
                <div class="card-label">F0 Median</div>
                <div class="card-value">{metrics.get('f0_median', 0):.0f} Hz</div>
            </div>
            <div class="card">
                <div class="card-label">Voiced</div>
                <div class="card-value">{metrics.get('voiced_fraction', 0)*100:.0f}%</div>
            </div>
            <div class="card">
                <div class="card-label">RMS</div>
                <div class="card-value">{metrics.get('rms_dbfs', 0):.1f} dBFS</div>
            </div>
            <div class="card">
                <div class="card-label">Peak</div>
                <div class="card-value">{metrics.get('peak_dbfs', 0):.1f} dBFS</div>
            </div>
            <div class="card">
                <div class="card-label">Warnings</div>
                <div class="card-value">{warnings_count}</div>
            </div>
        </div>
        
        <h2>Dashboard</h2>
        <img src="report.png" alt="Analysis Dashboard">
        
        {f'<div class="warnings"><h3>Warnings</h3><ul>' + ''.join(f'<li>{w}</li>' for w in warnings) + '</ul></div>' if warnings else ''}
    </div>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML: {html_file}")

def update_manifest_warnings(results_dir, warnings):
    """Обновляет manifest.json с warnings"""
    manifest_file = results_dir / 'manifest.json'
    if not manifest_file.exists():
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    manifest['warnings'] = warnings
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 create_report.py <audio_file> <file_name> <results_dir> [mode] [baseline_file]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    file_name = sys.argv[2]
    results_dir = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else 'single'
    baseline_file = sys.argv[5] if len(sys.argv) > 5 else None
    
    create_report([audio_file], [file_name], results_dir, mode, baseline_file)

