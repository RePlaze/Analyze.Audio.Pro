#!/usr/bin/env python3
"""
Верификация pitch вторым методом (pyin) для кросс-проверки результатов Praat
"""

import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

def verify_pitch_pyin(praat_csv, audio_file, output_csv):
    """Проверяет pitch через pyin (если доступен)"""
    try:
        import librosa
    except ImportError:
        print("Warning: librosa not available, skipping pyin verification")
        return False
    
    # Загружаем Praat pitch
    praat_df = pd.read_csv(praat_csv)
    if 'time_s' not in praat_df.columns:
        if 'Time (s)' in praat_df.columns:
            praat_df = praat_df.rename(columns={'Time (s)': 'time_s'})
        elif 'time' in praat_df.columns:
            praat_df = praat_df.rename(columns={'time': 'time_s'})
        else:
            print("Error: Cannot find time column in Praat CSV")
            return False
    
    f0_col = 'f0_hz' if 'f0_hz' in praat_df.columns else 'f0'
    if f0_col not in praat_df.columns and 'F0 (Hz)' in praat_df.columns:
        praat_df = praat_df.rename(columns={'F0 (Hz)': f0_col})
    
    # Загружаем аудио
    y, sr = sf.read(audio_file)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    
    # Pyin pitch tracking
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    
    # Создаем временную сетку для pyin
    hop_length = 512
    frame_time = librosa.frames_to_time(np.arange(len(f0_pyin)), sr=sr, hop_length=hop_length)
    
    # Сравниваем на voiced участках
    praat_voiced = praat_df[praat_df[f0_col] > 0]
    pyin_voiced = f0_pyin[~np.isnan(f0_pyin)]
    
    if len(praat_voiced) == 0 or len(pyin_voiced) == 0:
        print("Warning: No voiced frames for comparison")
        return False
    
    # Корреляция (на пересекающихся временных точках)
    correlation = np.corrcoef(
        praat_voiced[f0_col].values[:min(len(praat_voiced), len(pyin_voiced))],
        pyin_voiced[:min(len(praat_voiced), len(pyin_voiced))]
    )[0, 1]
    
    # MAE на voiced участках
    min_len = min(len(praat_voiced), len(pyin_voiced))
    mae = np.mean(np.abs(
        praat_voiced[f0_col].values[:min_len] - pyin_voiced[:min_len]
    ))
    
    print(f"Pitch verification (Praat vs pyin):")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  MAE: {mae:.1f} Hz")
    
    if correlation < 0.7:
        print("⚠️  Warning: Low correlation (< 0.7) - pitch disagreement")
    if mae > 20:
        print("⚠️  Warning: High MAE (> 20 Hz) - pitch disagreement")
    
    # Сохраняем результаты
    verify_df = pd.DataFrame({
        'time_s': frame_time,
        'f0_pyin_hz': f0_pyin,
        'voiced': voiced_flag.astype(int)
    })
    verify_df.to_csv(output_csv, index=False)
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 verify_pitch.py <praat_csv> <audio_file> <output_csv>")
        sys.exit(1)
    
    praat_csv = sys.argv[1]
    audio_file = sys.argv[2]
    output_csv = sys.argv[3]
    
    if verify_pitch_pyin(praat_csv, audio_file, output_csv):
        print(f"✅ Verification saved: {output_csv}")
    else:
        print("❌ Verification failed")
        sys.exit(1)

