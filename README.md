# analyze-audio

**Professional voice analysis. One command. Complete report.**
![Screenshot 2025-12-27 at 16.53.36.png](Screenshot%202025-12-27%20at%2016.53.36.png)
![Screenshot 2025-12-27 at 16.53.56.png](Screenshot%202025-12-27%20at%2016.53.56.png)
![Screenshot 2025-12-27 at 16.54.15.png](Screenshot%202025-12-27%20at%2016.54.15.png)
---

## 🚀 Quick Start

```bash
/analyze-audio <path_to_audio_or_video>
```

**Examples:**

```bash
/analyze-audio voice.mp3
/analyze-audio take.wav
/analyze-audio interview.m4a
/analyze-audio clip.mp4
```
**Output:**

```
Done
Report: /absolute/path/analysis_YYYYMMDD_HHMMSS/report.html
Open: open "/absolute/path/analysis_YYYYMMDD_HHMMSS"
```

---

## 📊 What It Does

Analyzes audio or video files and generates a comprehensive voice analysis report:

- **Pitch tracking** (F0) with dual-method verification
- **Formant analysis** (F1, F2, F3)
- **Spectrograms** (0–5 kHz for formants, 0–10 kHz for sibilance)
- **Long-Term Average Spectrum** (LTAS)
- **Loudness metrics** (RMS, Peak, True Peak in dBFS)
- **Quality checks** (clipping, voiced fraction, pitch confidence)
- **RUS/ENG support & Dark theme available**

All results are saved in a structured format with full reproducibility.

---

## 📋 Requirements

**macOS** (tested on macOS 14+)

**Praat** — installed at `/Applications/Praat.app`

**Python 3.8+** with packages:
```bash
pip3 install numpy scipy matplotlib pandas soundfile pysptk
```

**FFmpeg** (optional, required only for video input)
```bash
brew install ffmpeg
```

---

## 📁 Output Structure

Each analysis creates a timestamped folder:

```
analysis_YYYYMMDD_HHMMSS/
├── report.html          # Interactive report
├── report.png           # Preview image
├── report.pdf           # Print-ready PDF
├── manifest.json        # Metadata and settings
├── data/
│   ├── pitch.csv        # Pitch track (time, f0, voiced)
│   ├── formants.csv     # Formants (time, f1, f2, f3)
│   ├── ltas.csv         # Long-term spectrum
│   ├── metrics.json     # All metrics
│   └── spectrogram.npz  # Raw spectrogram data
└── logs/
    ├── ffmpeg.log       # Audio extraction log
    ├── praat.log        # Praat analysis log
    └── analyze.log      # Main pipeline log
```

---

## 📈 Report Contents

### Summary Cards

Key metrics at a glance:
- Duration, Sample Rate
- F0 (median, range)
- Voiced fraction
- RMS/Peak (dBFS)
- LTAS bands (Body, Presence, Sibilance)
- Warnings count

### Main Section

1. **Waveform** (full + 200ms micro-zoom)
2. **Spectrogram 0–5 kHz** (formants) with F0/F1/F2/F3 overlay
3. **Spectrogram 0–10 kHz** (sibilance)

### Bottom Section

- **LTAS** (log-frequency spectrum)
- **Analysis parameters** table
- **Warnings** (if any)
- **Raw data** links

---

## ⚙️ Technical Details

### Audio Processing

- **Input**: Any audio/video format supported by FFmpeg
- **Output**: WAV PCM, mono, 48 kHz
- **SHA256**: Calculated for input file integrity

<details>
<summary><strong>Analysis Parameters</strong></summary>

**STFT:**
- Window: 25 ms (Hann)
- Hop: 5 ms
- FFT: 2048
- Dynamic range: 50 dB (fixed)

**Pitch:**
- Method: Praat (primary) + pyin (verification)
- Range: 60–400 Hz
- Grid: 10 ms uniform

**Formants:**
- Method: Burg
- Number: 5
- Max frequency: 5000 Hz
- Grid: 20 ms uniform

**LTAS:**
- Bandwidth: 50 Hz

</details>

### Quality Checks

Automatic sanity checks for:
- Duration validity
- Sample rate consistency
- Clipping detection
- Voiced fraction thresholds
- Pitch tracking confidence
- NaN/undefined values

---

## 🎯 Philosophy

**One command. No flags. No modes.**

Everything is automatic:
- Audio/video detection
- Format conversion
- Full analysis pipeline
- Quality verification
- Report generation

**Honest comparison.**

- Fixed dynamic range (50 dB)
- Consistent reference dB
- No per-frame normalization
- Reproducible parameters

**Apple-grade quality.**

- Clean, minimal output
- Structured data
- Full traceability (manifest.json)
- Professional visualizations

---

## 🔧 Troubleshooting

### "Praat not found"

Ensure Praat is installed at `/Applications/Praat.app`. The script will check and provide instructions if missing.

### "Python module not found"

Install missing packages:
```bash
pip3 install <package_name>
```

### "FFmpeg not found" (video input)

Install FFmpeg:
```bash
brew install ffmpeg
```

Or use audio-only input (WAV, MP3, etc.).

### Empty metrics in report

Check `logs/praat.log` for Praat errors. Common issues:
- Corrupted audio file
- Too short duration (< 0.1s)
- Silence-only file

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

Built with:
- **Praat** (phonetic analysis)
- **FFmpeg** (audio processing)
- **Python** (visualization and automation)
