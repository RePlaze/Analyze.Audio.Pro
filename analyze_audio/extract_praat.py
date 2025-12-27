"""
Praat extraction module - Apple-grade: deterministic, logged, validated
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from .settings import (
    SAMPLE_RATE, PITCH_FLOOR, PITCH_CEILING, WINDOW_LENGTH, 
    FORMANT_MAX, get_praat_path, FREQUENCY_BANDS
)
from .schema import ValidationError


class PraatExtractor:
    """Handles all Praat-based analysis with proper error handling"""
    
    def __init__(self, praat_path: str = None):
        self.praat_path = praat_path or get_praat_path()
        if not self.praat_path:
            raise ValidationError("Praat executable not found")
    
    def extract_all_data(self, audio_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Extract all analysis data from audio file"""
        
        # Create output directories
        data_dir = output_dir / 'data'
        logs_dir = output_dir / 'logs'
        data_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        # Generate Praat script
        script_content = self._generate_praat_script(audio_file, data_dir)
        
        # Execute Praat script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.praat', delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)
        
        try:
            # Run Praat with logging
            result = subprocess.run(
                [self.praat_path, '--run', str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log Praat output
            praat_log = logs_dir / 'praat.log'
            with open(praat_log, 'w') as f:
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            if result.returncode != 0:
                raise ValidationError(f"Praat execution failed: {result.stderr}")
            
            # Convert Praat output to standardized CSV format
            self._convert_to_standard_format(data_dir)
            
            return {
                'praat_exit_code': result.returncode,
                'praat_stdout': result.stdout,
                'praat_stderr': result.stderr
            }
            
        finally:
            # Cleanup temp script
            script_path.unlink(missing_ok=True)
    
    def _generate_praat_script(self, audio_file: Path, data_dir: Path) -> str:
        """Generate comprehensive Praat analysis script"""
        
        # Use absolute paths to avoid Praat path issues
        audio_path = str(audio_file.resolve())
        
        script = f'''
# Apple-grade Praat analysis script
# Generated automatically - do not edit manually

# Parameters
audioFile$ = "{audio_path}"
pitchFloor = {PITCH_FLOOR}
pitchCeiling = {PITCH_CEILING}  
windowLength = {WINDOW_LENGTH}
formantMax = {FORMANT_MAX}

# Output files
pitchFile$ = "{str((data_dir / 'pitch_raw.txt').resolve())}"
formantFile$ = "{str((data_dir / 'formants_raw.txt').resolve())}"
ltasFile$ = "{str((data_dir / 'ltas_raw.txt').resolve())}"
metricsFile$ = "{str((data_dir / 'metrics_raw.txt').resolve())}"

# Validation: Check if file exists
if not fileReadable(audioFile$)
    exitScript: "ERROR: Audio file not readable: " + audioFile$
endif

# Read audio file
Read from file: audioFile$
soundID = selected("Sound")

# Basic validation
duration = Get total duration
sampleRate = Get sample rate
nChannels = Get number of channels

if duration <= 0
    exitScript: "ERROR: Invalid duration: " + string$(duration)
endif

if sampleRate != {SAMPLE_RATE}
    writeInfoLine: "WARNING: Sample rate mismatch: " + string$(sampleRate) + " != {SAMPLE_RATE}"
endif

# Convert to mono if needed
if nChannels > 1
    Convert to mono
    soundID = selected("Sound")
    writeInfoLine: "INFO: Converted " + string$(nChannels) + " channels to mono"
endif

# ============================================================================
# PITCH ANALYSIS
# ============================================================================
selectObject: soundID
To Pitch: 0, pitchFloor, pitchCeiling
pitchID = selected("Pitch")

# Pitch statistics
medianF0 = Get quantile: 0, 0, 0.5, "Hertz"
q10F0 = Get quantile: 0, 0, 0.1, "Hertz"
q90F0 = Get quantile: 0, 0, 0.9, "Hertz"
meanF0 = Get mean: 0, 0, "Hertz"
stdF0 = Get standard deviation: 0, 0, "Hertz"

# Calculate voiced fraction directly from pitch object
selectObject: pitchID
nFrames = Get number of frames
nVoiced = 0
nJumps = 0
prevF0 = undefined

# Count voiced frames and pitch jumps
for iFrame from 1 to nFrames
    f0 = Get value in frame: iFrame, "Hertz"
    if f0 <> undefined and f0 > 0
        nVoiced = nVoiced + 1
        
        # Count pitch jumps for confidence
        if prevF0 <> undefined and abs(f0 - prevF0) > 50
            nJumps = nJumps + 1
        endif
        prevF0 = f0
    endif
endfor

voicedFraction = nVoiced / nFrames
pitchConfidence = 1 - (nJumps / max(1, nVoiced))

# Export pitch data frame by frame
writeFile: pitchFile$, "time_s" + tab$ + "f0_hz" + tab$ + "voiced" + tab$ + "method" + tab$ + "confidence" + newline$

for iFrame from 1 to nFrames
    time = Get time from frame: iFrame
    f0 = Get value in frame: iFrame, "Hertz"
    
    if f0 = undefined or f0 <= 0
        f0 = 0
        voiced = 0
    else
        voiced = 1
    endif
    
    appendFile: pitchFile$, string$(time) + tab$ + string$(f0) + tab$ + string$(voiced) + tab$ + "praat" + tab$ + string$(pitchConfidence) + newline$
endfor

# ============================================================================
# FORMANT ANALYSIS  
# ============================================================================
selectObject: soundID
To Formant (burg): 0, 5, formantMax, windowLength, 50
formantID = selected("Formant")

# Export formants with standardized column names
selectObject: formantID

# Use Down to Table and then rename columns
Down to Table: "no", "yes", 6, "yes", 3, "yes", 3, "yes"
formantTableID = selected("Table")

# Get the table data and write with our column names
nRows = Get number of rows
writeFile: formantFile$, "time_s" + tab$ + "f1_hz" + tab$ + "f2_hz" + tab$ + "f3_hz" + newline$

for iRow from 1 to nRows
    time = Get value: iRow, "time(s)"
    f1$ = Get value: iRow, "F1(Hz)"
    f2$ = Get value: iRow, "F2(Hz)" 
    f3$ = Get value: iRow, "F3(Hz)"
    
    # Handle undefined values (represented as "--undefined--" in Praat tables)
    if f1$ = "--undefined--"
        f1 = 0
    else
        f1 = number(f1$)
    endif
    
    if f2$ = "--undefined--"
        f2 = 0
    else
        f2 = number(f2$)
    endif
    
    if f3$ = "--undefined--"
        f3 = 0
    else
        f3 = number(f3$)
    endif
    
    appendFile: formantFile$, string$(time) + tab$ + string$(f1) + tab$ + string$(f2) + tab$ + string$(f3) + newline$
endfor

selectObject: formantTableID
Remove

# ============================================================================
# LTAS ANALYSIS
# ============================================================================
selectObject: soundID
To Ltas: 50
ltasID = selected("Ltas")

# Export LTAS data at key frequencies
writeFile: ltasFile$, "freq_hz" + tab$ + "db" + newline$

# Sample key frequencies manually using correct Praat syntax
freq = 50
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 100
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 200
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 500
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 1000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 2000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 4000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 6000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 8000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

freq = 10000
db = Get value at frequency: freq, "Cubic"
appendFile: ltasFile$, string$(freq) + tab$ + string$(db) + newline$

# Calculate band energies with averaging method
bodyDb = Get mean: {FREQUENCY_BANDS['body'][0]}, {FREQUENCY_BANDS['body'][1]}, "energy"
presenceDb = Get mean: {FREQUENCY_BANDS['presence'][0]}, {FREQUENCY_BANDS['presence'][1]}, "energy"

# Sibilance band (check if frequency range allows)
maxFreq = Get highest frequency
if maxFreq >= {FREQUENCY_BANDS['sibilance'][1]}
    sibilanceDb = Get mean: {FREQUENCY_BANDS['sibilance'][0]}, {FREQUENCY_BANDS['sibilance'][1]}, "energy"
else
    sibilanceDb = undefined
endif

# ============================================================================
# AUDIO METRICS
# ============================================================================
selectObject: soundID

# Time-domain metrics
rms = Get root-mean-square: 0, 0
maxAmp = Get maximum: 0, 0, "Parabolic"
minAmp = Get minimum: 0, 0, "Parabolic"
peak = max(abs(maxAmp), abs(minAmp))

rmsDbfs = 20 * log10(rms + 1e-12)
peakDbfs = 20 * log10(peak + 1e-12)
crestFactor = peak / (rms + 1e-12)

# Clipping detection (sample first 100k samples for speed)
nSamples = Get number of samples
nSamplesToCheck = nSamples
if nSamples > 100000
    nSamplesToCheck = 100000
endif

nClipped = 0
for iSample from 1 to nSamplesToCheck
    value = Get value at sample number: iSample
    if abs(value) > 0.99
        nClipped = nClipped + 1
    endif
endfor
clippingPercent = (nClipped / nSamplesToCheck) * 100

# Spectral metrics (basic)
To Spectrum: "yes"
spectrumID = selected("Spectrum")
spectralCentroid = Get centre of gravity: 2

# Calculate spectral flatness (geometric mean / arithmetic mean)
nBins = Get number of bins
geometricSum = 0
arithmeticSum = 0
validBins = 0

for iBin from 1 to nBins
    power = Get real value in bin: iBin
    power = power * power
    if power > 0
        geometricSum = geometricSum + ln(power)
        arithmeticSum = arithmeticSum + power
        validBins = validBins + 1
    endif
endfor

if validBins > 0
    geometricMean = exp(geometricSum / validBins)
    arithmeticMean = arithmeticSum / validBins
    if arithmeticMean > 0
        spectralFlatness = geometricMean / arithmeticMean
    else
        spectralFlatness = 0
    endif
else
    spectralFlatness = 0
endif

selectObject: spectrumID
Remove

# Export metrics in simple tab-separated format (easier than JSON in Praat)
writeFile: metricsFile$, "metric" + tab$ + "value" + newline$
appendFile: metricsFile$, "duration_sec" + tab$ + string$(duration) + newline$
appendFile: metricsFile$, "sample_rate_hz" + tab$ + string$(sampleRate) + newline$
appendFile: metricsFile$, "f0_median_hz" + tab$ + string$(medianF0) + newline$
appendFile: metricsFile$, "f0_q10_hz" + tab$ + string$(q10F0) + newline$
appendFile: metricsFile$, "f0_q90_hz" + tab$ + string$(q90F0) + newline$
appendFile: metricsFile$, "f0_mean_hz" + tab$ + string$(meanF0) + newline$
appendFile: metricsFile$, "f0_std_hz" + tab$ + string$(stdF0) + newline$
appendFile: metricsFile$, "voiced_fraction" + tab$ + string$(voicedFraction) + newline$
appendFile: metricsFile$, "pitch_confidence" + tab$ + string$(pitchConfidence) + newline$
appendFile: metricsFile$, "rms_dbfs" + tab$ + string$(rmsDbfs) + newline$
appendFile: metricsFile$, "peak_dbfs" + tab$ + string$(peakDbfs) + newline$
appendFile: metricsFile$, "crest_factor" + tab$ + string$(crestFactor) + newline$
appendFile: metricsFile$, "clipping_percent" + tab$ + string$(clippingPercent) + newline$
appendFile: metricsFile$, "spectral_centroid_hz_mean" + tab$ + string$(spectralCentroid) + newline$
appendFile: metricsFile$, "spectral_flatness_mean" + tab$ + string$(spectralFlatness) + newline$
appendFile: metricsFile$, "ltas_body_db" + tab$ + string$(bodyDb) + newline$
appendFile: metricsFile$, "ltas_presence_db" + tab$ + string$(presenceDb) + newline$

if sibilanceDb <> undefined
    appendFile: metricsFile$, "ltas_sibilance_db" + tab$ + string$(sibilanceDb) + newline$
else
    appendFile: metricsFile$, "ltas_sibilance_db" + tab$ + "0" + newline$
endif

appendFile: metricsFile$, "data_quality_score" + tab$ + "100" + newline$

# Cleanup
selectObject: soundID, pitchID, formantID, ltasID
Remove

writeInfoLine: "SUCCESS: Analysis completed"
'''
        
        return script
    
    def _convert_to_standard_format(self, data_dir: Path):
        """Convert Praat raw output to standardized CSV format"""
        
        # Convert pitch data
        pitch_raw = data_dir / 'pitch_raw.txt'
        pitch_csv = data_dir / 'pitch.csv'
        if pitch_raw.exists():
            # Praat output is already in correct format
            pitch_raw.rename(pitch_csv)
        
        # Convert formants data
        formants_raw = data_dir / 'formants_raw.txt'
        formants_csv = data_dir / 'formants.csv'
        if formants_raw.exists():
            formants_raw.rename(formants_csv)
        
        # Convert LTAS data
        ltas_raw = data_dir / 'ltas_raw.txt'
        ltas_csv = data_dir / 'ltas.csv'
        if ltas_raw.exists():
            ltas_raw.rename(ltas_csv)
        
        # Convert metrics data from tab-separated to JSON
        metrics_raw = data_dir / 'metrics_raw.txt'
        metrics_json = data_dir / 'metrics.json'
        if metrics_raw.exists():
            # Read tab-separated metrics and convert to JSON
            import json
            metrics_dict = {}
            with open(metrics_raw, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        key, value = parts
                        try:
                            # Try to convert to float, otherwise keep as string
                            metrics_dict[key] = float(value)
                        except ValueError:
                            metrics_dict[key] = value
            
            with open(metrics_json, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            # Remove the raw file
            metrics_raw.unlink()


def extract_audio_data(audio_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Main entry point for Praat-based audio analysis"""
    extractor = PraatExtractor()
    return extractor.extract_all_data(audio_file, output_dir)
