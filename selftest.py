#!/usr/bin/env python3
"""
Автотесты: синус, гармоники, тишина, шум, сравнение
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import subprocess
import os

SAMPLE_RATE = 48000
DURATION = 1.0  # 1 секунда для тестов

def create_sine(freq, duration, sr):
    """Создает синусоиду"""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t)

def create_harmonics(fundamental, n_harmonics, duration, sr):
    """Создает гармоники"""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    for i in range(1, n_harmonics + 1):
        signal += np.sin(2 * np.pi * fundamental * i * t) / i
    return signal / np.max(np.abs(signal))

def create_silence(duration, sr):
    """Создает тишину"""
    return np.zeros(int(sr * duration))

def create_noise(duration, sr):
    """Создает белый шум"""
    return np.random.randn(int(sr * duration)) * 0.1

def run_test(test_name, audio_data, expected_checks, results_dir):
    """Запускает тест через voice-analyze"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio_data, SAMPLE_RATE)
        temp_file = f.name
    
    try:
        # Запускаем voice-analyze
        test_dir = Path(results_dir) / f"test_{test_name}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        script_dir = Path(__file__).parent
        voice_analyze = script_dir / "voice-analyze"
        
        result = subprocess.run(
            [str(voice_analyze), temp_file, "--out", str(test_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ {test_name}: voice-analyze failed")
            print(result.stderr)
            return False
        
        # Проверяем результаты
        metrics_file = test_dir / "data" / "metrics_audio.json"
        if not metrics_file.exists():
            print(f"❌ {test_name}: metrics.json not found")
            return False
        
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Выполняем проверки
        all_passed = True
        for check_name, check_func in expected_checks.items():
            if not check_func(metrics):
                print(f"❌ {test_name}: {check_name} failed")
                all_passed = False
            else:
                print(f"✅ {test_name}: {check_name} passed")
        
        return all_passed
        
    finally:
        os.unlink(temp_file)

def run_selftest(results_dir):
    """Запускает все автотесты"""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running selftest suite...")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Тест 1: Синус 100 Hz → pitch ~100
    print("\n1. Sine 100 Hz test...")
    sine_100 = create_sine(100, DURATION, SAMPLE_RATE)
    checks = {
        "pitch ~100 Hz": lambda m: abs(m.get('f0_median', 0) - 100) < 20,
        "voiced > 0.5": lambda m: m.get('voiced_fraction', 0) > 0.5
    }
    if run_test("sine_100hz", sine_100, checks, results_dir):
        tests_passed += 1
    tests_total += 1
    
    # Тест 2: Гармоники → гармоники видны
    print("\n2. Harmonics test...")
    harmonics = create_harmonics(100, 5, DURATION, SAMPLE_RATE)
    checks = {
        "pitch ~100 Hz": lambda m: abs(m.get('f0_median', 0) - 100) < 30,
        "voiced > 0.3": lambda m: m.get('voiced_fraction', 0) > 0.3
    }
    if run_test("harmonics", harmonics, checks, results_dir):
        tests_passed += 1
    tests_total += 1
    
    # Тест 3: Тишина → pitch=0, voiced~0
    print("\n3. Silence test...")
    silence = create_silence(DURATION, SAMPLE_RATE)
    checks = {
        "pitch = 0": lambda m: m.get('f0_median', 1) < 10,
        "voiced < 0.1": lambda m: m.get('voiced_fraction', 1) < 0.1
    }
    if run_test("silence", silence, checks, results_dir):
        tests_passed += 1
    tests_total += 1
    
    # Тест 4: Белый шум → pitch unstable/0 + warning
    print("\n4. White noise test...")
    noise = create_noise(DURATION, SAMPLE_RATE)
    checks = {
        "pitch unstable or 0": lambda m: m.get('f0_median', 100) < 50 or m.get('voiced_fraction', 1) < 0.2
    }
    if run_test("noise", noise, checks, results_dir):
        tests_passed += 1
    tests_total += 1
    
    # Итоги
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {tests_total - tests_passed} test(s) failed")
        return 1

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/voice_analyze_selftest"
    exit(run_selftest(results_dir))

