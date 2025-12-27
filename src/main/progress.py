"""
Progress bar для терминала в стиле Apple
"""

import sys
import time
from typing import Optional

class ProgressBar:
    """Простой progress bar для терминала"""
    
    def __init__(self, total: int, desc: str = "Processing", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Обновить прогресс"""
        self.current = min(self.current + n, self.total)
        self._display()
    
    def _display(self):
        """Отобразить progress bar"""
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        sys.stdout.write(f'\r{self.desc}: [{bar}] {percent*100:.1f}% ({self.current}/{self.total}) {eta_str}')
        sys.stdout.flush()
    
    def finish(self):
        """Завершить progress bar"""
        self.current = self.total
        self._display()
        sys.stdout.write('\n')
        sys.stdout.flush()

