"""
Генерация отчетов — Report V2 System
Apple-grade visual output
"""

from pathlib import Path
from typing import List, Dict, Any

from .voice_analyzer import VoiceAnalyzer
from .report_v2 import ReportV2Generator, create_report_v2


class ReportGenerator:
    """Generates visual reports with Apple-grade quality"""
    
    def __init__(self, use_professional_system: bool = True, theme: str = "light"):
        self.theme = theme
        self.voice_analyzer = VoiceAnalyzer()
        self.report_v2 = ReportV2Generator(theme=theme, lang='en')
        print("🍎 Using Apple-grade Report V2 system")
    
    def create_report(self, audio_files: List[Path], file_names: List[str], 
                     output_dir: Path, mode: str = 'single') -> Dict[str, Any]:
        """Create complete report: HTML (EN + RU)"""
        
        output_dir = Path(output_dir)
        
        try:
            results = create_report_v2(
                output_dir=output_dir,
                theme=self.theme,
                lang='en'
            )
            
            if results.get('success'):
                print("🍎 Apple-grade Report V2 generated successfully")
                print(f"   EN: {results.get('html_en', 'report_en.html')}")
                print(f"   RU: {results.get('html_ru', 'report_ru.html')}")
                return results
            else:
                print(f"❌ Report V2 failed: {results.get('error')}")
                return {'success': False, 'error': results.get('error')}
                
        except Exception as e:
            print(f"❌ Report V2 failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
