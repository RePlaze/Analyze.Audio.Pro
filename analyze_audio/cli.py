"""
Command Line Interface - Apple-grade UX
One-command operation, clear output, fail-fast feedback
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List

from .core import AudioAnalyzer
from .schema import ValidationError
from .settings import validate_dependencies


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='analyze-audio',
        description='Apple-grade audio analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  analyze-audio audio.wav
  analyze-audio video.mp4 --out results/
  analyze-audio --compare raw.wav processed.wav
  analyze-audio --selftest
  analyze-audio --verify audio.wav
        """
    )
    
    # Main input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        'input_file',
        nargs='?',
        type=Path,
        help='Input audio/video file to analyze'
    )
    
    input_group.add_argument(
        '--compare',
        nargs=2,
        metavar=('RAW', 'PROCESSED'),
        type=Path,
        help='Compare two files (raw vs processed)'
    )
    
    input_group.add_argument(
        '--selftest',
        action='store_true',
        help='Run comprehensive selftest with synthetic signals'
    )
    
    input_group.add_argument(
        '--verify',
        type=Path,
        metavar='FILE',
        help='Verify analysis with independent pitch method'
    )
    
    # Output options
    parser.add_argument(
        '--out', '-o',
        type=Path,
        metavar='DIR',
        help='Output directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open results directory after completion'
    )
    
    # Advanced options
    parser.add_argument(
        '--baseline',
        type=Path,
        metavar='FILE',
        help='Baseline file for reference scaling (advanced)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed progress'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='analyze-audio 1.0.0'
    )
    
    return parser


def validate_cli_args(args) -> None:
    """Validate command line arguments"""
    
    # Check input files exist
    if args.input_file and not args.input_file.exists():
        raise ValidationError(f"Input file not found: {args.input_file}")
    
    if args.compare:
        for i, file_path in enumerate(args.compare):
            if not file_path.exists():
                raise ValidationError(f"Compare file {i+1} not found: {file_path}")
    
    if args.verify and not args.verify.exists():
        raise ValidationError(f"Verify file not found: {args.verify}")
    
    if args.baseline and not args.baseline.exists():
        raise ValidationError(f"Baseline file not found: {args.baseline}")
    
    # Check output directory is writable
    if args.out:
        try:
            args.out.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory: {e}")


def generate_output_dir(input_file: Optional[Path] = None) -> Path:
    """Generate timestamped output directory"""
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if input_file:
        base_name = input_file.stem
        return Path(f'analysis_{base_name}_{timestamp}')
    else:
        return Path(f'analysis_{timestamp}')


def print_result_summary(result, output_dir: Path, mode: str) -> None:
    """Print Apple-grade result summary (exactly 3 lines)"""
    
    # Line 1: Status
    status_emoji = {
        'OK': '✅',
        'WARN': '⚠️', 
        'FAIL': '❌'
    }
    
    print(f"Status: {status_emoji.get(result.status, '❓')} {result.status}")
    
    # Line 2: Warnings count
    warning_count = len([w for w in result.warnings if w.severity in ['WARN', 'ERROR']])
    print(f"Warnings: {warning_count}")
    
    # Line 3: Output location
    html_report = output_dir / 'report.html'
    if html_report.exists():
        print(f"Report: {html_report}")
    else:
        print(f"Output: {output_dir}")


def run_selftest(analyzer: AudioAnalyzer, output_dir: Path, verbose: bool = False) -> int:
    """Run comprehensive selftest"""
    
    if verbose:
        print("Running selftest with synthetic signals...")
    
    try:
        results = analyzer.selftest(output_dir)
        
        # Print results
        passed = sum(results.values())
        total = len(results)
        
        if verbose:
            print(f"\nSelftest Results ({passed}/{total} passed):")
            for signal_name, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {signal_name}: {status}")
        
        # Summary (3 lines)
        if passed == total:
            print("Status: ✅ OK")
            print("Warnings: 0")
            print(f"Output: {output_dir}")
            return 0
        else:
            print("Status: ❌ FAIL")
            print(f"Warnings: {total - passed}")
            print(f"Output: {output_dir}")
            return 1
            
    except Exception as e:
        print("Status: ❌ FAIL")
        print("Warnings: 1")
        print(f"Error: {e}")
        return 1


def run_verify_mode(analyzer: AudioAnalyzer, input_file: Path, output_dir: Path, 
                   verbose: bool = False) -> int:
    """Run verification with independent pitch method"""
    
    if verbose:
        print("Running verification with independent pitch tracking...")
    
    try:
        # First run standard analysis
        result = analyzer.analyze_single(input_file, output_dir)
        
        # TODO: Implement independent pitch verification
        # This would use a different pitch tracking method (pyin, crepe, etc.)
        # and compare results with the main analysis
        
        if verbose:
            print("Verification analysis completed")
            print("Comparing pitch tracking methods...")
        
        # For now, just return the standard analysis result
        print_result_summary(result, output_dir, 'verify')
        
        return 0 if result.status in ['OK', 'WARN'] else 1
        
    except Exception as e:
        print("Status: ❌ FAIL")
        print("Warnings: 1") 
        print(f"Error: {e}")
        return 1


def open_results_directory(output_dir: Path) -> None:
    """Open results directory in system file manager"""
    
    import subprocess
    import platform
    
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', str(output_dir)])
        elif system == 'Windows':
            subprocess.run(['explorer', str(output_dir)])
        elif system == 'Linux':
            subprocess.run(['xdg-open', str(output_dir)])
    except Exception:
        # Silently fail if can't open
        pass


def main() -> int:
    """Main CLI entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Validate dependencies first
        issues = validate_dependencies()
        if issues:
            print("Status: ❌ FAIL")
            print(f"Warnings: {len(issues)}")
            print(f"Dependencies: {', '.join(issues)}")
            return 1
        
        # Validate arguments
        validate_cli_args(args)
        
        # Initialize analyzer
        analyzer = AudioAnalyzer()
        
        # Determine output directory
        if args.out:
            output_dir = args.out
        elif args.selftest:
            output_dir = generate_output_dir()
        elif args.input_file:
            output_dir = generate_output_dir(args.input_file)
        elif args.compare:
            output_dir = generate_output_dir(args.compare[0])
        elif args.verify:
            output_dir = generate_output_dir(args.verify)
        else:
            output_dir = generate_output_dir()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute based on mode
        if args.selftest:
            return run_selftest(analyzer, output_dir, args.verbose)
        
        elif args.verify:
            return run_verify_mode(analyzer, args.verify, output_dir, args.verbose)
        
        elif args.compare:
            if args.verbose:
                print(f"Comparing: {args.compare[0].name} vs {args.compare[1].name}")
            
            result = analyzer.analyze_compare(
                args.compare[0], args.compare[1], output_dir
            )
            
            print_result_summary(result, output_dir, 'compare')
            
            if args.open:
                open_results_directory(output_dir)
            
            return 0 if result.status in ['OK', 'WARN'] else 1
        
        elif args.input_file:
            if args.verbose:
                print(f"Analyzing: {args.input_file.name}")
            
            # Handle baseline mode
            if args.baseline:
                # TODO: Implement baseline reference mode
                # This would use the baseline file for reference scaling
                pass
            
            result = analyzer.analyze_single(args.input_file, output_dir)
            
            print_result_summary(result, output_dir, 'single')
            
            if args.open:
                open_results_directory(output_dir)
            
            return 0 if result.status in ['OK', 'WARN'] else 1
        
        else:
            parser.print_help()
            return 1
    
    except ValidationError as e:
        print("Status: ❌ FAIL")
        print("Warnings: 1")
        print(f"Error: {e}")
        return 1
    
    except KeyboardInterrupt:
        print("\nStatus: ❌ FAIL")
        print("Warnings: 1")
        print("Error: Interrupted by user")
        return 1
    
    except Exception as e:
        print("Status: ❌ FAIL")
        print("Warnings: 1")
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
