"""
Benchmark Runner Entry Point

Main entry point for running benchmarks via python -m benchmarks
"""

import asyncio
import logging
import sys

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import after logging is set up to avoid circular import issues
    from benchmarks.runner import run_default_benchmarks
    
    # Run benchmarks
    try:
        asyncio.run(run_default_benchmarks())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
