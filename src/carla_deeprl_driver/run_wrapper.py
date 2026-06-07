#!/usr/bin/env python
import sys
import io

class OutputCapture:
    def __init__(self):
        self.outputs = []
    
    def write(self, text):
        self.outputs.append(text)
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
    
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = OutputCapture()
sys.stderr = OutputCapture()

print("Starting training script...")
print("=" * 50)

try:
    exec(open('run_sac.py').read())
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
