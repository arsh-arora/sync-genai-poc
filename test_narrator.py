#!/usr/bin/env python3
"""
Test Portfolio Intel Narrator Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.narrator import test_narrator

if __name__ == "__main__":
    print("🚀 Testing Portfolio Intel Narrator Agent")
    print("=" * 50)
    
    success = test_narrator()
    
    if success:
        print("\n✅ All Portfolio Intel Narrator tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some Portfolio Intel Narrator tests failed!")
        sys.exit(1)