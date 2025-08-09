#!/usr/bin/env python3
"""
Test DevCopilot Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.devcopilot import test_devcopilot

if __name__ == "__main__":
    print("🚀 Testing DevCopilot Agent")
    print("=" * 50)
    
    success = test_devcopilot()
    
    if success:
        print("\n✅ All DevCopilot tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some DevCopilot tests failed!")
        sys.exit(1)
