#!/usr/bin/env python3
"""
Test CareCredit Treatment Translator Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.carecredit import test_carecredit

if __name__ == "__main__":
    print("🚀 Testing CareCredit Treatment Translator Agent")
    print("=" * 50)
    
    success = test_carecredit()
    
    if success:
        print("\n✅ All CareCredit tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some CareCredit tests failed!")
        sys.exit(1)