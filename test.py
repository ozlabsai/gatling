#!/usr/bin/env python3
"""
Quick API Test Script

Verifies your Anthropic API key is set up correctly.

Usage:
    uv run python test_api.py
"""

import os
import sys
from pathlib import Path


def load_env():
    """Load .env file if it exists"""
    env_path = Path('.env')
    if env_path.exists():
        print("✓ Found .env file")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip()
                    print(f"✓ Loaded {key}")
    else:
        print("⚠ No .env file found")
        print("  Create one with: echo 'ANTHROPIC_API_KEY=your-key' > .env")


def test_api():
    """Test API connection"""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n❌ anthropic package not installed")
        print("   Run: uv add anthropic")
        return False
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("\n❌ ANTHROPIC_API_KEY not found in environment")
        print("\nSteps to fix:")
        print("1. Go to https://console.anthropic.com/")
        print("2. Settings → API Keys → Create Key")
        print("3. Create .env file: echo 'ANTHROPIC_API_KEY=your-key' > .env")
        return False
    
    print(f"\n✓ API key found: {api_key[:20]}...")
    
    print("\n🧪 Testing API connection...")
    
    try:
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'API test successful!' and nothing else."}
            ]
        )
        
        result = response.content[0].text
        
        print(f"\n✅ API Response: {result}")
        print("\n" + "="*60)
        print("SUCCESS! Your API is working correctly.")
        print("="*60)
        print("\nYou can now run autonomous agents:")
        print("  uv run python agents/automated_runner.py --monitor")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ API Error: {str(e)}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- No internet connection")
        print("- API key doesn't have access to claude-sonnet-4-5")
        print("\nVerify at: https://console.anthropic.com/")
        return False


def main():
    print("="*60)
    print("ANTHROPIC API TEST")
    print("="*60 + "\n")
    
    # Load environment
    load_env()
    
    # Test API
    success = test_api()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())