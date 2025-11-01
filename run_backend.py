#!/usr/bin/env python3
"""
Backend startup script for DATAmat
This script ensures proper setup before starting the server.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are met before starting the server."""
    errors = []
    warnings = []
    
    # Check for AI21_API_KEY
    if not os.getenv("AI21_API_KEY"):
        errors.append("AI21_API_KEY environment variable is not set")
    
    # Check for datasets directory
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        warnings.append(f"Creating datasets directory: {datasets_dir}")
        datasets_dir.mkdir(exist_ok=True)
    
    # Check if datasets directory is empty (warning, not error)
    if datasets_dir.exists():
        supported_extensions = ('.csv', '.pdf', '.json', '.txt', '.xlsx', '.xls')
        has_files = any(datasets_dir.rglob(f"*{ext}") for ext in supported_extensions)
        if not has_files:
            warnings.append("No datasets found. You can upload datasets via the frontend.")
    
    # Print errors and warnings
    if errors:
        print("❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 To set AI21_API_KEY:")
        print("  Windows PowerShell: $env:AI21_API_KEY='your-api-key'")
        print("  Windows CMD: set AI21_API_KEY=your-api-key")
        print("  Linux/Mac: export AI21_API_KEY=your-api-key")
        return False
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    return True

if __name__ == "__main__":
    print("🚀 Starting DATAmat Backend Server...\n")
    
    if not check_requirements():
        print("\n❌ Cannot start server. Please fix the errors above.")
        sys.exit(1)
    
    print("✅ Requirements check passed!\n")
    print("🌐 Starting server on http://localhost:8001")
    print("📚 API docs will be available at http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Import and run the server
    import uvicorn
    from src.datamat.ai21lab_api import app
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

