"""
server/app.py — OpenEnv server entry point
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import HTML, Handler
from http.server import HTTPServer

def main():
    print("WordPuzzle environment running on port 7860...")
    HTTPServer(("0.0.0.0", 7860), Handler).serve_forever()

if __name__ == "__main__":
    main()
