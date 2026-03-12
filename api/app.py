import os
import sys

# Add parent directory to path to access modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel serverless handler
from vercel_python import handle

def handler(request, response):
    """Vercel serverless function handler"""
    with app.request_context(request.environ):
        return app(request.environ, response)

# For local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
