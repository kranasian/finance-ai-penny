#!/bin/bash

# AI Assistant Chatbot - Setup Script
echo "🔧 Setting up AI Assistant Chatbot environment..."

# Check if .env file exists
if [ ! -f .env ]; then
  echo "⚠️  .env file not found. Creating from example..."
  if [ -f env_example.txt ]; then
    cp env_example.txt .env
    echo "📝 Please edit .env file and add your GEMINI_API_KEY"
    echo "   You can get your API key from: https://makersuite.google.com/app/apikey"
  else
    echo "❌ env_example.txt not found. Please create .env file manually."
    exit 1
  fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
# Try to install pandas with pre-compiled wheels first to avoid compilation issues
pip install --only-binary=all pandas==2.2.0 || pip install pandas==2.2.0
# Install remaining dependencies
pip install -r requirements.txt

# Check if Gemini API key is set
if ! grep -q "GEMINI_API_KEY=your_gemini_api_key_here" .env; then
  echo "✅ Gemini API key appears to be configured"
else
  echo "⚠️  Please configure your GEMINI_API_KEY in the .env file"
  echo "   Get your API key from: https://makersuite.google.com/app/apikey"
  echo "   Then edit .env file and replace 'your_gemini_api_key_here' with your actual key"
fi

echo "✅ Environment setup complete!"
echo "💡 You can now run ./start_app_debug.sh to start the application"
