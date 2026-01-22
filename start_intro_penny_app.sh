#!/bin/bash

# Intro Penny App - Startup Script
echo "🚀 Starting Intro Penny Chat App..."

# Check if setup has been run
if [ ! -d "venv" ] || [ ! -f ".env" ]; then
  echo "⚠️  Setup not complete. Please run ./setup_app.sh first"
  exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Kill old servers
echo "🛑 Checking for and killing old servers..."

# Function to kill processes on a port
kill_port() {
  local port=$1
  local service_name=$2
  
  # Try lsof first
  if command -v lsof > /dev/null 2>&1; then
    if lsof -ti:$port > /dev/null 2>&1; then
      echo "🔪 Found processes on port $port ($service_name), killing them..."
      PIDS=$(lsof -ti:$port)
      echo "   PIDs: $PIDS"
      echo $PIDS | xargs kill -9 2>/dev/null
      sleep 2
    fi
  else
    # Fallback to netstat if lsof is not available
    if command -v netstat > /dev/null 2>&1; then
      PIDS=$(netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | grep -v '-' | sort -u)
      if [ ! -z "$PIDS" ]; then
        echo "🔪 Found processes on port $port ($service_name), killing them..."
        echo "   PIDs: $PIDS"
        echo $PIDS | xargs kill -9 2>/dev/null
        sleep 2
      fi
    fi
  fi
}

# Kill processes on port 8502 (Intro Penny App)
kill_port 8502 "Intro Penny App"

# Kill any remaining Python processes that might be running our app
echo "🔪 Killing any remaining app processes..."
pkill -f "streamlit run intro_penny_app.py" 2>/dev/null

# Final verification and force kill if needed
sleep 2
if command -v lsof > /dev/null 2>&1; then
  if lsof -ti:8502 > /dev/null 2>&1; then
    echo "⚠️  Port 8502 still in use, force killing..."
    lsof -ti:8502 | xargs kill -9 2>/dev/null
    sleep 1
  fi
fi

echo "✅ Old servers cleaned up"

# Start Streamlit frontend
echo "🎨 Starting Intro Penny Chat App on port 8502..."
echo "📱 Application available at: http://0.0.0.0:8502"
echo ""
echo "Press Ctrl+C to stop the service"

# Function to cleanup on exit
cleanup() {
  echo ""
  echo "🛑 Shutting down service..."
  echo "✅ Service stopped"
  exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Streamlit
streamlit run intro_penny_app.py --server.port 8502 --server.address 0.0.0.0
