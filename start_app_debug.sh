#!/bin/bash

# AI Assistant Chatbot - Debug Mode Startup Script
echo "🚀 Starting AI Assistant Chatbot in Debug Mode..."

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

# Kill processes on both ports
kill_port 5001 "Flask"
kill_port 8501 "Streamlit"

# Kill any remaining Python processes that might be running our apps
echo "🔪 Killing any remaining app processes..."
pkill -f "python flask_app.py" 2>/dev/null
pkill -f "streamlit run streamlit_app.py" 2>/dev/null

# Final verification and force kill if needed
sleep 2
if command -v lsof > /dev/null 2>&1; then
  if lsof -ti:5001 > /dev/null 2>&1; then
    echo "⚠️  Port 5001 still in use, force killing..."
    lsof -ti:5001 | xargs kill -9 2>/dev/null
    sleep 1
  fi
  
  if lsof -ti:8501 > /dev/null 2>&1; then
    echo "⚠️  Port 8501 still in use, force killing..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 1
  fi
fi

echo "✅ Old servers cleaned up"

# Start Flask backend in background
echo "🌐 Starting Flask backend on port 5001..."
python flask_app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 5

# Check if Flask is running
if curl -s http://localhost:5001/health > /dev/null; then
  echo "✅ Flask backend is running successfully"
else
  echo "❌ Flask backend failed to start. Check the logs above."
  echo "💡 If port 5001 is in use, try running the script again to kill old processes"
  kill $FLASK_PID 2>/dev/null
  exit 1
fi

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port 8501..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo "🔧 Flask API available at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
  echo ""
  echo "🛑 Shutting down services..."
  kill $FLASK_PID 2>/dev/null
  echo "✅ All services stopped"
  exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
