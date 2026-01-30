#!/bin/bash

# Agent Categorize App - Startup Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Agent Categorize App..."

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

kill_port() {
  local port=$1
  local service_name=$2
  if command -v lsof > /dev/null 2>&1; then
    if lsof -ti:$port > /dev/null 2>&1; then
      echo "🔪 Found processes on port $port ($service_name), killing them..."
      PIDS=$(lsof -ti:$port)
      echo $PIDS | xargs kill -9 2>/dev/null
      sleep 2
    fi
  else
    if command -v netstat > /dev/null 2>&1; then
      PIDS=$(netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | grep -v '-' | sort -u)
      if [ ! -z "$PIDS" ]; then
        echo "🔪 Found processes on port $port ($service_name), killing them..."
        echo $PIDS | xargs kill -9 2>/dev/null
        sleep 2
      fi
    fi
  fi
}

kill_port 8503 "Agent Categorize App"
pkill -f "streamlit run agent_categorize_app.py" 2>/dev/null

sleep 2
if command -v lsof > /dev/null 2>&1; then
  if lsof -ti:8503 > /dev/null 2>&1; then
    echo "⚠️  Port 8503 still in use, force killing..."
    lsof -ti:8503 | xargs kill -9 2>/dev/null
    sleep 1
  fi
fi

echo "✅ Old servers cleaned up"

echo "🎨 Starting Agent Categorize App on port 8503..."
echo "📱 Application available at: http://0.0.0.0:8503"
echo ""
echo "Press Ctrl+C to stop the service"

cleanup() {
  echo ""
  echo "🛑 Shutting down service..."
  echo "✅ Service stopped"
  exit 0
}

trap cleanup SIGINT SIGTERM

streamlit run agent_categorize_app.py --server.port 8503 --server.address 0.0.0.0
