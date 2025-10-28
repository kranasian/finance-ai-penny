from database import Database
from fastrack.reminders_simple import create_reminder_from_message
from flask import Flask, request, jsonify
from gemini_agent_code_gen import create_gemini_agent_code_gen
from user_seeder import seed_users
import json
import logging
import time
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed users when the app starts
print("Seeding users...")
seed_users()

db = Database()


@app.route('/chat', methods=['POST'])
def chat():
  """Handle chat messages with model selection"""
  # Start timing
  start_time = time.time()
  timing_data = {
    'request_received': start_time,
    'backend_processing_start': None,
    'gemini_api_calls': [],
    'execution_time': [],
    'total_processing_time': None,
    'end_to_end_latency': None
  }
  
  try:
    data = request.get_json()
    user_message = data.get('message', '')
    username = data.get('username', 'default_user')
    model_name = data.get('model', 'gemini-2.0-flash')
    session_messages = data.get('messages', [])  # Get st.session_state.messages
    
    # Filter messages to only include those sent within the last 30 seconds
    current_time = time.time()
    recent_messages = []
    
    for msg in session_messages:
      # Check if message has request_time and is within 30 seconds
      if "request_time" in msg and (current_time - msg["request_time"]) <= 30:
        recent_messages.append(msg)
      # If no request_time, include it (backward compatibility)
      elif "request_time" not in msg:
        recent_messages.append(msg)
    
    # Log the session messages for debugging/analysis
    logger.info(f"Chat request from user '{username}' with {len(session_messages)} total messages, {len(recent_messages)} recent messages (filtered from last 30 seconds)")
    
    # Log message timing information
    for i, msg in enumerate(recent_messages):
      if "request_time" in msg:
        age_seconds = current_time - msg["request_time"]
        logger.info(f"Recent message {i+1} ({msg['role']}): {age_seconds:.1f}s old")
      else:
        logger.info(f"Recent message {i+1} ({msg['role']}): no timestamp (legacy)")
    
    logger.info(f"Recent session messages: {json.dumps(recent_messages, indent=2)}")
    
    # Mark backend processing start
    timing_data['backend_processing_start'] = time.time()
    
    # Get or create user
    user = db.get_user(username)
    if not user:
      user_id = db.create_user(username, f"{username}@example.com")
      user = db.get_user(username)

    # Use Gemini agent for code generation
    gemini_agent_code_gen = create_gemini_agent_code_gen(model_name)
    response_data = gemini_agent_code_gen.generate_response(
      recent_messages, 
      timing_data,
      user['id']
    )
    
    # Calculate timing
    end_time = time.time()
    timing_data['total_processing_time'] = (end_time - timing_data['backend_processing_start']) * 1000
    timing_data['end_to_end_latency'] = (end_time - start_time) * 1000
    
    return jsonify({
      **response_data,
      'timing': timing_data
    })
    
  except Exception as e:
    # Calculate timing even for errors
    end_time = time.time()
    timing_data['total_processing_time'] = (end_time - timing_data['backend_processing_start']) * 1000
    timing_data['end_to_end_latency'] = (end_time - start_time) * 1000
    
    # Log the error
    logger.error(f'An error occurred: {str(e)}')
    logger.error(f'Traceback: {traceback.format_exc()}')
    
    return jsonify({
      'error': f'An error occurred: {str(e)}',
      'timing': timing_data
    }), 500

@app.route('/users', methods=['GET'])
def get_users():
  """Get all users"""
  try:
    users = db.get_all_users()
    return jsonify({'users': users})
  except Exception as e:
    return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
  """Health check endpoint"""
  return jsonify({'status': 'healthy'})

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5001)
