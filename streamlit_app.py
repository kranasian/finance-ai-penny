import streamlit as st
import requests
import json
import time

from streamlit_app.error_handling import format_error_message, display_error, is_error_response

# Configure Streamlit page
st.set_page_config(
  page_title="AI Assistant Chatbot",
  page_icon="🤖",
  layout="wide"
)

# Flask backend URL
FLASK_URL = "http://localhost:5001"

def send_message_to_backend(message, username="default_user", mode="function_calling", model="gemini-2.0-flash", messages=None):
  """Send message to Flask backend"""
  try:
    # Include session messages if provided
    request_data = {
      "message": message, 
      "username": username, 
      "mode": mode, 
      "model": model
    }
    
    if messages is not None:
      request_data["messages"] = messages
    
    response = requests.post(
      f"{FLASK_URL}/chat",
      json=request_data,
      timeout=30
    )
    return response.json()
  except requests.exceptions.RequestException as e:
    return {"error": f"Failed to connect to backend: {str(e)}"}

def get_users():
  """Get all users from backend"""
  try:
    response = requests.get(f"{FLASK_URL}/users")
    return response.json()
  except requests.exceptions.RequestException as e:
    return {"error": f"Failed to get users: {str(e)}"}

def process_prompt(prompt):
  """Process a prompt and add it to chat"""
  # Add user message to chat history
  st.session_state.messages.append({
    "role": "user", 
    "content": prompt,
    "request_time": time.time(),
    "timing": None
  })
  
  # Display user message
  with st.chat_message("user"):
    st.markdown(prompt)
  
  # Get AI response
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      response_data = send_message_to_backend(
        prompt, 
        st.session_state.username, 
        st.session_state.current_mode, 
        st.session_state.current_model, 
        st.session_state.messages
      )
      
      if "error" in response_data:
        # Format error for better readability
        display_error(response_data["error"], "⚠️ **Error occurred**")
        main_error, _ = format_error_message(response_data["error"])
        response_content = f"Sorry, I encountered an error: {main_error or response_data['error']}"
      else:
        response_content = response_data.get("response", "I'm sorry, I didn't understand that.")
        
        # Check if response contains error information
        if is_error_response(response_content):
          # Format error response for better readability
          display_error(response_content, "⚠️ **Execution Error**")
        else:
          st.text(response_content)
        
        # Show function call information
        if response_data.get("function_called"):
          with st.expander(f"🔧 Function Called: {response_data['function_called']}"):
            st.json(response_data.get("function_result", {}))
        
        # Show logs if available
        if response_data.get("logs"):
          with st.expander("📋 Execution Logs"):
            st.markdown(response_data['logs'])
  
  # Add assistant response to chat history
  st.session_state.messages.append({
    "role": "assistant", 
    "content": response_content,
    "function_called": response_data.get("function_called"),
    "function_result": response_data.get("function_result"),
    "code_generated": response_data.get("code_generated"),
    "logs": response_data.get("logs"),
    "timing": response_data.get("timing"),
    "request_time": time.time()
  })

def render_example_prompts():
  """Render example prompts for checking account balances"""
  st.markdown("##### 💡 Example Prompts")
  
  example_prompts = [
    # General account balance questions
    "What is my account balance?",
    "Show me all my account balances",
    "What are my current balances?",
    "Check my account balances",
    "Do I have accounts with current balance over 5k?",
    "How much is my credit limit?",
    "What is the balance of my Amex accounts?",
    "What is the balance of my BoF and Citibank accounts?",
    # General transaction questions
    "What are my recent transactions?",
    "What are my recent Amex transactions?",
    "Do I have dining out transactions with amount over $40 last month?",
    "List my dining out transactions last month.",
    "List income past 2 weeks.",
    # General comparison questions
    "Compare my dining out and groceries spending last month.",
    "Compare my dining out last sept 2025 vs oct 2025.",
    "Compare my spending on entertainment and travel to my bills and medicines last month.",
    "Compare how much I earned last month to September 2025."
  ]
  
  cols = st.columns(4)
  for idx, prompt in enumerate(example_prompts):
    col = cols[idx % 4]
    with col:
      if st.button(prompt, key=f"example_prompt_{idx}", use_container_width=True):
        # Store prompt in session state to process it
        st.session_state.pending_prompt = prompt
        st.rerun()
  
  # Process pending prompt if one exists
  if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt_to_process = st.session_state.pending_prompt
    del st.session_state.pending_prompt
    process_prompt(prompt_to_process)

def render_chat_input():
  """Render the chat input"""
  placeholder_text = "Ask me to help with code generation or programming!"
  prompt = st.chat_input(placeholder_text)
  if prompt:
    process_prompt(prompt)

def main():
  
  # Initialize session state
  if "messages" not in st.session_state:
    st.session_state.messages = []
  if "username" not in st.session_state:
    st.session_state.username = "EmptyUser"  # Default to first seeded user
  
  # Set AI Mode to Proposed option (Code Gen: Gemini Flash Lite Latest)
  st.session_state.current_mode = "code_gen"
  st.session_state.current_model = "gemini-flash-lite-latest"

  
  # Render the chat interface
  render_chat_interface()
  
  # Example prompts
  render_example_prompts()
  
  # Chat input
  render_chat_input()

def render_chat_interface():
  """Render the chat interface"""
  mode = st.session_state.current_mode
  
  # Sidebar for user settings and reminders (available for all modes)
  with st.sidebar:
    st.header("App Controls")
    
    # Refresh UI button
    if st.button("🔄 Refresh UI", help="Reload the entire Streamlit app"):
      st.rerun()
    
    st.markdown("---")
    st.header("Set Context")
    
    # User dropdown
    users_data = get_users()
    if "users" in users_data:
      users = users_data["users"]
      if users:
        user_options = [user["username"] for user in users]
        selected_user = st.selectbox(
          "Select User",
          options=user_options,
          index=user_options.index(st.session_state.username) if st.session_state.username in user_options else 0,
          help="Choose a user to view their reminders and create new ones"
        )
        st.session_state.username = selected_user
      else:
        st.error("No users found")
    else:
      st.error("Failed to load users")

    st.markdown("""
    <div style="font-size: 10px; line-height: 1.8em; color: #aaa;">
      <p><b>Disclaimer:</b> I only did the bare minimum on the prompts, so don't use it as a barrometer for comparing quality between approaches.</p>
    </div>
    """, unsafe_allow_html=True)

  # Display chat messages
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      content = message.get("content", "")
      
      # Check if content contains error information and format it nicely
      if message["role"] == "assistant" and is_error_response(content):
        # Format error for better readability
        display_error(content, "⚠️ **Error occurred**")
      else:
        st.text(content)
      
      # Show function call results if available
      if "function_called" in message and message["function_called"]:
        with st.expander(f"🔧 Function Called: {message['function_called']}"):
          st.json(message.get("function_result", {}))
      
      # Show generated code if available (for code generation mode)
      if "code_generated" in message and message["code_generated"]:
        with st.expander("💻 Generated Code"):
          st.code(message["code_generated"], language="python")
      
      # Show logs if available
      if "logs" in message and message["logs"]:
        with st.expander("📋 Execution Logs"):
          st.markdown(message['logs'])
      
      # Show timing information if available
      if ("timing" in message and 
          message["timing"] is not None and
          "backend_processing_start" in message["timing"] and
          "request_received" in message["timing"] and
          "total_processing_time" in message["timing"] and
          "end_to_end_latency" in message["timing"]):
        with st.expander("⏱️ Performance Metrics"):
          timing = message["timing"]
          
          # Calculate backend processing start latency
          backend_start_latency = (timing["backend_processing_start"] - timing["request_received"]) * 1000
          
          col1, col2, col3 = st.columns(3)
          
          with col1:
            st.metric(
              label="Backend Processing Start",
              value=f"{backend_start_latency:.1f}ms",
              help="Time from request received until backend starts processing"
            )
          
          with col2:
            st.metric(
              label="Total Processing Time",
              value=f"{timing['total_processing_time']:.1f}ms",
              help="Time spent in backend processing"
            )
          
          with col3:
            st.metric(
              label="End-to-End Latency",
              value=f"{timing['end_to_end_latency']:.1f}ms",
              help="Total time from request to response"
            )


          col4, col5 = st.columns(2)
          
          with col4:
            if timing.get("gemini_api_calls"):
              total_gemini_time = sum(call["duration_ms"] for call in timing["gemini_api_calls"])
              st.metric("Gemini API", f"{total_gemini_time:.1f}ms")
            else:
              st.metric("Gemini API", "")      
        
            # Show Gemini API call details if available
            if timing.get("gemini_api_calls") and len(timing["gemini_api_calls"]) > 1:
              st.subheader("🤖 Gemini API Calls")
              for i, call in enumerate(timing["gemini_api_calls"]):
                st.metric(
                  label=f"API Call #{call['call_number']}",
                  value=f"{call['duration_ms']:.1f}ms",
                  help=f"Duration of Gemini API call #{call['call_number']}"
                )
          
          with col5:
            if timing.get("execution_time"):
              total_execution_time = sum(call["duration_ms"] for call in timing["execution_time"])
              st.metric("Execution Time", f"{total_execution_time:.1f}ms")
            else:
              st.metric("Execution Time", "")
          
            # Show execution time details if available
            if timing.get("execution_time") and len(timing["execution_time"]) > 1:
              st.subheader("🔧 Execution Time")
              for i, call in enumerate(timing["execution_time"]):
                st.metric(
                  label=f"Execution #{call['call_number']}",
                  value=f"{call['duration_ms']:.1f}ms",
                  help=f"Duration of execution #{call['call_number']}"
                )
  
  

if __name__ == "__main__":
  main()
