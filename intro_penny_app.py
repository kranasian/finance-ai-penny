import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the optimizer
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv()

# Import the optimizer
from experiments.intro_penny_optimizer import IntroPennyOptimizer, extract_python_code

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "optimizer" not in st.session_state:
    try:
        st.session_state.optimizer = IntroPennyOptimizer()
    except Exception as e:
        st.error(f"Failed to initialize optimizer: {str(e)}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Intro Penny Chat",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Intro Penny Chat")
st.markdown("Chat with your financial planner agent!")

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant responses
        if message["role"] == "assistant" and "test_case_data" in message:
            with st.expander("📋 Test Case Metadata", expanded=False):
                test_case = message["test_case_data"]
                # Format as Python dict with proper escaping
                # Escape backslashes and quotes in double-quoted string (last_user_request)
                escaped_request = test_case["last_user_request"].replace("\\", "\\\\").replace('"', '\\"')
                # Triple-quoted strings don't need escaping, but handle triple quotes if present
                escaped_conv = test_case["previous_conversation"]
                if '"""' in escaped_conv:
                    # Replace triple quotes with escaped version (rare edge case)
                    escaped_conv = escaped_conv.replace('"""', '\\"\\"\\"')
                
                formatted = f'''{{
  "name": "{test_case["name"]}",
  "last_user_request": "{escaped_request}",
  "previous_conversation": """{escaped_conv}"""
}}'''
                st.code(formatted, language="python")
                # Add copy button
                st.caption("💡 Click the code block above and copy (Cmd/Ctrl+C)")

# Chat input
if prompt := st.chat_input("Ask me about your finances..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Format previous conversation from session state
                # Skip the last message (current user message) when building previous conversation
                previous_conversation = ""
                for i, msg in enumerate(st.session_state.messages[:-1]):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    previous_conversation += f"{role}: {msg['content']}\n"
                
                # Get the last user request (current prompt)
                last_user_request = prompt
                
                # Generate response using the optimizer
                generated_code = st.session_state.optimizer.generate_response(
                    last_user_request=last_user_request,
                    previous_conversation=previous_conversation.strip()
                )
                
                # Extract Python code from the generated response
                code = extract_python_code(generated_code)
                
                if code:
                    try:
                        # Create a namespace for executing the code
                        namespace = {}
                        
                        # Execute the generated code
                        exec(code, namespace)
                        
                        # Call execute_plan if it exists
                        if 'execute_plan' in namespace:
                            success, response = namespace['execute_plan']()
                            
                            # Generate test case name from the user request
                            test_case_name = last_user_request.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").replace(",", "")[:50]
                            
                            # Create test case data
                            test_case_data = {
                                "name": test_case_name,
                                "last_user_request": last_user_request,
                                "previous_conversation": previous_conversation.strip()
                            }
                            
                            # Display the response
                            if success:
                                st.markdown(response)
                                # Add assistant response to chat history with metadata
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response,
                                    "test_case_data": test_case_data
                                })
                            else:
                                st.warning(response)
                                # Add assistant response to chat history even if not successful
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response,
                                    "test_case_data": test_case_data
                                })
                        else:
                            error_msg = "Error: execute_plan() function not found in generated code"
                            st.error(error_msg)
                            st.code(generated_code, language="python")
                            # Generate test case data even for errors
                            test_case_name = last_user_request.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").replace(",", "")[:50]
                            test_case_data = {
                                "name": test_case_name,
                                "last_user_request": last_user_request,
                                "previous_conversation": previous_conversation.strip()
                            }
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg,
                                "test_case_data": test_case_data
                            })
                    except Exception as e:
                        error_msg = f"Error executing generated code: {str(e)}"
                        st.error(error_msg)
                        st.code(code, language="python")
                        # Generate test case data even for errors
                        test_case_name = last_user_request.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").replace(",", "")[:50]
                        test_case_data = {
                            "name": test_case_name,
                            "last_user_request": last_user_request,
                            "previous_conversation": previous_conversation.strip()
                        }
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "test_case_data": test_case_data
                        })
                else:
                    error_msg = "Error: No Python code found in generated response"
                    st.error(error_msg)
                    st.code(generated_code, language="text")
                    # Generate test case data even for errors
                    test_case_name = last_user_request.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").replace(",", "")[:50]
                    test_case_data = {
                        "name": test_case_name,
                        "last_user_request": last_user_request,
                        "previous_conversation": previous_conversation.strip()
                    }
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "test_case_data": test_case_data
                    })
                    
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                # Try to get test case data if available
                try:
                    test_case_name = prompt.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "").replace(",", "")[:50]
                    previous_conversation = ""
                    for msg in st.session_state.messages[:-1]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        previous_conversation += f"{role}: {msg['content']}\n"
                    test_case_data = {
                        "name": test_case_name,
                        "last_user_request": prompt,
                        "previous_conversation": previous_conversation.strip()
                    }
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "test_case_data": test_case_data
                    })
                except:
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses the Intro Penny Optimizer to help you with your financial planning. It maintains conversation context to provide relevant responses.")
    
    # Display conversation stats
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### Conversation Stats")
        user_messages = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
        assistant_messages = sum(1 for msg in st.session_state.messages if msg["role"] == "assistant")
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("User Messages", user_messages)
        st.metric("Assistant Messages", assistant_messages)
