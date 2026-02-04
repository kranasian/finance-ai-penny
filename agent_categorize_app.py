import streamlit as st
import os
import sys
import json
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

load_dotenv()

from experiments.agent_categorize_optimizer_v2 import (
    CategoryGrounderOptimizer,
    extract_python_code,
    TEST_CASES,
)
from penny.tool_funcs.update_transaction_category_or_create_category_rules import create_category_rules
from datetime import datetime

try:
    from penny.tool_funcs.date_utils import (
        get_date,
        get_start_of_month,
        get_end_of_month,
        get_start_of_year,
        get_end_of_year,
        get_start_of_week,
        get_end_of_week,
        get_after_periods,
        get_date_string,
    )
except ImportError:
    def get_date(y, m, d):
        return datetime(y, m, d)
    def get_date_string(d):
        return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
    def get_start_of_month(d):
        return d
    def get_end_of_month(d):
        return d
    def get_start_of_year(d):
        return d
    def get_end_of_year(d):
        return d
    def get_start_of_week(d):
        return d
    def get_end_of_week(d):
        return d
    def get_after_periods(d, granularity, count):
        return d

if "messages" not in st.session_state:
    st.session_state.messages = []

if "optimizer" not in st.session_state:
    try:
        st.session_state.optimizer = CategoryGrounderOptimizer()
    except Exception as e:
        st.error(f"Failed to initialize optimizer: {str(e)}")
        st.stop()

st.set_page_config(
    page_title="Agent Categorize",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Agent Categorize")
st.markdown("Create category rules for transactions (demo: rules are stubbed).")


def _exec_namespace():
    return {
        "create_category_rules": create_category_rules,
        "datetime": datetime,
        "get_date": get_date,
        "get_start_of_month": get_start_of_month,
        "get_end_of_month": get_end_of_month,
        "get_start_of_year": get_start_of_year,
        "get_end_of_year": get_end_of_year,
        "get_start_of_week": get_start_of_week,
        "get_end_of_week": get_end_of_week,
        "get_after_periods": get_after_periods,
        "get_date_string": get_date_string,
    }


def process_user_request(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                previous_conversation = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Penny"
                    previous_conversation += f"{role}: {msg['content']}\n"

                last_user_request = prompt
                generated_code = st.session_state.optimizer.generate_response(
                    last_user_request=last_user_request,
                    previous_conversation=previous_conversation.strip(),
                )
                code = extract_python_code(generated_code)

                if code:
                    try:
                        namespace = _exec_namespace()
                        exec(code, namespace)

                        if "execute_plan" in namespace:
                            success, response = namespace["execute_plan"]()
                            if success and not (response or "").strip():
                                response = "Rule(s) created successfully."

                            test_case_data = {
                                "name": last_user_request.lower().replace(" ", "_")[:50],
                                "last_user_request": last_user_request,
                                "previous_conversation": previous_conversation.strip(),
                            }

                            with st.expander("📜 Generated code", expanded=True):
                                st.code(code, language="python")

                            if success:
                                st.markdown(response)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response,
                                    "test_case_data": test_case_data,
                                    "generated_code": code,
                                })
                            else:
                                st.warning(response)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response,
                                    "test_case_data": test_case_data,
                                    "generated_code": code,
                                })
                        else:
                            error_msg = "execute_plan() not found in generated code"
                            st.error(error_msg)
                            with st.expander("📜 Generated code", expanded=True):
                                st.code(generated_code, language="python")
                            test_case_data = {
                                "name": last_user_request.lower().replace(" ", "_")[:50],
                                "last_user_request": last_user_request,
                                "previous_conversation": previous_conversation.strip(),
                            }
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "test_case_data": test_case_data,
                                "generated_code": code or generated_code,
                            })
                    except Exception as e:
                        error_msg = f"Error executing generated code: {str(e)}"
                        st.error(error_msg)
                        with st.expander("📜 Generated code", expanded=True):
                            st.code(code, language="python")
                        test_case_data = {
                            "name": last_user_request.lower().replace(" ", "_")[:50],
                            "last_user_request": last_user_request,
                            "previous_conversation": previous_conversation.strip(),
                        }
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "test_case_data": test_case_data,
                            "generated_code": code,
                        })
                else:
                    error_msg = "No Python code found in generated response"
                    st.error(error_msg)
                    with st.expander("📜 LLM output", expanded=True):
                        st.code(generated_code, language="text")
                    test_case_data = {
                        "name": last_user_request.lower().replace(" ", "_")[:50],
                        "last_user_request": last_user_request,
                        "previous_conversation": previous_conversation.strip(),
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "test_case_data": test_case_data,
                        "generated_code": generated_code,
                        "generated_code_is_raw": True,
                    })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                test_case_data = {
                    "name": prompt.lower().replace(" ", "_")[:50],
                    "last_user_request": prompt,
                    "previous_conversation": "",
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "test_case_data": test_case_data,
                    "generated_code": None,
                })


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "test_case_data" in msg:
            if msg.get("generated_code"):
                lang = "text" if msg.get("generated_code_is_raw") else "python"
                with st.expander("📜 Generated code", expanded=False):
                    st.code(msg["generated_code"], language=lang)
            with st.expander("📋 Test Case Metadata", expanded=False):
                st.code(json.dumps(msg["test_case_data"], indent=2), language="json")

if "pending_test_prompt" in st.session_state:
    prompt = st.session_state.pending_test_prompt
    del st.session_state.pending_test_prompt
    process_user_request(prompt)
    st.rerun()

if prompt := st.chat_input("Describe a category rule (e.g. 'Walmart should be groceries')..."):
    process_user_request(prompt)
    st.rerun()

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧪 Test Cases")
    for idx, test_case in enumerate(TEST_CASES):
        test_name = test_case.get("name", f"Test {idx + 1}")
        last_request = test_case.get("last_user_request", "")
        if st.button(
            f"🧪 {test_name}",
            key=f"test_case_{idx}",
            use_container_width=True,
            help=last_request[:60] + ("..." if len(last_request) > 60 else ""),
        ):
            st.session_state.messages = []
            previous_conversation = test_case.get("previous_conversation", "")
            if previous_conversation:
                lines = previous_conversation.strip().split("\n")
                current_role = None
                current_content = []

                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if line_stripped.startswith("User:"):
                        if current_role and current_content:
                            st.session_state.messages.append({
                                "role": current_role,
                                "content": "\n".join(current_content).strip(),
                            })
                        current_role = "user"
                        current_content = [line_stripped.replace("User:", "").strip()]
                    elif line_stripped.startswith("Penny:") or line_stripped.startswith("Assistant:"):
                        if current_role and current_content:
                            st.session_state.messages.append({
                                "role": current_role,
                                "content": "\n".join(current_content).strip(),
                            })
                        current_role = "assistant"
                        prefix = "Penny:" if line_stripped.startswith("Penny:") else "Assistant:"
                        current_content = [line_stripped.replace(prefix, "").strip()]
                    else:
                        if current_role:
                            current_content.append(line_stripped)

                if current_role and current_content:
                    st.session_state.messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip(),
                    })

            last_request = test_case.get("last_user_request", "")
            if last_request:
                st.session_state.pending_test_prompt = last_request
                st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.caption("Agent Categorize uses the optimizer to generate category rules. Execution uses the local stub (no DB).")
