import streamlit as st
import requests
import uuid
import time

CONCIERGE_URL = "http://localhost:8000/chat"

# Keep a persistent session ID so conversations can be resumed
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your real estate assistant. How can I help you today?"}
    ]

session_id = st.session_state.session_id

# Sidebar with session info and a reset button
with st.sidebar:
    st.title("🏠 Concierge")
    st.caption("Federated Multi‑Agent System")
    st.divider()
    st.write(f"**Session ID**")
    st.code(session_id, language="text")
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your real estate assistant. How can I help you today?"}
        ]
        st.rerun()
    st.divider()
    st.write("*Backend agents:*")
    st.write("- Customer Onboarding (`:8001`)")
    st.write("- Deal Onboarding (`:8002`)")
    st.write("- Marketing Intelligence (`:8003`)")

st.title("Real Estate Agent")
st.caption("Powered by LangGraph + A2A Protocol")

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")
        try:
            resp = requests.post(
                CONCIERGE_URL,
                json={"message": prompt, "session_id": session_id},
                timeout=180
            )
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "")
            else:
                response_text = f"⚠️ Error: {resp.status_code} – {resp.json().get('detail', '')}"
        except requests.exceptions.ConnectionError:
            response_text = "❌ Could not connect to the Concierge. Is it running on port 8000?"
        except Exception as e:
            response_text = f"❌ An error occurred: {str(e)}"

        # Give a fake streaming effect for a nicer feel
        displayed = ""
        for char in response_text:
            displayed += char
            message_placeholder.markdown(displayed)
            time.sleep(0.005)
        message_placeholder.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})