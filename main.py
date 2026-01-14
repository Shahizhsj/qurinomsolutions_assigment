import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain.agents import create_agent

# -------------------------
# ENV VARIABLES
# -------------------------

os.environ["GOOGLE_API_KEY"] = "AIzaSyCxSKTQl6wGViRlIidUHm7XBjS_hnZVe4Y"
os.environ["SERPER_API_KEY"] = "4675fe6f6a602e139fc7f75a8e57ab354695e284"

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Web Search Agent", page_icon="🌐")
st.title("🌐 AI Web Search Agent")
st.write("Ask a question. The agent will search the web and answer using real data.")

query = st.text_input("Enter your question:")

# -------------------------
# LANGCHAIN SETUP
# -------------------------
search = GoogleSerperAPIWrapper()

@tool
def web_search(query: str) -> str:
    """Search the web using Google Serper and return results."""
    return search.run(query)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key="AIzaSyBcFSfAS0uIv0eckSHzPs5aXILAud47mfQ"
)

agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=(
        "You are an AI agent that answers questions using real-time web data. "
        "Always search the web before answering."
    )
)

# -------------------------
# RUN AGENT
# -------------------------
if st.button("Search & Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the web and generating answer..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )

            # ✅ CLEAN FINAL ANSWER
            final_answer = response["messages"][-1].content[0]['text']
            st.success("Answer:")
            st.write(final_answer)
