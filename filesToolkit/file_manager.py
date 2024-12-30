import streamlit as st
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler


st.title("🗂️ File Management Agent")
st.write("Enter a command to interact with the file system using natural language.")

api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")
st.sidebar.write("You can get your API Key from the [here](https://groq.com/).")

if not api_key:
    st.warning("Please enter your API Key in the sidebar to continue.")
    st.stop()

@tool(return_direct=True)
def make_formatted_text(text: str) -> str:
    """
    Format text with proper indentation and line breaks.
    """
    formatted_text = text.encode().decode('unicode_escape')
    return formatted_text

format_tool = Tool(
    name="make_formatted_text",
    func=make_formatted_text,
    description="Formats text to ensure proper indentation and line breaks before writing to a file. Input: text"
)

working_dir = 'temp'
toolkit = FileManagementToolkit(
    root_dir=str(working_dir),
    selected_tools=["read_file", "write_file", "list_directory", "file_delete", "copy_file", "move_file", "file_search"],
)
tools = toolkit.get_tools()
tools.append(format_tool)

llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a file manager agent. You can:
        - Read, write, and list files.
        - Use a formatting tool to ensure text is properly formatted before writing.
        - Handle multi-step tasks autonomously.
        - Always format code before writing to a file.
        """),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I am a file manager agent. How can I assist you today?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Enter your command here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        
        try:
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = agent.run({"input": user_input}, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
