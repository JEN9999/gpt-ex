import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime

# st.title("Hello!")

# st.subheader("Streamlit")

# st.markdown("""
#     #### OHOHOHOHOH
# """)

# st.write("Hello")

# a = [1,2,3,4]

# d = {"x" : 1}

# p = PromptTemplate.from_template("xxxx")



# a
# d
# p

# PromptTemplate

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

model = st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
    ),
)

if model == "GPT-3":
    st.write("cheap")
else:
    st.write("not cheap")
    name = st.text_input("What is your name?")
    st.write(name)

    value = st.slider(
        "temperature",
        min_value=0.1,
        max_value=1.0,
    )

    st.write(value)



st.title("title")


with st.sidebar:
    st.title("sidebar title")
   
tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

with tab_one:
    st.write('a')

with tab_two:
    st.write('b')

with tab_three:
    st.write('c')
-------------------------------------------
# 7.5

# with st.chat_message("human"):
#     st.write("Hellloo")

# with st.chat_message("ai"):
#     st.write("how are u")

# with st.status("Embedding file...", expanded= True) as status:
#     time.sleep(1)
#     st.write("Getting the file")
#     time.sleep(1)
#     st.write("Embedding the file")
#     time.sleep(1)
#     st.write("Caching the file")
#     status.update(label="Error", state="error")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )



message = st.chat_input("Send a message to the AI")

if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)