from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st
from langchain.document_loaders import SitemapLoader, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.storage import LocalFileStore
import os
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from urllib.parse import urlparse
from langchain.memory import ConversationBufferMemory

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False



class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question, 
                        "context": doc.page_content,
                    },
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


# if "win32" in sys.platform:
#     # Windows specific event-loop policy & cmd
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
#     cmds = [['C:/Windows/system32/HOSTNAME.EXE']]
# else:
#     # Unix default event-loop policy & cmds
#     cmds = [
#         ['du', '-sh', '/Users/fredrik/Desktop'],
#         ['du', '-sh', '/Users/fredrik'],
#         ['du', '-sh', '/Users/fredrik/Pictures']
#     ]
# ------------------------------------------------------
# Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize a UserAgent object
# ua = UserAgent()


# html2text_transformer = Html2TextTransformer()
# ----------------------------------------
@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    file_folder = f"./.cache/embeddings/xml"
    
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    cache_dir = LocalFileStore(file_folder)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)

    return vector_store.as_retriever()







st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
with st.sidebar:
    if api_key and check_api_key(api_key):
        st.success("API Key is valid!")
        url = "https://developers.cloudflare.com/sitemap-0.xml"
    else:
        st.error("Please enter a valid OpenAI API Key.")
    st.link_button("Github", "https://github.com/JEN9999/Assignment2/blob/main/pages/SiteGPT.py")



if  check_api_key(api_key) and ".xml" in url:
    retriever = load_website(url)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input()
    if message:
        send_message(message, "human")
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),

            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)
            result_message = result.content.replace("$", "\$")

else:
    st.session_state["messages"] = []