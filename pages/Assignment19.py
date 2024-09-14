import streamlit as st
from openai import OpenAI
import yfinance
import json
import re
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

client = OpenAI()

functions = [
    {
        "name": "get_ticker",
        "description": "회사 이름으로 티커(symbol)를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "회사의 이름"
                }
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "get_income_statement",
        "description": "주어진 티커의 손익계산서를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "회사의 티커(symbol)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_balance_sheet",
        "description": "주어진 티커의 대차대조표를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "회사의 티커(symbol)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_daily_stock_performance",
        "description": "주어진 티커의 최근 3개월간의 주식 성과를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "회사의 티커(symbol)"
                }
            },
            "required": ["ticker"]
        }
    }
]

def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")

def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())

def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())

def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with args {function.arguments}")
        outputs.append(
            {
                "tool_call_id": action_id,
                "output": functions_map[function.name](json.loads(function.arguments)),
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run.id, thread.id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})





def paint_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(format_text(message))
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        paint_message(
            message["message"].replace("$", "\$"), message["role"], save=False
        )



   

functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}





st.set_page_config(page_title="Financial Research AI Agent", page_icon=":chart_with_upwards_trend:")
st.title("AssistantGPT")


api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")
if api_key:
    client.api_key = api_key 

st.link_button("Github", "https://github.com/JEN9999/Assignment2/blob/main/pages/Assignment19.py")

st.title("금융 리서치 AI 에이전트")

if 'assistant' not in st.session_state:
    assistant = client.beta.assistants.create(
        name="금융 리서치 어시스턴트",
        instructions="당신은 금융 리서치 전문가 AI입니다. 사용자에게 주식 성과와 금융 분석에 대해 상세하게 답변해주세요.",
        model="gpt-4-1106-preview",
        tools=[] 
    )
    st.session_state['assistant'] = assistant

if 'thread' not in st.session_state:
    thread = client.beta.threads.create() 
    st.session_state['thread'] = thread


def extract_text_from_content(content):
    if hasattr(content, "text") and hasattr(content.text, "value"):
        return content.text.value
    elif isinstance(content, str):
        return content
    return str(content)

def format_text(text):
    return text.replace("\n", "  \n")


def create_run_and_poll():

    client.beta.threads.messages.create(
        thread_id=st.session_state['thread'].id,
        role="user",
        content=query
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=st.session_state['thread'].id,
        assistant_id=st.session_state['assistant'].id,
        instructions="금융 관련 질문에 대해 자세하게 설명하세요."
    )

    return run


query = st.chat_input("AssistantGPT에게 주식 정보를 물어보세요!")

paint_history()

if query: 
    paint_message(query, "user")

    client.beta.threads.messages.create(
        thread_id=st.session_state['thread'].id,
        role="user",
        content=query
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=st.session_state['thread'].id,
        assistant_id=st.session_state['assistant'].id,
        instructions="금융 관련 질문에 대해 자세하게 설명하세요."
    )
    
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=st.session_state['thread'].id)
        for message in messages:
            role = "사용자" if message.role == "user" else "어시스턴트"
            content_value = extract_text_from_content(message.content)

            if role == "어시스턴트":
                paint_message(content_value, "assistant")
    else:
        st.write(f"Run 상태: {run.status}")