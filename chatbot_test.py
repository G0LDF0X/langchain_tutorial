# main.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from collections import defaultdict

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4-0125-preview"

#main.py
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
노선별 지하철역 정보는 content 데이터를 참고해.
각 호선은 외부코드의 오름차순 순서, 혹은 내림차순 순서로 운행하고 있어. 예를 들면 2호선을 타고 홍대 입구에 가려면 당산과 합정을 거쳐 홍대입구를 가는 식이야.
최소 환승을 할 때, 최단거리도 고려해줘.
사용자가 시작역과 도착역을 주면 두 역 사이의 지하철을 대조해서 최소거리 최소환승을 할 수 있는 방법을 알려줘.
시작역에서 다니는 열차가 도착역에 다니지 않으면 반드시 중간에 환승해야해.
content
{}
"""
@st.cache_data
def load_data():
    data = pd.read_csv("서울교통공사 노선별 지하철역 정보.csv", encoding = "cp949")
    subway = data["호선"].unique()
    subway_dict = defaultdict(list)
    for rail in subway:
        subway_dict[rail].append(data[data["호선"] == rail].sort_values(by="외부코드"))

    return subway_dict

data = load_data()

content=load_data()

st.header("백엔드 스쿨/파이썬 2회차(9기)")
st.info("지하철 환승에 대한 내용을 알아볼 수 있는 Q&A 로봇입니다.")
st.error("지하철 환승에 대한 내용이 적용되어 있습니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 백엔드 스쿨 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))