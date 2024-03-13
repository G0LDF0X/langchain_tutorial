import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# langchain 모델 기본 사용하기
llm = ChatOpenAI(openai_api_key=API_KEY)
# output = llm.invoke("2024년 청년 지원 정책에 대해 알려줘.")
# print(output)

# # Template 기반 사용법
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야."),
#     ("user", "{input}")
# ])
# chain = prompt | llm
# output = chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘."})
# print(output)

# # 내용 파싱하기
# # 파싱(Parsing) : 일련의 문자열을 의미있는 token(어휘 분석의 단위)으로 분해하고
# # 그것들로 이루어진 Parse tree를 만드는 과정
# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser
# output = chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘."})
# print(output)

# # 검색 기능 적용하기(임베딩 포함)
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")
docs = loader.load()
# print(docs)

embeddings = OpenAIEmbeddings(openai_api_type=API_KEY)
text_spliteter = RecursiveCharacterTextSplitter()
documents = text_spliteter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# 바로 Docs 내용을 반영도 가능

output = document_chain.invoke({
    "input": "국민취업제도가 뭐야",
    "context": [Document(page_content="""국민취업제도란?
취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
[출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24""")]
})    
print(output)         

retriever = vector.as_retriever()
retrieer_chain = create_retrieval_chain(retriever, document_chain)

response = retrieer_chain.invoke({"input": "상담센터 전화번호가 뭐야"})
print(response["answer"])
print(response)