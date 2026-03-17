import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

sf_emb_model = os.getenv("SF_EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")
sf_chat_model = os.getenv("SF_CHAT_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")
sf_base_url = os.getenv("SF_BASE_URL", "https://api.siliconflow.cn/v1")
sf_api_key = os.getenv("SF_API_KEY")

embedding = OpenAIEmbeddings(
    model=sf_emb_model,
    base_url=sf_base_url,
    api_key=sf_api_key,
    timeout=30
)

llm = ChatOpenAI(
    model=sf_chat_model,
    base_url=sf_base_url,
    api_key=sf_api_key,
    timeout=30
)

vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

prompt_template = """你是一个专业的问答助手。请根据以下参考文档回答用户的问题。
    如果参考文档中没有相关信息，请诚实地说不知道，不要编造答案。

    参考文档：
    {context}

    用户问题：{question}

    回答：
    """
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

def make_rag_db(file_path: str):
    documents = TextLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(documents)
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)
    print(f"成功将 {len(splits)} 个文本块存入向量数据库")

def generate_answer(query: str):
    print(f"\n问题: {query}")
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, question=query)
    print(f"\n提示词：{final_prompt}")
    messages = [HumanMessage(content=final_prompt)]
    response = llm.invoke(messages)
    print(f"\n回答: {response.content}")


if __name__ == "__main__":
    make_rag_db("three_body.txt")
    question = "宇宙文明的两条公理是什么？"
    generate_answer(question)

