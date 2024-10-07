from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, memory, uploaded_file, question):
    # 定义模型
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, openai_api_base="https://api.aigc369.com/v1")
    # 1.对上传文档进行读取
    file_content = uploaded_file.read()  # 返回内容的二进制数据(bytes)
    temp_file_path = "temp.pdf"
    # 把前面读取到的二进制数据写入临时文件
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)  # 得到加载器实例
    docs = loader.load()  # 得到加载出的documents列表
    # 2.对文档进行分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)  # 得到分割好的一系列文档
    # 3.嵌入（文本->向量）
    embeddings_model = OpenAIEmbeddings()
    # 4. 储存（向量->向量数据库）
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    # 5.创建出带记忆的带记忆的检索增强对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
