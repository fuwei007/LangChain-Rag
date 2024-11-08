# RAG base on langchain: How I Built a RAG System based on My Wife's Recipes!
import os
os.environ['OPENAI_API_KEY'] = ''


import shutil
import gradio as gr
from langchain import hub
from langchain.agents import  AgentExecutor, create_json_chat_agent
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path):
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path,autodetect_encoding=True)
        return loader.load()
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    else:
        loader = DirectoryLoader(file_path)
        return loader.load()


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path

def handle_file(file_obj):
    rag_file_dir = './RAG_data/'

    dir_check(rag_file_dir)
    print(f'RAG files dir: {rag_file_dir}')
    print(f'file uploaded path: {file_obj.name}')

    # copy the upload file into given dir
    shutil.copy(file_obj.name, rag_file_dir)

    # upload give file
    docs = load_file(os.path.join(rag_file_dir, file_obj.name.split('\\')[-1]))
    print("Loaded documents:", len(docs))

    # spilt content into different parts
    splits = text_splitter.split_documents(docs)

    # embedding and add the spiltted parts into the vector database
    retriever.add_documents(splits)

    # update the rag file list
    filename = file_obj.name.split('\\')[-1]
    files.append(filename)
    return '\n'.join(files)

def chat_response(message, chat_history):
    # get the response give the user's input
    response = agent_executor.invoke({"input": [HumanMessage(content=message)]}, config=config)
    response_str = response['output']
    print(f"response :{response}")
    chat_history.append((message, response_str))
    return "", chat_history

if __name__=='__main__':

    # init llm and text embedding model
    llm = OpenAI()
    embed_model = OpenAIEmbeddings() # model used to convert text to text embeddings

    # Splitting text by recursively look at characters.
    # link: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Chroma: AI-native open-source vector database
    # link: https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
    vectorstore = Chroma(collection_name="langchain_RAG", embedding_function=embed_model)

    # create retriver
    retriever = vectorstore.as_retriever()

    # construct the agent
    tool = create_retriever_tool(
        retriever,
        "retriever",
        "answer user's question on recipes",
    )
    tools = [tool]

    # init the agent prompt
    # link: https://smith.langchain.com/hub/hwchase17/react-chat-json?tab=0
    prompt = hub.pull("hwchase17/react-chat-json")

    # Create an agent that uses JSON to format its logic, build for Chat Models
    # url: https://api.python.langchain.com/en/latest/agents/langchain.agents.json_chat.base.create_json_chat_agent.html
    agent = create_json_chat_agent(llm=llm, tools=tools, prompt=prompt)

    # url: https://python.langchain.com/v0.2/docs/how_to/agent_executor/
    # AgentExecutor equip the agent with the ability to call tools
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    config = {"configurable": {"thread_id": "abc123"}}

    with gr.Blocks() as demo:
        with gr.Column():
            chatbot = gr.Chatbot()

            with gr.Row():
                file_input = gr.components.File(label="Upload a file")
                # upload_btn = gr.Button(value="Upload File")
                file_lst = gr.Textbox(label="Uploaded Files")
                files = []
                file_input.upload(fn=handle_file, inputs=file_input, outputs=file_lst)
                # upload_btn.click(fn=handle_file, inputs=file_input,outputs=file_lst)

            question_input = gr.Textbox(label="Ask a question")
            question_input.submit(fn=chat_response, inputs=[question_input, chatbot], outputs=[question_input, chatbot])

    demo.launch()
