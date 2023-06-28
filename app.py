from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from git import Repo
from urllib.parse import urlparse
from flask import Flask, request, render_template
from flask_cors import CORS
import shutil


import constants
import os
import sys

app = Flask(__name__)
CORS(app)


def set_environment_variables():
    os.environ["ACTIVELOOP_TOKEN"] = constants.ACTIVELOOP_APIKEY
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_APIKEY


def clone_repo(github_link):
    codebase_dir = "codebase"
    
    # Check if the directory exists, and if it does, remove it
    if os.path.exists(codebase_dir):
        print(f"'{codebase_dir}' exists. Attempting to remove...")
        shutil.rmtree(codebase_dir)
        if os.path.exists(codebase_dir):
            print(f"Failed to remove '{codebase_dir}'")
            return None
        else:
            print(f"Successfully removed '{codebase_dir}'")
    
    print(f"Cloning from '{github_link}' into '{codebase_dir}'...")
    try:
        Repo.clone_from(github_link, codebase_dir)
    except Exception as e:
        print(f"Failed to clone: {e}")
        return None

    print(f"Successfully cloned '{github_link}' into '{codebase_dir}'")
    return codebase_dir


def get_docs_from_directory(root_dir='./motion-canvas'):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(
                    dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(f"Error in file {file}:", e)
    return docs


def chunk_files(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def create_deep_lake(username, texts, repo):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path=f"hub://{username}/{repo}",
                  embedding_function=embeddings)
    db.add_documents(texts, embedding_data=texts)
    return db


def setup_retriever(db):
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    return retriever


def handle_questions(questions, qa):
    chat_history = []
    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
    return chat_history



def get_repo_name(github_link):
    url = urlparse(github_link)
    
    # Ex: '/username/repo_name.git'
    # Split the path into segments and get the last segment (removing '.git' from the end)
    repo_name = url.path.split('/')[-1].replace('.git', '')
    
    return repo_name


def main(github_link, question):
    set_environment_variables()

    repo_dir = clone_repo(github_link)

    docs = get_docs_from_directory(repo_dir)
    texts = chunk_files(docs)

    username = "brendanm12345"
    db = create_deep_lake(username, texts, get_repo_name(github_link))
    retriever = setup_retriever(db)

    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    # query = sys.argv[1]
    questions = [question]

    chat_history =  handle_questions(questions, qa)
    return chat_history[-1][1], chat_history # Returns the last answer and the whole chat history.

# Flask server
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    github_link = data['githubLink']
    question = data['question']

    answer, chat_history = main(github_link, question)
    return {'answer': answer, 'chat_history': chat_history}


if __name__ == "__main__":
    # run the flask app.
    app.run(port=5002)
