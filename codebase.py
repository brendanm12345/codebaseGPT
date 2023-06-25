from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from git import Repo
from flask import Flask, request
import constants
import os
import sys

app = Flask(__name__)

def set_environment_variables():
    os.environ["ACTIVELOOP_TOKEN"] = constants.ACTIVELOOP_APIKEY
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_APIKEY


def clone_repo(github_link):
    dir = Repo.clone_from(github_link, "./")
    print("CLONED: ", dir)
    return dir

def get_docs_from_directory(root_dir='./motion-canvas'):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(f"Error in file {file}:", e)
    return docs


def chunk_files(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def create_deep_lake(username, texts):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path=f"hub://{username}/motion-canvas",
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
        return result['answer']


def main(github_link, question):
    set_environment_variables()

    repo_dir = clone_repo(github_link)

    docs = get_docs_from_directory(repo_dir)
    texts = chunk_files(docs)

    username = "brendanm12345"
    db = create_deep_lake(username, texts)
    retriever = setup_retriever(db)

    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    # query = sys.argv[1]
    questions = [question]

    handle_questions(questions, qa)

# Flask server
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    github_link = data['githubLink']
    question = data['question']

    answer = main(github_link, question)

    return {'answer': answer}

if __name__ == "__main__":
    # run the flask app.
    app.run(port=5000)