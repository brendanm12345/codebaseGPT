import os
import sys

import constants
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]

loader = DirectoryLoader('./noramp-prism', glob="*.tsx", show_progress=True)
# loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])


# print(index.query(query))
print(index.query(query, llm=ChatOpenAI))

# llm = OpenAI(temperature=0.9)

# print(llm.predict("What would be a good company name for a company that makes colorful socks?"))
# # >> Feetful of Fun


