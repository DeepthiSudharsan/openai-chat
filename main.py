import os

from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader("./docs", recursive=True).load_data()
print(f"number of documents : {len(documents)}")
print('Document ID:', documents[0].doc_id, 'Document Hash:', documents[0].doc_hash, 'Extra Info:',
      documents[0].extra_info)

###########################
 #  CREATE THE INDEX
 #
 #from llama_index.indices.vector_store.vector_indices import GPTSimpleVectorIndex
from llama_index.indices.vector_store import GPTVectorStoreIndex

index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./index2")


############################
#  LOAD THE INDEX
from llama_index.storage.storage_context import StorageContext
from llama_index import load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./index2")
index = load_index_from_storage(storage_context=storage_context)
# response = index.as_query_engine().query("How can we enable real time booking controls?")
response = index.as_query_engine().query("How many years has Erdogan ruled Turkey?")
print(str(response))


