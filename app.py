import json
import PyPDF2
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pinecone
import os
import io
from time import sleep

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')


import time
def readfile_and_createembeddings(event, context): 
    start_time = time.time()
    records = event['Records']
    for record in records:
        bucket = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
    print(bucket+" objectname: "+object_key)  
    pdf_text = read_pdf(bucket, object_key)

    chunks = convert_to_chunks(pdf_text)
    filename = object_key.split(".")[0].lower()
    filename = filename.replace('(', '').replace(')','').replace('-', '').replace('_', '')
    print(filename)
    vectore_store = insert_or_fetch_embedding(filename, chunks)
    q = 'How many states in the USA?'
    answer = fetch_answer(vectore_store, q)
    end_time = time.time()
    print(answer) 
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    return answer

       
def read_pdf(bucket, object_key):
  client = boto3.client("s3")
  response = client.get_object(Bucket = bucket, Key = object_key)
  pdf_content = response['Body'].read()
  pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
  pdf_text = ""
  for page_num in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[page_num]
      pdf_text += page.extract_text()
      
  pdf_text = pdf_text.encode('unicode_escape').decode('utf-8') 
  return pdf_text


def convert_to_chunks(data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                                       chunk_overlap = 0, 
                                                       length_function =len)
  chunks = text_splitter.split_text(text = data)
  return chunks

def  wait_on_index(index):
  ready = False
  while not ready: 
    try:
      desc = pinecone.describe_index(index)
      if desc[7]['ready']:
        return True
    except pinecone.core.client.exceptions.NotFoundException:
      pass
    sleep(5)   


def insert_or_fetch_embedding(index_name, chunks):
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
  
  if index_name in pinecone.list_indexes():
    vector_store = Pinecone.from_existing_index(index_name, embeddings)
    print('Fetched from embeddings ')
  else:
    print(f'Creating index {index_name} and embeddings ...', end='')
    pinecone.create_index(index_name, dimension=1536, metric = 'cosine')
    wait_on_index(index_name)
    vector_store = Pinecone.from_texts(chunks, embeddings,  index_name)
    print('Created new index')

  return vector_store
  
  
    
def fetch_answer(vector_store, q):
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=OPENAI_API_KEY)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
  chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  answer = chain.run(q)
  return(answer)


if __name__ == "__main__":
    readfile_and_createembeddings("","")

