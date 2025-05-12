import streamlit as st
import  time
import config
import json
import os
import sys
import uuid
import datetime
import glob
import time
import uuid
import csv
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos import ThroughputProperties
from dotenv import dotenv_values
from azure.cosmos import ThroughputProperties
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId
from bson import json_util, ObjectId
import json
from datetime import datetime

load_dotenv()





# specify the name of the .env file name 
env_name = "example.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)
# Azure Cosmos DB connection details
vcore = config['servervcore']

ta_endpoint = config["TEXT_ANALYTICS_ENDPOINT"]
ta_key = config["TEXT_ANALYTICS_KEY"]

container_name = "ChatMessages"


# Azure OpenAI connection details
openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_version = config['openai_version']
openai_embeddings_model = config['openai_embeddings_deployment']
openai_chat_model = config['AZURE_OPENAI_CHAT_MODEL']



dbsource = config['cosmosdbsourcedb'] 
colvector = config['cosmosdbsourcecol']
cachecol = config['cosmsodbcache']
cosmosdbcolcompletion = config['cosmosdbcolcompletion']
container_name = config['cosmosdbcolcompletion']
targetcolection = config['cosmosdargussource']

# Create the OpenAI client
openai_client = AzureOpenAI(
  api_key = openai_key,  
  api_version = openai_version,  
  azure_endpoint =openai_endpoint 
)



def create_index(collection,index_key ,index_name, dimension_size=1536, num_lists=100):
    # check if index exists
    existing_indexes = collection.index_information()
    if index_name not in existing_indexes:
        # Define the index key, specifying the field "embedding" with a cosine similarity search type
        index_key = [("embedding", "cosmosSearch")]# Define the options for the index
        index_options = {
            "kind": "vector-ivf",   # Index type: vector-ivf
            "numLists": num_lists,        # Number of lists (partitions) in the index
            "similarity": "COS",    # Similarity metric: Cosine similarity
            "dimensions": dimension_size      # Number of dimensions in the vectors
        }
        # Create the index with the specified name and options
        index = collection.create_index(index_key, name=index_name, cosmosSearchOptions=index_options)
        return index

def init_connection():
    """
    Initializes a connection to the MongoDB database using the configured server connection string.
    
    Returns:
        MongoClient: A MongoDB client instance connected to the server specified in the configuration.
        
    Note:
        This is a central connection function used throughout the application. Using a single
        connection initialization function helps maintain consistent connection management.
    """
    client = MongoClient(vcore)
    return client

def createvectordb(collection):
    """
    Creates and configures a vector database collection in Azure Cosmos DB with MongoDB API.
    
    This function initializes three essential indexes:
    1. A unique index on the "id" field for fast document lookups
    2. A vector index for embedding-based similarity search with cosine similarity
    3. A text index for traditional text search capabilities
    
    Args:
        collection (str): The name of the collection to create or configure
        
    Note:
        This function assumes the database specified in dbsource configuration exists.
        The vector index is created with 1536 dimensions matching OpenAI's embedding model.
    """
    
    client = init_connection()   
    
    mydbt = client.get_database(dbsource)
    container = mydbt[collection]
    container.create_index([("id", pymongo.ASCENDING)], unique=True)
    index_key = "embedding" 
    create_index(container,index_key, "vectorIndex1536", 1536)
    container.create_index([('text', pymongo.TEXT)], name='search_index', default_language='english')


     

def count_products(collection):
    """
    Counts the total number of documents in a MongoDB collection.
    
    This simple utility function provides a consistent way to count documents
    across different parts of the application.
    
    Args:
        collection: MongoDB collection object to count documents in
        
    Returns:
        int: The total number of documents in the collection
    """
    c = collection.count_documents({})
    return c


def loaddata(db, collection, name, filepath):
    """
    Loads JSON data from a file into an Azure Cosmos DB MongoDB collection.
    
    This function reads a JSON file, assigns unique IDs and additional metadata
    to each document, and inserts them into the specified collection.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the file (used for reference in searches)
        filepath (str): The path to the JSON file to load
        
    Raises:
        Exception: If there's an error during the file loading or document insertion process
        
    Note:
        Each JSON object in the file will be inserted as a separate document
        with added metadata including a unique ID, file reference, and timestamp.
    """
    client = init_connection()   
    mydbt = client.get_database(db)
  

    try:
        container = mydbt[collection]
        
        time = datetime.now() 
       
        
        filename = filepath
        with open(filename, encoding="utf8") as file:
            docu = json.load(file)
            for d in docu:
                doc = {} 
                doc["id"] = str(uuid.uuid4())
                doc["file"] = str(name)
                doc["text"] = json.dumps(d)
                doc["date"] = time
                container.insert_one(doc)
        

      

        total_count = count_products(container)
        print("Total documents in container:", total_count)     
       
    
    except : 
     raise  


def loadpptfile(db,collection, name,filepath) :
    """
    Loads and processes a PowerPoint file into an Azure Cosmos DB MongoDB collection.
    
    This function uses LangChain's UnstructuredPowerPointLoader to extract content from a PPT file,
    splits it into manageable chunks, and stores each chunk as a separate document
    in the specified collection with appropriate metadata.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the file (used for reference in searches)
        filepath (str): The path to the PowerPoint file to load
        
    Raises:
        Exception: If there's an error during PowerPoint processing or document insertion
        
    Note:
        The text is split into chunks of approximately 1000 characters with 150 character
        overlap to maintain context between chunks while keeping document size manageable
        for vector embedding generation.
    """
    loader = UnstructuredPowerPointLoader(filepath)
    data = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    
    client = init_connection()   
    mydbt = client.get_database(db)
    time = datetime.now() 

        

    try:
        container = mydbt[collection]
        
        time = datetime.now() 
       
        for d in docs : 
            docu= {} 
            docu["id"] = str(uuid.uuid4())
            docu["file"] = name
            docu["text"] = str(d)
            docu["date"] = time
           
            container.insert_one(docu)
        

      

        total_count = count_products(container)
        print("Total documents in container:", total_count)
    except : 
     raise     
# 



def loadxlsfile(db,collection, name,filepath):
    """
    Loads and processes an Excel file into an Azure Cosmos DB MongoDB collection.
    
    This function uses LangChain's UnstructuredExcelLoader to extract content from an Excel file,
    splits it into manageable chunks, and stores each chunk as a separate document
    in the specified collection with appropriate metadata.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the file (used for reference in searches)
        filepath (str): The path to the Excel file to load
        
    Raises:
        Exception: If there's an error during Excel file processing or document insertion
        
    Note:
        The text is split into chunks of approximately 1000 characters with 150 character
        overlap to maintain context between chunks while keeping document size manageable
        for vector embedding generation.
    """
    loader = UnstructuredExcelLoader(filepath, mode="elements")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    client = init_connection()   
    mydbt = client.get_database(db)
    time = datetime.now() 
  
        

    try:
        container = mydbt[collection]

        
        for d in docs : 
            docu= {} 
            docu["id"] = str(uuid.uuid4())
            docu["file"] = name
            docu["text"] = str(d)
            docu["date"] = time
           
            container.insert_one(docu)
            
        total_count = count_products(container)
        print("Total documents in container:", total_count)
   
    except:
        raise

def loadpdffile(db, collection, name, file):
    """
    Loads and processes a PDF file into an Azure Cosmos DB MongoDB collection.
    
    This function uses LangChain's PyPDFLoader to extract content from a PDF file,
    splits it into manageable chunks, and stores each chunk as a separate document
    in the specified collection with appropriate metadata.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the file (used for reference in searches)
        file (str): The path to the PDF file to load
        
    Raises:
        Exception: If there's an error during PDF processing or document insertion
        
    Note:
        The text is split into chunks of approximately 1000 characters with 150 character
        overlap to maintain context between chunks while keeping document size manageable
        for vector embedding generation.
    """
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
     
    client = init_connection()   
    mydbt = client.get_database(db)
    time = datetime.now() 
 
        
  

    try:
        container = mydbt[collection]
        
        
       
        for d in docs: 
            docu = {} 
            docu["id"] = str(uuid.uuid4())
            docu["file"] = name
            docu["text"] = str(d)
            docu["date"] = time
            container.insert_one(docu)
        

      

        total_count = count_products(container)
        print("Total documents in container:", total_count)
    except: 
     raise     
# 

def generate_embeddings(openai_client, text):
    """
    Generates vector embeddings for a given text using the Azure OpenAI embeddings API.
    
    This function sends the provided text to Azure OpenAI's embedding service to convert
    text into a high-dimensional vector representation. These vectors represent the 
    semantic meaning of the text and are used for similarity searches.
    
    Args:
        openai_client: An initialized Azure OpenAI client
        text (str): The text to generate embeddings for
        
    Returns:
        list: A list of floating-point numbers representing the text embedding vector
        
    Note:
        Uses the embedding model specified in the configuration. The default OpenAI
        embedding model typically returns vectors with 1536 dimensions.
    """
    
    response = openai_client.embeddings.create(
        input = text,
        model= openai_embeddings_model
    
    )
    embeddings = response.data[0].embedding
    return embeddings

def loadcsvfile(db,collection,name,file) :
    """
    Loads and processes a CSV file into an Azure Cosmos DB MongoDB collection.
    
    This function reads a CSV file line by line, converts each row to a JSON document,
    and stores it in the specified MongoDB collection with additional metadata.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the file (used for reference in searches)
        file (str): The path to the CSV file to load
        
    Raises:
        Exception: If there's an error during CSV processing or document insertion
        
    Note:
        Each row in the CSV becomes a separate document in the collection.
        The function adds metadata including a unique ID, file reference, and timestamp.
    """
    
    client = init_connection()   
    mydbt = client.get_database(db)
  
   
    try:
        container = mydbt[collection]
        time = datetime.now() 
   
        
 
    # Read CSV file and convert to JSON
        with open(file, mode='r', encoding='utf-8-sig') as file:
         csv_reader = csv.DictReader(file)
         for row in csv_reader:
                row["id"] = str(uuid.uuid4())
                row["file"] = name
                row["text"] = json.dumps(row)
                row["date"] = time
                         
                json_data = json.dumps(row)
                # Insert JSON data into Cosmos DB
                container.insert_one(json.loads(json_data))
    
        
        total_count = count_products(container)
        print("Total documents in container:", total_count)
       
    
    except : 
     raise

def loadwordfile(db, collection, name, file):
    """
    Loads and processes a Word document into an Azure Cosmos DB MongoDB collection.
    
    This function uses LangChain's Docx2txtLoader to extract content from a Word document,
    splits it into manageable chunks, and stores each chunk as a separate document
    in the specified collection with appropriate metadata.
    
    Args:
        db (str): The name of the database to use
        collection (str): The name of the collection to insert documents into
        name (str): A name identifier for the source of this document
        file (str): The path to the Word document to load
        
    Raises:
        Exception: If there's an error during Word document processing or document insertion
        
    Note:
        The text is split into chunks of approximately 1000 characters with 150 character
        overlap to maintain context between chunks while keeping document size manageable
        for vector embedding generation.
    """
    
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file)
    data = loader.load()    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
   
  
    client = init_connection()   
    mydbt = client[db]
    container = mydbt[collection]
    time = datetime.now() 
  
        

    try:
        
        for d in docs : 
            docu= {} 
            docu["id"] = str(uuid.uuid4())
            docu["file"] = name
            docu["text"] = str(d)
            docu["date"] = time
        
            container.insert_one(docu)
        

      

        total_count = count_products(container)
        print("Total documents in container:", total_count)
    except : 
     raise     


def serialize_datetime(obj):
    """
    Serializes datetime objects to ISO format for JSON serialization.
    
    This helper function enables datetime objects to be properly serialized
    to JSON when used with json.dumps().
    
    Args:
        obj: The object to serialize (expected to be a datetime object)
        
    Returns:
        str: ISO formatted datetime string
        
    Raises:
        TypeError: If the object is not a datetime instance
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

           
def add_doc(openai_client, collection, doc, name):
    """
    Adds a document to the vector collection with embedding generation.
    
    This function takes a document, enriches it with metadata, generates
    a vector embedding using OpenAI's embedding API, and stores it in the 
    vector collection. It uses upsert functionality to update the document
    if it already exists based on the document ID.
    
    Args:
        openai_client: An initialized Azure OpenAI client
        collection: MongoDB collection where the document will be stored
        doc (dict): The document to add or update, must contain an 'id' field
        name (str): A name identifier for the source of this document
        
    Note:
        This function implements a 1-second delay to avoid rate limiting when
        generating embeddings for multiple documents in sequence.
    """
   
    doc1 = {}
    doc1["id"] = doc["id"]
    doc1["source"]= name
    doc1["text"]= doc["text"]
    doc1["file"]= doc["file"]
    time.sleep(1)  # Rate limiting delay to avoid OpenAI API throttling
    doc1["embedding"] = generate_embeddings(openai_client, json.dumps(doc, default=serialize_datetime))
    collection.replace_one({"id": doc["id"]}, doc1, upsert=True)
       
   
               
        
def get_completion(openai_client, model, prompt: str):
    """
    Generates a completion using the Azure OpenAI chat completion API.
    
    This function sends a structured prompt to the OpenAI chat completions API
    and returns the full response including tokens used, model information, and
    the generated text.
    
    Args:
        openai_client: An initialized Azure OpenAI client
        model (str): The OpenAI model deployment name to use
        prompt (list): A list of message objects with 'role' and 'content' keys
        
    Returns:
        dict: The complete model response as a dictionary including all metadata
        
    Note:
        The temperature is set to 0.1 to produce more focused and deterministic responses.
        This is suitable for factual Q&A scenarios where creativity is less important
        than accuracy.
    """
   
    response = openai_client.chat.completions.create(
        model = model,
        messages =   prompt,
        temperature = 0.1
    )   
    return response.model_dump()

def chat_completion(user_message):
    # Dummy implementation of chat_completion
    # Replace this with the actual implementation
    response_payload = f"Response to: {user_message}"
    cached = False
    return response_payload, cached


def get_similar_docs(openai_client, db, query_text, limit, sim, typesearch):
    """
    Retrieves documents similar to the query text using either vector or full-text search.
    
    This function is the main search entry point that supports multiple search types:
    - Vector search: Uses embeddings for semantic similarity search
    - Full text search: Uses traditional text indexing for keyword matching
    - Hybrid search: (Placeholder for future implementation)
    
    Args:
        openai_client: Initialized Azure OpenAI client for generating embeddings
        db (str): Database name to query
        query_text (str): The text to search for
        limit (int): Maximum number of results to return
        sim (float): Similarity threshold (0.0-1.0) for filtering results
        typesearch (str): Type of search to perform ("vector", "full text", or "hybrid")
        
    Returns:
        list: A list of dictionaries containing the matching documents, with each
              dictionary containing at minimum a 'text' field with document content
              
    Note:
        Vector search performs semantic matching using cosine similarity on embeddings.
        Full text search uses MongoDB's text index capabilities for keyword matching.
        Results are formatted consistently regardless of search type to provide a
        uniform interface for downstream processing.
    """
   
    client = init_connection()   
    mydbt = client[db]
    cvector = mydbt[colvector]
    
    if  typesearch == "vector":
       
        
        query_vector = generate_embeddings(openai_client, query_text)
        pipeline = [
        {
        '$search': {
            "cosmosSearch": {
                "vector":   query_vector,
                "path": "embedding",
                "k": limit
              
            },
            "returnStoredSource": True }},
            { '$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } },
            { '$match': { "similarityScore": { '$gt': sim } } 
        } ]
       
    
        results = list(cvector.aggregate(pipeline))
        
       

      
         
    elif  typesearch == "full text":
   
        search_query = {"$text": {"$search": query_text}}
        
        # Set up projection to get the text search score
        projection = {"score": {"$meta": "textScore"}, "text": 1, "file": 1, "id": 1, "source": 1}
        
        # Execute the query and sort by text search score
        cursor = cvector.find(
            search_query,
            projection
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        # Process results
        results = list(cursor)
        print(f"Found {len(results)} documents matching the query")
        
        # Ensure results format matches the vector search format for consistent processing
        formatted_results = []
        for doc in results:
            # Format the result to match the structure expected by the rest of the code
            formatted_doc = {
                "document": {
                    "text": doc.get("text", ""),
                    "file": doc.get("file", ""),
                    "id": doc.get("id", ""),
                    "source": doc.get("source", "")
                },
                "similarityScore": doc.get("score", 0)
            }
            formatted_results.append(formatted_doc)
        
        results = formatted_results
        

        
            

         
            
    products = []      
   
    for result in results:
            product = {}  # Create a new dictionary for each result
            product['text'] = result['document']['text']  # Assign text from document
            products.append(product)  # Append to list
         
          
  
    
         
    return products


def ReadFeed(collection):
        
        client = init_connection()   
        mydbt = client[dbsource]
        mycolt = mydbt[collection]
        
       
        mycoltembed = mydbt["vector"] 
        name = collection
        
     
        # Define a point in time to start reading the feed from
            
        query = {}
        response = mycolt.find(query)
        
        
        for doc in response:
            doc = json.loads(json_util.dumps(doc))
            add_doc(openai_client, mycoltembed, doc,name)



def get_chat_history(  username,completions=1):
    
   
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[cachecol]

  
  
    results = container.find({}, {"prompt": 1, "completion": 1}).sort([("_id", -1)]).limit(completions)
    return results






def cachesearch(query_text, username, similarity_score=0.95, num_results=1):
    # Execute the query
   
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[cachecol]
    
  
    vectors = generate_embeddings(openai_client, query_text)
           

    pipeline = [
        {
        '$search': {
            "cosmosSearch": {
                "vector":   vectors,
                "path": "embedding",
                "k": num_results ,
                "filter": { "name":  username } 
                   
            },
            "returnStoredSource": True }},
            { '$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } },
            { '$match': { "similarityScore": { '$gt': similarity_score } } }
       
         ]
       
 
    results = list(container.aggregate(pipeline))
        
    formatted_results = list( results)
    return formatted_results

  


def cacheresponse(user_prompt, prompt_vectors, response, username):
    
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[cachecol]
    
   
    ta_client = TextAnalyticsClient(
        endpoint=ta_endpoint,
        credential=AzureKeyCredential(ta_key)
    )
    
    docu = [user_prompt]
    
    resp = ta_client.analyze_sentiment(docu)
    for doc in resp:
            Sentiment = doc.sentiment
            if Sentiment == "positive":
                Sentiment = "positive"
                sco = doc.confidence_scores.positive
            elif Sentiment == "negative":
                Sentiment = "negative"
                sco = doc.confidence_scores.negative
            elif Sentiment == "neutral":
                Sentiment = "neutral"
                sco = doc.confidence_scores.neutral
          
          
  
    
    
    # Create a dictionary representing the chat document
    chat_document = {
        'id':  str(uuid.uuid4()),  
        'prompt': user_prompt,
        "Sentiment": Sentiment,
        "Scores": sco,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
        'name': username,
        'embedding': prompt_vectors
    }
    # Insert the chat document into the Cosmos DB container
    container.insert_one(chat_document)
 

def clearall(): 
    
    client = init_connection()   
    client.drop_database(dbsource)

def createcachecollection():
    
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[cachecol]
    container.create_index([("name", pymongo.ASCENDING)], unique=False) 
    container.create_index([("id", pymongo.ASCENDING)], unique=True) 
   
    index_key = "embedding" 
    create_index(container,index_key, "vectorIndex1536", 1536)
    

   
def clearcache ():
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[cachecol]
    container.delete_many({})

  
def generatecompletionede(user_prompt, username,vector_search_results, chat_history):
    
    system_prompt = '''
    You are an intelligent assistant for yourdata, translate the answer in the same langage use for the ask. You are designed to provide answers to user questions about user's data.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses.use the name of the file where the information is stored to provide the answer.
        - start with the hello ''' + username + '''
        - Only answer questions related to the information provided below. 
        '''

    # Create a list of messages as a payload to send to the OpenAI Completions API

    # system prompt
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    #chat history
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})

    #vector search results
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': result['text']})

    
    # Create the completion
    response = get_completion(openai_client, openai_chat_model, messages)
  
    
    return response

def chat_completion(user_input,username, cachecoeficient, coefficient, maxresult, typesearch):

    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(openai_client, user_input)
    
    # Query the chat history cache first to see if this question has been asked before
    
    cache_results = cachesearch(user_input ,username,cachecoeficient, 1)
 
    if len(cache_results) > 0:
        print(f"cache results : {cache_results}")
        
        return cache_results[0]['document']['completion'], True
    else:
        # Perform vector search on the movie collection
       
        search_results = get_similar_docs(openai_client, dbsource, user_input, maxresult,coefficient, typesearch)
        
        
        # Chat history
        chat_history = get_chat_history(username,1)

        # Generate the completion
        completions_results = generatecompletionede(user_input,username ,search_results, chat_history)

        # Cache the response
        cacheresponse(user_input, user_embeddings, completions_results,username)

        
        
        return completions_results['choices'][0]['message']['content'], False

def loaddataargus( argusdb,arguscollection , argusurl,arguskey, targetcolection) :
    
    clientargus = CosmosClient(argusurl, {'masterKey': arguskey})
  
    
    mydbtsource = clientargus.get_database_client(argusdb)   
    
    client = init_connection()   
    mydbt = client[dbsource]
    container = mydbt[targetcolection]
    

    
 
    
    try:
        container = mydbt[targetcolection]
        query = "SELECT  c.id,c.extracted_data  FROM c"
        source = mydbtsource.get_container_client(arguscollection)
        result = source.query_items(
            query=query,
            enable_cross_partition_query=True)

        for item in result:
            item['text']= json.dumps(item)
            container.insert_one(item)

        
    except : 
        raise  


def extract_gpt_summary_output(data,data2):
    """
    Extrait la valeur de 'gpt_summary_output' d'un dictionnaire donné.

    Args:
    data (dict): Le dictionnaire contenant les données.

    Returns:
    str: La valeur de 'gpt_summary_output' si elle existe, sinon None.
    """
    # return data.get('gpt_summary_output')
    return data.get(data2)


def ReadFeedargus(collection):
    """
    Processes documents from Argus data source and adds them to the vector collection with embeddings.
    
    This function reads documents from the specified Argus source collection, extracts summary 
    and extraction output data, formats them as new documents, and adds them to the vector
    collection with embeddings for similarity search.
    
    Args:
        collection (str): Name of the collection to process
        
    Note:
        This function assumes that documents in the source collection have an "extracted_data" field
        containing "gpt_summary_output" and "gpt_extraction_output" fields from Argus processing.
        
        The function adds a 1-second delay between documents to avoid rate limiting from the 
        embedding API service.
    """
        
    client = init_connection()   
    mydbt = client[dbsource]
    mycolt = mydbt[targetcolection]
    
    mycoltembed = mydbt[colvector]  
    name = collection
    
    # Define a point in time to start reading the feed from
    myquery = {}
       
    response = mycolt.find(myquery)
    
    for doc in response:
        summary_output = extract_gpt_summary_output(doc["extracted_data"],'gpt_summary_output')
        details = extract_gpt_summary_output(doc["extracted_data"],'gpt_extraction_output')
        doc1 = {}
        doc1["file"] = doc["id"]
        doc1["id"] = str(uuid.uuid4())
        doc1["text"] = summary_output
     #   doc1["details"] = details
        add_doc(openai_client, mycoltembed, doc1, name)


# Fonction pour authentifier l'utilisateur
def authenticate(username):
    # Pour des raisons de démonstration, nous utilisons une vérification simple
    return username 




# Fonction pour charger des documents (fonction de remplacement)

# Application Streamlit
def main():
    st.title("Web application to show how to load your data and vectorize in cosmosdb mongo VCORE  Connection page")

    # Coefficient used to set the similarity threshold for database searches
    # Coefficient used to determine similarity threshold for cache search
   
    cachecoeficient = 0.95
    maxresult = 5
    global chat_history
    chat_history = []
  
    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
       
        username = st.session_state.username
        display = "Welcome to the applicaton: " + username 
        st.success(display)

        # Onglets
        tab1, tab2, tab3, tab4 ,tab5  = st.tabs(["Configuration", "Loading file", "Chat with your data","load argus data " ,"Documents load "])

        with tab1:
            st.header("Configuration")
            
            if st.button("create the Vector collection and cache collection "):
                st.write("start the operation")
                createvectordb(colvector)
                createcachecollection()
                st.write("the database and collection vector and cache are created ")
                
            
            if st.button("clear the cache collection for show cache"):
                st.write("start clear the Cache")
                clearcache()
                st.write("Cache cleared.")
                
                

            if st.button("delete all the database and collection to reinit"):
                st.write("delete all collection")
                clearall()
                st.write("all collection delete ")
            
            coefficient = st.slider("similarity coef for the search in database by default 78 % ", 0.0, 1.0, 0.78)
            cachecoeficient = st.slider("similarity coef for the cache search in database by default 99 %", 0.0, 1.0, 0.99)
            maxresult = st.slider("Numbers of max result in the database retrieve by similarity", 1, 10, 5)
            
  

        with tab2:
            st.header("Load document ")
        
            uploaded_file = st.file_uploader("Choose your file to upload", type=["pdf", "docx","csv", "ppt","xls","xlsx" ,"pptx", "json"])
            if uploaded_file is not None:
                st.write("File selected: ", uploaded_file.name)
        
            # Enregistrer temporairement le fichier téléchargé pour obtenir le chemin absolu
                with open(uploaded_file.name, "wb") as f:
                     f.write(uploaded_file.getbuffer())

            # Obtenir le chemin absolu du fichier
                absolute_file_path = os.path.abspath(uploaded_file.name)
                st.write(f"the file path is  : {absolute_file_path}")
                
                
                if st.button("load data "):
                    st.write("start the operation")
                
                    if ".doc" in uploaded_file.name:
                        st.write("this is a file type word "+ uploaded_file.name )
                        loadwordfile(dbsource,'word',uploaded_file.name,absolute_file_path )
                        ReadFeed('word')
                        st.write("file load" +uploaded_file.name )
                        
                        
                    elif ".xls" in uploaded_file.name:
                        st.write("this is a file type xls "+ uploaded_file.name )
                        loadxlsfile(dbsource,'xls',uploaded_file.name,absolute_file_path )
                        ReadFeed('xls')
                        st.write("file load" +uploaded_file.name )    
                    
                    elif ".ppt" in uploaded_file.name:
                        st.write("this is a file type ppt "+ uploaded_file.name )
                        loadpptfile(dbsource,'ppt',uploaded_file.name,absolute_file_path )
                        ReadFeed('ppt')
                        st.write("file load" +uploaded_file.name )    
                    
                         
                    elif ".pdf" in uploaded_file.name:
                        st.write("this is a file type pdf "+ uploaded_file.name )
                        loadpdffile(dbsource,'pdf',uploaded_file.name,absolute_file_path )
                        ReadFeed('pdf')
                        st.write("file load" +uploaded_file.name )
                        
                    elif ".json" in uploaded_file.name:
                        st.write("this is a file type json "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.json', '')
                        loaddata(dbsource,name,uploaded_file.name,absolute_file_path )
                        ReadFeed(name)
                        st.write("file load" +uploaded_file.name )
                        
                    elif ".csv" in uploaded_file.name:
                        st.write("this is a file type csv "+ uploaded_file.name )
                        loadcsvfile(dbsource,'csv',uploaded_file.name,absolute_file_path )
                        ReadFeed('csv')
                        st.write("file load" +uploaded_file.name )

                    os.remove(absolute_file_path)
                    st.write(f"the temp file  {absolute_file_path} was delete .")

         
        with tab3:
            st.header("Chat")
            models = [
                "vector",
                "full text"
                ]
            if st.button("clear the cache"):
                st.write("start clear the Cache")
                clearcache()
                st.write("Cache cleared.")
            typesearch = st.selectbox(
                'type search',
                    (models))
                
            st.write("Chatbot goes here")
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
                ]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
           
                          
            if prompt := st.chat_input(placeholder="enter your question here ? "):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    question = prompt
                    start_time = time.time()
                    response_payload, cached = chat_completion(question,username,cachecoeficient,coefficient, maxresult, typesearch)
                    end_time = time.time()
                    elapsed_time = round((end_time - start_time) * 1000, 2)
                    response = response_payload

                    details = f"\n (Time: {elapsed_time}ms)"
                    if cached:
                        details += " (Cached)"
                        chat_history.append([question, response + "for "+ username + details])
                    else:
                        chat_history.append([question, response + details])
        
                    st.session_state.messages.append({"role": "assistant", "content":chat_history})
                    st.write(chat_history)
            
        with tab4:
            st.write("load the data and connect the data from argus accelerator")
            st.write("result Getting data from ARGUS ACCELERATOR : https://github.com/Azure-Samples/ARGUS")
            argusdb = st.text_input("your Argus cosmosdb database", "doc-extracts")
            argusurl = st.text_input("your Argus csomsodb URI", "http... ")
            arguskey = st.text_input("your Argus csomsodb key", "xxxx... ")
            arguscollection = st.text_input("your Argus cosmosdb collection source", "documents")
            
            if st.button("load the data "):
                if arguscollection == None or arguskey == None or argusurl == None : 
                    st.write ( "parameters non correct , please entry your key , url and colleciton")
                else:
                     # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Initialize connection (10%)
                    status_text.text("Connecting to Argus database...")
                    progress_bar.progress(10)
                    time.sleep(1)  # Small delay for UI feedback
                    
                    # Step 2: Loading data (50%)
                    status_text.text("Loading data from Argus collection...")
                    progress_bar.progress(30)
                    
                    try:
                        total = loaddataargus(argusdb, arguscollection, argusurl, arguskey, targetcolection)
                        
                        # Step 3: Processing data (70%)
                        status_text.text("Processing documents and generating embeddings...")
                        progress_bar.progress(70)
                        
                        # Step 4: Reading feed (90%)
                        status_text.text("Generating vector embeddings for search...")
                        ReadFeedargus(targetcolection)
                        progress_bar.progress(90)
                        
                        # Step 5: Complete (100%)
                        progress_bar.progress(100)
                        status_text.text("Data loading complete!")
                        
                        # Show completion message
                        st.success(f"Successfully loaded data from Argus. Data is now available in the vector store.")
                        st.write("You can now chat with your data in the chat tab")
                        
                    except Exception as e:
                        # Handle error case
                        progress_bar.empty()
                      
        with tab5:
            
            if st.button("show file loads  "):
                
                  
                st.write(f"LIST OF FILE LOAD ")  
                
                client = init_connection()   
                mydbt = client[dbsource]
                container = mydbt[colvector]
                
                
             
                result = container.distinct("file")
             
                
               

                df = pd.DataFrame(result, columns=['file'])

                st.dataframe(df)
                
            
            st.write("made by emmanuel deletang in case of need contact him at edeletang@microsoft.com")
                

 


    else:
        # Formulaire de connexion
      
        username_input = st.text_input("User name ")
        email_input = st.text_input("Email")
        country_input = st.text_input("country")

        if st.button("Connexion"):
            if authenticate(username_input):
                st.session_state.logged_in = True
                st.session_state.username = username_input
                client = init_connection()   
                mydbt = client['user']
                container = mydbt['user']
                
              
                docu= {} 
                docu["id"] = str(uuid.uuid4())
                docu["name"] = username_input
                docu["email"] = email_input
                docu["country"] = country_input
                container.replace_one({'name': username_input}, docu, upsert=True)
               
                
                
                
                
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")


if __name__ == "__main__":
    main()
