import json 
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector
import time


def addapt_boilerplate():

    def addapt_numpy_float64(numpy_float64):
        return AsIs(numpy_float64)

    def addapt_numpy_int64(numpy_int64):
        return AsIs(numpy_int64)

    def addapt_numpy_float32(numpy_float32):
        return AsIs(numpy_float32)

    def addapt_numpy_int32(numpy_int32):
        return AsIs(numpy_int32)

    def addapt_numpy_array(numpy_array):
        return AsIs(tuple(numpy_array))

    register_adapter(np.float64, addapt_numpy_float64)
    register_adapter(np.int64, addapt_numpy_int64)
    register_adapter(np.float32, addapt_numpy_float32)
    register_adapter(np.int32, addapt_numpy_int32)
    register_adapter(np.ndarray, addapt_numpy_array)


def get_text_embedding(proposed_idea):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_idea = model.encode(proposed_idea,convert_to_numpy=True)
    return embedding_idea

def read_arxiv_json(conn):
    count = 1
    overall_count = 1
    db_list = list()
    with open('arxiv.json', 'r') as f:
        for i, line in enumerate(f):
            print("Count : ",count,"\n")
            if(count%100 == 0):
                print("100 files have been read from json. ")
                insertIntoTable(conn,db_list)
                print("Query is done! Resetting the list! ")
                db_list = list()
                count = 1
            if(overall_count > 10000):
                print("Overall count has reached 10000. So breaking. \n")
                break

            article = json.loads(line)
            dict_obj = {}
        
            title = article.get('title', '').lower()
            abstract = article.get('abstract', '').lower()
            
            paper_id = article.get('id','').lower()
            authors = article.get('authors','').lower()
            #print(paper_id,". ",title,"\n")
            #print("Authors : ",authors,"\n")
            #print("Abstract: \n",abstract,"\n")
            abstract = abstract.replace("\n"," ")
            embeddings = get_text_embedding(abstract)
            #print("Len of embeddings ",len(embeddings))
            #embedding_vector = np.array(embeddings.tolist())
            #print("Shape ",embedding_vector.shape)
            tup = (paper_id,title,authors,abstract,embeddings)
            db_list.append(tup)
            count+=1
            overall_count += 1


    print("Parsing json done!!!\n")    
    print(len(db_list))
    return db_list

def establish_connection():
    conn = psycopg2.connect(
        user ="",
        password = "",
        host = "",
        port = '',
        database = ''
    )
    conn.autocommit = True
    return conn

def createDatabase(conn):
    cursor = conn.cursor()
    database_create_query = """ create database paperDb"""
    cursor.execute(database_create_query)
    cursor.close()

def createTable(conn):
    cursor = conn.cursor()
    table_create_query = """ create table arxivVectorDB(paperId text PRIMARY KEY,title text, authors text, abstract text,embedding vector(384)) """
    cursor.execute(table_create_query)
    cursor.close()


def insertIntoTable(conn,list_to_insert):
    cursor = conn.cursor()
    table_insert_batch = execute_values(cursor, "INSERT INTO arxivVectorDB(paperId, title, authors, abstract, embedding) VALUES %s", list_to_insert)
    print("Executed the batch command!! \n")
    cursor.close()

def indexVectorsOnDB(conn):
    cursor = conn.cursor()
    #cursor.execute("")
    cursor.execute(f'CREATE INDEX ON arxivVectorDB USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);')


def searchVectorDatabase(conn,proposed_idea):
    embedding_idea = np.array(get_text_embedding(proposed_idea))
    # Register pgvector extension
    register_vector(conn)
    cursor = conn.cursor()
    # Get the top 3 most similar documents using the KNN <=> operator
    cursor.execute("SET ivfflat.probes = 100")
    cursor.execute("SELECT * FROM arxivVectorDB ORDER BY embedding <=> %s LIMIT 10", (embedding_idea,))
    top3_docs = cursor.fetchall()
    return top3_docs

def readJson():
    json_data = pd.read_json('arxiv.json',lines=True)
    print(json_data.head)
    
    
if __name__ == '__main__':
    conn = establish_connection()
    register_vector(conn)
    #createDatabase(conn)
    #createTable(conn)
    #db_list = read_arxiv_json(conn)
    #insertIntoTable(conn,db_list)
    #indexVectorsOnDB(conn)
    #idea = str(input("Enter your proposed idea."))
    
    idea = "We study the eigenvalue spectrum of a large real antisymmetric random matrix Jij. Using a fermionic approach and replica trick, we obtain a semicircular spectrum of eigenvalues when the mean value of each matrix element is zero, and in the case of a non-zero mean, we show that there is a set of critical finite mean values above which eigenvalues arise that are split off from the semicircular continuum of eigenvalues. The result converged with numerical simulations."
    time_begin = time.time()
    results = searchVectorDatabase(conn,idea)
    time_end = time.time()
    print("Time taken ",time_end-time_begin)
    print("Abstract given as input : ",idea,"\n")
    for result in results :
        print("Result:*********************************\n")
        print("Paper ID : ",result[0],"\n")
        print("Title : ",result[1],"\n")
        print("Authors : ",result[2],"\n")
        print("Abstract : ",result[3],"\n")
        print("**************************************\n")
    
    
