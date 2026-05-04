"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Module: 7
Assignment: Build Your Own Adventure. I chose to do an Ai tutor, Embedding Retrieval + LLM

I reused a lot of the code from the week 4 RAGproject from the Ai foundations class.

"""
import os
import logging
from typing import List, Dict, Optional, Callable
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import pymupdf4llm
import numpy as np
from huggingface_hub import InferenceClient




# Setting up loging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",

)

# Getting logger
logger = logging.getLogger(__name__)

# Creating a system prompt that will define my tutors persona
tutor_system_prompt: str = """

You are an experienced, patient AI tutor. Your job is to help a student build
deep, lasting understanding of the topics in the provided CONTEXT not just
give them a quick answer.

Every answer you give must follow this five part structure:

- Definition: Restate the concept in one or two plain sentences.
   Avoid jargon overly technical jargon. If a technical term is unavoidable, define it the first time
   it appears.
- Intuition: Explain the underlying idea with an analogy, a mental model,
   or a comparison to something the student likely already knows.
- Concrete Example: Walk through a specific example, drawn from the CONTEXT
   whenever possible. Show the concept in action, not in an  abstract way.
- Why It Matters:  Connect the concept to a bigger picture: when does it
   come up, what problem does it solve, what does it enable next?
- Source: Cite the source or sources you you used in the exact format the CONTEXT
   provides (e.g., [Source: filename.pdf, Page: 12]).

For straightforward factual questions, you may keep some sections to a single
sentence, but Definition and Source are always required.

GROUNDING RULES (non-negotiable):

- Answer ONLY using information found in the CONTEXT below. Do not use outside
  knowledge, even if you are confident it is correct.
- If the CONTEXT does not contain enough information to answer, say exactly:
  "I don't have that in my study materials." Then suggest one related
  question the student could ask that the CONTEXT could answer.
- Never invent sources, page numbers, or facts. If you cannot cite a real
  source from the CONTEXT, do not cite at all.

TONE:

- Encouraging and direct. Skip filler like "Great question!" or
  "Let me think about that...".
- Speak to the student as a peer who is learning, not a beginner who needs
  hand holding.
- Choose depth over length. A focused answer beats a padded one with a wall of text.

""".strip()

# Creating few shot examples that mimic what the real outputs will look like
# So the model will know how i want the output formatted
few_shot_examples: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "CONTEXT:\n"
            "[Source: CSC6314Week2Mulhall.pdf, Page: 18]\n"
            "Overfitting occurs when a model learns training data too well, "
            "including noise, leading to poor generalization. Regularization "
            "techniques like dropout and L2 weight decay mitigate this.\n\n"
            "QUESTION: What is overfitting?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "**Definition:** Overfitting is when a model memorizes training data "
            "instead of learning the underlying patterns.\n\n"
            "**Intuition:** Like a student who memorizes last year's exam "
            "answers, they ace that test but fail this year's.\n\n"
            "**Concrete Example:** A network hits 99% training accuracy but only "
            "70% on validation. That gap is what points to overfitting.\n\n"
            "**Why It Matters:** Generalization is the whole point of training. "
            "Dropout and L2 exist to prevent this.\n\n"
            "**Source:** [Source: CSC6314Week2Mulhall.pdf, Page: 18]"
        ),
    },

    {
        "role": "user",
        "content": (
            "CONTEXT:\n"
            "[Source: CSC6314Week3Mulhall.pdf, Page: 7]\n"
            "CNNs use filters that slide across input images to detect features "
            "like edges and textures. Pooling layers downsample feature maps.\n\n"
            "QUESTION: What learning rate schedule does AdamW use?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I don't have that in my study materials.\n\n"
            "The CONTEXT covers CNN filters and pooling, not optimizer schedules. "
            "A related question I could answer: \"How do CNN filters detect "
            "features in an image?\""
        ),
    },
]


# This class loads documents to be chunked
class DocumentLoader:
    """
    This class loads documents, parses the content and stores them as chunks in a list of dictionaries.
    """
 
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Docstring for __init__
 
        :param chunk_size: Number of characters per chunk
        :type chunk_size: int
        :param chunk_overlap: Number of overlapping characters between chunks
        :type chunk_overlap: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
    # Helper method to create a dictionary with the extracted file content
    def collect_file_content(self, source: str, content: str, page: Optional[str] = None, row: Optional[str] = None) -> Dict[str, str]:
        # Creating a dictionary that holds the file name and content as keys and values
        file_data: Dict = {
            "content": content,
            "source": source,
            "page": page,
            "row": row
        }
        return file_data
 
    # This method returns content from a text file or markdown file
    def get_text_content(self, file_path: str) -> str:
        """
        Reads a .txt or .md file and returns its content as a string.
 
        :param file_path: Absolute path to the file
        :type file_path: str
        :return: File content
        :rtype: str
        """
        try:
            with open(file_path, "r", encoding="utf-8") as text_file:
                text_file_content = text_file.read()
                return text_file_content
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we hvave a problem {e}")
 
    # This method returns content from a pdf file
    def get_pdf_content(self, file_path: str, documents: List, collect_content_func: Callable[[str, str], Dict]) -> None:
        """
        This method gets the contents of a pdf file along with the page num and file name
 
        :param file_path: The location of the file
        :type file_path: str
        :param documents: The list that holds dictionaries with all the content extracted from files
        :type documents: List
        :param collect_content_func: the collect_file_content function that takes the filename and file content as params
        :type collect_content_func: Callable[[str, str], Dict]
        """
        try:
            # Getting PDF metadata that includes content, page number and file path. page_chunks=True makes the metadata available
            pages: List[Dict[str, str]] = pymupdf4llm.to_markdown(
                file_path, page_chunks=True)
            # Looping through the list of dictionaries to extract the needed info from the PDF metadata
            for page_dict in pages:
                # Extracting the pdf page content
                page_content: str = page_dict["text"]
                # Getting metadata for the specific page so we can get the page number later
                metadata: str = page_dict["metadata"]
                # Extracting number for the specific page
                page_num: int = metadata["page"]
 
                # creating a dictionary with the needed page details
                pdf_data_dict = collect_content_func(
                    file_path, page_content, page_num)
                # adding page data  to documents list
                documents.append(pdf_data_dict)
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we have a problem {e}")
 
    # This method returns the content from a CSV
    def get_csv_content(self, file_path: str, documents: List, collect_content_func: Callable[[str, str], Dict]) -> None:
        """
        This method gets the contents of a csv file along with the row num and file name
 
        :param file_path: The location of the file
        :type file_path: str
        :param documents: The list that holds dictionaries with all the content extracted from files
        :type documents: List
        :param collect_content_func: the collect_file_content function that takes the filename and file content as params
        :type collect_content_func: Callable[[str, str], Dict]
        """
        try:
            df = pd.read_csv(file_path)
            # the index is the row number
            # the row is a pandas series that has the column value pairs for that row
            for index, row in df.iterrows():
                lines: list = []
                # replacing all missing NaN values in the specifed row with Unknown
                row = row.fillna('Unknown')
                # column is the column name, value is the value for that column in the specified row
                for column, value in row.items():
                    lines.append(f"{column}: {value}")
                # Joining all the column: value lines as one multi line string
                content = "\n".join(lines)
 
                csv_data_dict = collect_content_func(
                    file_path, content, None, index)
                documents.append(csv_data_dict)
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we have a problem {e}")
 
    # This method reads all files from a specified directory
    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Loads every supported document from a directory and returns a list of
        content+metadata dictionaries.
 
        :param directory: Directory path to scan
        :type directory: str
        :return: List of document dicts
        :rtype: List[Dict[str, str]]
        """
        # Document types that can be loaded
        doc_types: List = [".txt", ".pdf", ".md", ".csv"]
        # This list will hold the dictionaries that have content and source keys
        documents: List = []
        files: List = os.listdir(directory)
 
        for file in files:
            file_path: str = os.path.join(directory, file)
            # Getting file extension by using split, also using a throw away variable for the filename because we only need the extension
            _: str
            extension: str
            _, extension = os.path.splitext(file)
            extension: str = extension.lower()
 
            # If this item is not a file skip it.
            if not os.path.isfile(file_path):
                continue
            # If the file extension is not in the approved list skip it
            if extension not in doc_types:
                continue
 
            if extension == ".txt":
                text_file_content: str = self.get_text_content(file_path)
                text_data_dict = self.collect_file_content(
                    source=file_path, content=text_file_content)
                documents.append(text_data_dict)
 
            if extension == ".pdf":
                self.get_pdf_content(file_path, documents,
                                     self.collect_file_content)
 
            if extension == ".md":
                # get_text_content works for .md as well, since it's just text
                markdown_file_content: str = self.get_text_content(file_path)
                markdown_data_dict = self.collect_file_content(
                    source=file_path, content=markdown_file_content)
                documents.append(markdown_data_dict)
 
            if extension == ".csv":
                self.get_csv_content(file_path, documents,
                                     self.collect_file_content)
 
        logger.info("Loaded %d documents from %s", len(documents), directory)
        return documents
 
    def chunk_text(self, text: str, source: str,
                   page: Optional[int] = None, row: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks while preserving metadata.
 
        :param text: The text content to chunk
        :type text: str
        :param source: Source filename/path
        :type source: str
        :param page: Optional page number (for PDFs)
        :type page: Optional[int]
        :param row: Optional row number (for CSVs)
        :type row: Optional[int]
        :return: List of chunk dictionaries with content and metadata
        :rtype: List[Dict[str, str]]
        """
        chunks: List = []
        chunk_size: int = 200
        # Each chunk overlaps the previous by 50 chars so we don't break sentences across boundaries
        chunk_overlap: int = 50
 
        # If the amount of text is smaller than the chunk_size, return a single chunk
        if len(text) <= chunk_size:
            chunk_dict: Dict = self.collect_file_content(
                source, text, page, row)
            chunks.append(chunk_dict)
            return chunks
 
        # step_size = chunk_size - chunk_overlap; this is what enforces the overlap
        step_size: int = chunk_size - chunk_overlap
 
        for starting_slice_position in range(0, len(text), step_size):
            ending_slice_position: int = starting_slice_position + chunk_size
            chunk_text: str = text[starting_slice_position:ending_slice_position]
            chunk_dict: Dict = self.collect_file_content(
                source, chunk_text, page, row)
            chunks.append(chunk_dict)
            if ending_slice_position >= len(text):
                break
 
        return chunks
    

# This class manages Chroma DB and stores chunks as embeddings
class VectorStore:
    """
    This class:
    - Initializes ChromaDB and the embedding model.
    - Stores chunked text as vector embeddings.
    - Handles semantic similarity searches.
    """
 
    def __init__(self, collection_name: str = "tutor_documents", persist_directory: str = "./chroma_db") -> None:
        # Initializing the sentence transformer embedding model (Transformer encoder, 384-dim output)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Initializing the Chroma DB client
        self.db_client: chromadb = chromadb.PersistentClient(path=persist_directory)
        # A collection in ChromaDB is like a table in SQL — one row holds an embedding, the chunked text and metadata
        self.collection = self.db_client.get_or_create_collection(
            name=collection_name)
 
        logger.info("VectorStore initialized")
        logger.info("  - Model: all-MiniLM-L6-v2 (384 dimensions)")
        logger.info("  - Collection: %s", collection_name)
        logger.info("  - Persist directory: %s", persist_directory)
 
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Adds document chunks to the database
 
        :param chunks: List of chunk dictionaries from the DocumentLoader
        :type chunks: List[Dict[str, str]]
        """
        # List comprehension to grab just the text content out of the chunk dicts
        texts: List = [chunk["content"] for chunk in chunks]
        # Generating embeddings for all the text — this is the "Inputs vectorized" step required by the rubric
        embeddings: np.ndarray = self.embedding_model.encode(
            texts, show_progress_bar=False)
        logger.info("Inputs vectorized: %d chunks → embedding shape %s", len(texts), embeddings.shape)
 
        # Converting the embeddings, which is an np array, to a list of lists, which is what chromadb is expecting
        embeddings_list = embeddings.tolist()
        # Creating unique IDs for each chunk by looping through the number of chunks and using it as a counter
        chunk_ids: List = [f"chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas: List = []
        # Looping through the chunks to filter out the None fields so the metadata is clean
        for chunk in chunks:
            metadata: Dict = {
                "source": chunk["source"]
            }
            # ChromaDB requires string format for metadata so converting before adding
            if chunk.get("page") is not None:
                metadata["page"] = str(chunk["page"])
            if chunk.get("row") is not None:
                metadata["row"] = str(chunk["row"])
 
            chunk_metadatas.append(metadata)
 
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=chunk_metadatas
        )
        logger.info("Added %d chunks to ChromaDB collection", len(chunks))
 
    def query_db(self, query: str, num_of_results: int = 3) -> List[Dict[str, str]]:
        """
        Allows the user to query the DB using the user's question.
 
        :param query: The user's question
        :type query: str
        :param num_of_results: Number of results to return, default is 3.
        :type num_of_results: int
        :return: List of relevant chunk dicts containing text, source and metadata
        :rtype: List[Dict[str, str]]
        """
        # Converting the query to embedding before it can be used to query the vector db
        # Doing the embedding manually so I'm not depending on Chroma's auto-embedding behavior
        query_embedding = self.embedding_model.encode(query)
        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_of_results
        )
 
        # Formatting Chroma's wonky list-of-lists output into a clean list of dicts
        results: list = []
        documents: List = search_results["documents"][0] if search_results["documents"] else []
        metadatas: List = search_results["metadatas"][0] if search_results["metadatas"] else []
        for i in range(len(documents)):
            result_dict: Dict = {
                "content": documents[i],
                "source": metadatas[i].get("source", "Unknown"),
                "page": metadatas[i].get("page"),
                "row": metadatas[i].get("row")
            }
            results.append(result_dict)
 
        return results

# The TutorChatbot class ties it all together, document loading, vector retrieval,
# structured prompt construction with few shot examples, and LLM generation
# via the Hugging Face Inference API. 
class TutorChatbot:
    """
    Orchestrator that ties it all together: document loading, vector retrieval,
    structured prompt construction with few-shot examples, and LLM generation
    via the Hugging Face Inference API.
    """
 
    def __init__(self, hf_api_key: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        :param hf_api_key: Hugging Face API key for authentication.
        :type hf_api_key: str
        :param model_name: The LLM that will use the retrieved chunks + the user's question to generate the final answer.
        :type model_name: str
        """
        self.hf_api_key = hf_api_key
        self.model_name = model_name
        self.client = InferenceClient(token=hf_api_key)
        self.document_loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        self.vector_store = VectorStore()
 
        print(f"\n{'='*60}")
        print(f"AI TUTOR INITIALIZED")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Ready to load the knowledge base.\n")


    # This methods loads documents from the directory into chromadb vector database
    def load_knowledge_base(self, directory: str) -> None:
        """
        Loads documents from the directory into ChromaDB vector database.
 
        :param directory: Directory path where documents will be loaded from
        :type directory: str
        """
        # Loading documents from directory
        documents = self.document_loader.load_documents(directory)
        # Empty list to hold chunked documents and their metadata as a list of chunked dictionaries
        all_chunks: List = []
        # Looping through all documents and chunking texts
        for document in documents:
            document_chunks: List[Dict[str, str]] = self.document_loader.chunk_text(
                text=document["content"],
                source=document["source"],
                page=document.get("page"),
                row=document.get("row")
            )
            # Adding the chunked dictionary tot he all chunks list
            all_chunks.extend(document_chunks)
        
        # Loggingg 
        logger.info("Created %d chunks across %d documents", len(all_chunks), len(documents))
        # Adding chunks to the vector store
        self.vector_store.add_documents(all_chunks)
        print("Knowledge base is now online.\n")
    
    # This method query's the vector store(chromaDB) to retrieve relevant context chunks for a query.
    def query_vector_db(self, query: str, num_of_results: int = 3) -> str:
        """
        Queries the vector store (ChromaDB) to retrieve relevant context chunks for a query.
 
        :param query: The user's question
        :type query: str
        :param num_of_results: Number of chunks to be returned that associates with the user's question
        :type num_of_results: int
        :return: A formatted context string that combines all the relevant returned chunks
        :rtype: str
        """
        # Using the user's question to query the vecor DB and specify the
        # number of relevant vectors we want back that corresponds to the user's question
        relevant_chunks: List = self.vector_store.query_db(
            query, num_of_results)
        logger.info("Retrieved %d chunks for query", len(relevant_chunks))
 
        # Formatting each chunk with a source header so the model can cite where the info came from
        # Empty List to hold formatted chunk strings aka context parts
        # I say context parts because the chunks add context to the query when passed to the LLM together
        context_parts: list = []
        # Looping throuh the returned chunks to create a list of formatted chunk strings
        for chunk in relevant_chunks:
            # Creating teh source info header string
            source_info: str = f"Source: {chunk['source']}"
            # If the metadata page exists, add it to the header string
            if chunk.get("page") is not None:
                # Adding the page to the string
                source_info += f", Page: {chunk['page']}"
            # If the matadata for row exists, add it to the header string
            if chunk.get("row") is not None:
                source_info += f", Row: {chunk['row']}"
 
            # Formatting chunk by adding the source_info as a header to the text rtruned in the chunk
            # The source header helps the model understand which text came from which file.
            # This allows the model to reason better
            formatted_chunk = f"[{source_info}]\n{chunk['content']}\n"
            # Adding the formatted chunk with the source header to the context_parts list
            context_parts.append(formatted_chunk)
 
        # joining all chunks together with a separator
        # I am using a separator because without a separator things get messy real fast.
        # chunk boundaries will disappear and sources are blurred
        # This can make "reasoning" wonky, so the separator makes it useful context
        context: str = "\n---\n".join(context_parts)
        return context
    
    # This method builds the structured prompting strategy 
    def build_messages(self, query: str, context: str) -> List[Dict[str, str]]:

        # Starting the messages list with the system message
        # The system message is what defines the tutors persona,
        # the 5 part output structure defined aearlier, and the grounding rules
        # This msut be first because chat completion APIs treat the first system message as the persona instruction
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": tutor_system_prompt},
        ]

        # Adding the few shot examples right after the system message
        messages.extend(few_shot_examples)

        # Now appending the real user message at the end
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
        })

        return messages
    
    # This method generates a response using the Hugging Face Inference API.
    def generate_response(self, query: str, context: str) -> str:
        """
        Generates a response using the Hugging Face Inference API.
        Combines the system prompt, few shot examples, retrieved context, and user question.
 
        :param query: The user's question
        :type query: str
        :param context: Context retrieved from the vector DB
        :type context: str
        :return: The generated answer from the LLM
        :rtype: str
        """
        # Building the structured messages list (system few-shot user)
        messages = self.build_messages(query, context)
        logger.info("Built messages list (%d messages)", len(messages))
 
        try:
            # Calling the HF Inference API this is the "Model called successfully" step from the rubric
            logger.info("Calling HF model %s ...", self.model_name)
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                # limits the response length so the tutor stays focused
                max_tokens=400,
                # This controls how random vs deterministic the output will be, 0.7 is balanced
                temperature=0.5,
                # nucleus sampling so we only consider the most likely tokens whose combined probability adds up to top_p
                top_p=0.9,
            )

            # Getting the generated text from teh response
            generated_text: str = response.choices[0].message.content
            logger.info("Model called successfully (response length=%d chars)", len(generated_text))

            return generated_text
 
        except Exception as e:
            # Saving error message:
            error_msg = str(e)

            # Checking for th eissues i ran into during buildign and testing along with common issues referenced in the documentation
            if "403" in error_msg or "not have access" in error_msg.lower():
                return f"Error: You need to accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Details: {error_msg}"
            
            elif "503" in error_msg or "loading" in error_msg.lower():
                return "Error: Model is loading. Please wait 30 seconds and try again."
            
            elif "429" in error_msg or "rate" in error_msg.lower():
                return "Error: Rate limited. Please wait a minute and try again."
            
            else:
                return f"Error generating response: {error_msg}"
    
    # This method will be the main chat function that will run the full RAG workflow/pipeline
    def tutor(self, query: str) -> str:
        """
        Main pipeline: retrieve context → build prompt with few-shot → generate answer.
 
        :param query: The user's question
        :type query: str
        :return: The final tutor-style answer
        :rtype: str
        """

        # Using the user's question(query) to retrieve additional context from vector DB
        context: str = self.query_vector_db(query)
        answer: str = self.generate_response(query, context)
        return answer
 

def main():
    print("\n" + "="*60)
    print("AI TUTOR: Week 7 Project")
    print("="*60)
    print("\nAsk me questions about the documents in the knowledge base.")
    print("I retrieve relevant context first, then explain in tutor style.\n")
    
    # Getting hugging face api key from user input and removing white space
    hf_api_key: str = input("Enter your Hugging Face API key: ").strip()
    if not hf_api_key:
        print("Your API key is required.")
        return
 
    tutor = TutorChatbot(hf_api_key)
 
    # Default knowledge base directory
    data_dir = "./documents_project7/"
 
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found!")
        return
 
    # Embed all documents and load them into ChromaDB
    tutor.load_knowledge_base(data_dir)
 
    print("\n" + "="*60)
    print("CHAT STARTED")
    print("="*60)
    print("Type 'quit', 'exit', or 'q' to end the session.\n")
 
    while True:
        user_question = input("\nYou: ").strip()
 
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye — happy studying!\n")
            break
 
        if not user_question:
            print("Please ask a question.")
            continue
 
        answer = tutor.tutor(user_question)
        print(f"\nTutor: {answer}")
 
 
if __name__ == "__main__":
    main()