# Import necessary modules
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# def list_files_with_extension(folder_path):
#     # Get all files in the folder
#     files = os.listdir(folder_path)
#     # Initialize lists to store files with different extensions
#     pdf_files = []
#     word_files = []
#     text_files = []
#     powerpoint_files = []
#     # Iterate over the files and extract their names along with extensions
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         if os.path.isfile(file_path):
#             file_name, file_extension = os.path.splitext(file)
#             if file_extension.lower() == '.pdf':
#                 pdf_files.append(file_path)
#             elif file_extension.lower() in ['.doc', '.docx']:
#                 word_files.append(file_path)
#             elif file_extension.lower() == '.txt':
#                 text_files.append(file_path)
#             elif file_extension.lower() in ['.ppt', '.pptx']:
#                 powerpoint_files.append(file_path)
#     return pdf_files, word_files, text_files, powerpoint_files

# pdf_files, word_files, text_files, powerpoint_files = list_files_with_extension('./data')

# # print(pdf_files, '\n', word_files)

# file_paths = [pdf_files[0],word_files[0]]

# print(file_paths)

# def load_documents(file_paths):
#     documents = []
#     for path in file_paths:
#         if path.endswith('.pdf'):
#           loader = PyPDFLoader(path)
#         elif path.endswith('.docx'):
#             loader = Docx2txtLoader(path)
#             # print(loader)
#         elif path.endswith('.txt'):
#             loader = UnstructuredFileLoader(path)
#         elif path.startswith('https://www.youtube.com'):
#             loader = YoutubeLoader.from_youtube_url(path, add_video_info=True)
#         else:
#             loader = WebBaseLoader(path)
#         documents.extend(loader.load())
#     return documents

def create_vector_db():
    """Create a vector database from PDF documents."""
    # Load documents from the specified directory
    
    loader = PyPDFDirectoryLoader(os.getenv('DATA_PATH'))
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")
    print(documents)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(texts)

    # Create a vector store from the document chunks
    vectorStore = Chroma.from_documents(
        documents=texts,
        embedding=OllamaEmbeddings(model='llama3'),
        persist_directory=os.getenv('DB_PATH')
    )

    # Persist the vector store to disk
    vectorStore.persist()


def main():
    """Main function to create the vector database."""
    create_vector_db()


if __name__ == "__main__":
    main()
