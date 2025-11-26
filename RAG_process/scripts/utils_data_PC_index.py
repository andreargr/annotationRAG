import os
from pathlib import Path
from rdflib import Graph
import re
import json
import chromadb

# For langchain and RAG
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


os.environ['CURL_CA_BUNDLE'] = ''

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

global_uri_to_description = {}

def dataframe2prettyjson(dataframe: pd.DataFrame, file: str = None, save: bool = False) -> str:
    """
    Convert a Pandas DataFrame to pretty JSON and optionally save it to a file.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        file (str): The file path to save the pretty JSON.
        save (bool): Whether to save the JSON to a file.

    Returns:
        str: The pretty JSON string representation.
    """
    try:
        json_data = dataframe.to_json(orient='index')
        parsed = json.loads(json_data)
        pretty_json = json.dumps(parsed, indent=4)

        if save and file:
            with open(file, 'w') as f:
                f.write(pretty_json)

        return pretty_json
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {str(je)}")
    except ValueError as ve:
        print(f"Value Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return ""

def load_ontology(ontology_path: str, ontology_format: str = "turtle") -> Graph:
    """
    Load an ontology from a file in the specified format.

    Args:
        ontology_path (str): Path to the ontology file.
        ontology_format (str): Format of the ontology (default is "turtle").

    Returns:
        Graph: rdflib Graph object containing the ontology.
    """
    g = Graph()
    g.parse(ontology_path, format=ontology_format)
    return g


def query_elements(graph: Graph,prefix) -> dict:
    """
    Execute a SPARQL query to fetch classes and their identifiers.

    Args:
        graph (Graph): rdflib Graph object containing the ontology.

    Returns:
        dict: Dictionary where keys are class labels and values are their identifiers.
    """
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/> 
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 
    SELECT ?element ?label (COALESCE(?description,?definition,?comm) AS ?information)
    WHERE {{
    {{
        ?element a owl:Class .
        ?element rdfs:label ?label . 
        OPTIONAL {{ ?element dc:description ?description . }}
        OPTIONAL {{ ?element skos:definition ?definition . }}
        OPTIONAL {{ ?element rdfs:comment ?comm . }}
    }}
    UNION
    {{
        ?element a owl:DatatypeProperty .
        ?element rdfs:label ?label .
        OPTIONAL {{ ?element dc:description ?description . }}
        OPTIONAL {{ ?element skos:definition ?definition . }}
        OPTIONAL {{ ?element rdfs:comment ?comm . }}
    }}
    UNION
    {{
        ?element a owl:ObjectProperty .
        ?element rdfs:label ?label .
        OPTIONAL {{ ?element dc:description ?description . }}
        OPTIONAL {{ ?element skos:definition ?definition . }}
        OPTIONAL {{ ?element rdfs:comment ?comm . }}
    }}
    }}
    """
    results = graph.query(query)

    label_to_uri = {}
    for row in results:
        class_uri = str(row[0])  # First element: URIRef
        class_id = class_uri.split("/")[-1]  # Extract the ID from the URI
        if class_id.startswith(prefix):
            class_label = str(row[1]) if len(row) > 1 and row[1] else class_id  # Use label if available, otherwise ID
            label_to_uri[class_label] = f"<{class_id}>"
        # class_label = str(row[1]) if len(row) > 1 and row[1] else class_uri.split("/")[-1]  # Second element: Literal, if present
        # label_to_uri[class_label] = f"<{class_uri}>"
        class_description = str(row[2]) if row[2] else None  # Third element: URIRef

        if class_description:
            global_uri_to_description[class_uri] = class_description

    return label_to_uri


def save_label_to_uri(data: dict, output_path: str, ontology_name: str) -> None:
    """
    Save a dictionary to a text file.

    Args:
        data (dict): Dictionary with data to save.
        output_path (str): Path to the output file.
    """
    file_path = os.path.join(output_path, f"label_to_uri_{ontology_name}.ttl")
    with open(file_path, "w", encoding="utf-8") as f:
        for label, uri in data.items():
            f.write(f"{label} : {uri}\n")


def save_uri_to_description(data: dict, output_path: str) -> None:
    """
    Save a dictionary to a text file.

    Args:
        data (dict): Dictionary with data to save.
        output_path (str): Path to the output file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for uri, description in data.items():
            f.write(f"{uri} : {description}\n")


def process_directory(input_directory: str, output_directory: str, output_description: str,
                      ontology_format: str = "turtle"):
    """
    Process all ontology files in a directory, execute the SPARQL query,
    and save the results_4o_mini for each file.

    Args:
        input_directory (str): Directory containing ontology files.
        output_directory (str): Directory where the result files will be saved.
        ontology_format (str): Format of the ontology (default is "turtle").
    """

    for filename in os.listdir(input_directory):
        if filename.endswith(".ttl"):
            base_name = filename[:-4]
            input_path = os.path.join(input_directory, filename)

            print(f"Processing: {filename}")
            graph = load_ontology(input_path, ontology_format)
            classes_dict = query_elements(graph,base_name)
            save_label_to_uri(classes_dict, output_directory, base_name)
            print(f"Results saved in: {output_directory}")
    save_uri_to_description(global_uri_to_description, output_description)

def rag_loader(database):  # Data preprocessing and split
    ontologies = []
    directory_path = f'./ontologies/{database}'
    description_file = "./ontologies/global_uri_to_description.txt"

    process_directory(directory_path, directory_path, description_file)

    # Iterate over all files in the directory
    for file in os.listdir(directory_path):
        if file.startswith("label_to_uri_"):  # Check if the file ends with .txt
            with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ontologies.append(line)

    # Create a DataFrame with the content
    df = pd.DataFrame({"output_reduced_ontologies_4o_mini": ontologies})
    print(df)

    loader = DataFrameLoader(df, page_content_column="output_reduced_ontologies_4o_mini")
    data = loader.load()

    return data


def save_chromedb(documents, full_name):
    """
        Include the embeddings into the target vector database, specifically Chroma DB in this case.
    """
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings,
                                        persist_directory=full_name)  # Use openAI embedding to get text vector
    vectorstore.persist()


def execute_if_not_exists(persist_directory,database):
    directory = Path(persist_directory)
    if not directory.is_dir():
        documents = rag_loader(database)
        save_chromedb(documents, persist_directory)
        print("Ontologies correctly stored.")
    else:
        print("Using already created Chroma directory.")

def dataframe2prettyjson(dataframe: pd.DataFrame, file: str = None, save: bool = False) -> str:
    """
    Convert a Pandas DataFrame to pretty JSON and optionally save it to a file.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        file (str): The file path to save the pretty JSON.
        save (bool): Whether to save the JSON to a file.

    Returns:
        str: The pretty JSON string representation.
    """
    try:
        json_data = dataframe.to_json(orient='index')
        parsed = json.loads(json_data)
        pretty_json = json.dumps(parsed, indent=4)

        if save and file:
            with open(file, 'w') as f:
                f.write(pretty_json)

        return pretty_json
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {str(je)}")
    except ValueError as ve:
        print(f"Value Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return ""

def load_dataset(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    sample_dataframe = dataframe.head(10)
    json_data = dataframe2prettyjson(sample_dataframe)
    return json.loads(json_data)

def load_description(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_txt_file_as_string(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

def similarity_search_context(query, persist_directory):
    """
            A similarity search is performed based on the cosine similarity of the query performed and the top 4 results_4o_mini are returned.
    """
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings)  # Initiate the chromadb with the directory of interest
    docs = vectordb.similarity_search(query, k=10)
    context = ""
    for document in docs:
        if document.page_content not in context:
            context = context + document.page_content + '\n'
    return context

def extract_iris_and_descriptions(label,persist_directory):
    print("Extracting identifiers for label: ", label)
    iri_candidates = []
    iri_descriptions = []
    description_file= "../../ontologies/global_uri_to_description.txt"
    result = similarity_search_context(label,persist_directory)  # get the examples from the RAG
    iris = re.findall(r'<([^>]+)>', result)
    #print("IRIs extracted:", iris)
    with open(description_file, "r", encoding="utf-8") as f:
        for line in f:
            for iri in iris:
                if iri in line:
                    iri_descriptions.append(line)
    iri_candidates.append(result)
    return "\n".join(iri_candidates), "\n".join(iri_descriptions)


def parse_json_from_string(raw_string):
    """
    Cleans and parses a string that may contain Markdown-style JSON code blocks
    (```json ... ```), returning a Python dictionary.

    Args:
        raw_string (str): A string potentially containing JSON.

    Returns:
        dict: Parsed JSON as a Python dictionary.

    Raises:
        ValueError: If the string cannot be parsed as valid JSON.
    """
    # Try to extract JSON content between markdown-style delimiters
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_string, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = raw_string.strip()  # Attempt to parse the string directly

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def save_candidates_json(output_folder, label, ontology_key, query_term, candidates):
        """
        Save candidate IRIs related to a label and ontology into a JSON file.
        Creates the file if it does not exist yet.
        """
        candidates_output_path = os.path.join(output_folder, "label_candidates.json")

        entry = {
            "Label": label,
            "Ontology": ontology_key,
            "Query": query_term,
            "Candidates": candidates
        }

        # Load existing data or start a new list
        if os.path.exists(candidates_output_path):
            with open(candidates_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        # Write updated data to file
        with open(candidates_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def save_label_inference_json(output_folder, label, cell_line, cell_type, anatomical_structure, additional_info):
    """
    Save label inference results (cell line, cell type, anatomical structure, etc.)
    to a JSON file. Creates or updates the file if it already exists.
    """
    label_info_path = os.path.join(output_folder, "label_inferences_gpt5_mini.json")

    entry = {
        "Label": label,
        "Cell_Line": cell_line,
        "Cell_Type": cell_type,
        "Anatomical_Structure": anatomical_structure,
        "Additional_Info": additional_info
    }

    # Load existing data or create a new list
    if os.path.exists(label_info_path):
        with open(label_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Add the new entry
    data.append(entry)

    # Save updated data back to file
    with open(label_info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




