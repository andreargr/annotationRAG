import argparse
from utils_data_PC_index import extract_iris_and_descriptions,parse_json_from_string,read_txt_file_as_string, execute_if_not_exists, dataframe2prettyjson
import json
from dotenv import dotenv_values
import openai
import pandas as pd
import os

db_metadata = {
    "CLO": {
        "persist_directory": "../VDB_reduced_ontologies_index/CLO",
        "output_path": "CLO_identifiers_GPT4o_mini.txt"
    },
    "CL": {
        "persist_directory": "../VDB_reduced_ontologies_index/CL",
        "output_path": "CL_identifiers_GPT4o_mini.txt"
    },
    "UBERON": {
        "persist_directory": "../VDB_reduced_ontologies_index/UBERON",
        "output_path": "UBERON_identifiers_GPT4o_mini.txt"
    },
    "BTO":{
        "persist_directory" : "../VDB_reduced_ontologies_index/BTO",
        "output_path": "BTO_identifiers_GPT4o_mini.txt"
    }
}

def load_environment():
    """
    Load environment variables.
    :return: The OPENAI API key.
    """
    config = dotenv_values(dotenv_path=".env")
    return config.get('OPENAI_API_KEY')

# Define the API key of OpenAI
openai.api_key=load_environment()

class GptLLM:
    def __init__(self, model_name, temperature):
        if model_name not in {
            'gpt-4o-mini', 'gpt-4.1-mini-2025-04-14','gpt-5-mini-2025-08-07'
        }:
            raise Exception('Invalid model')

        self.model = model_name
        self.temperature = temperature
        self.responses = []

    def run_inference(self, chat_template):
        if self.model == "gpt-5-mini-2025-08-07":
            completion = openai.chat.completions.create(
                model=self.model,
                messages=chat_template,
                max_completion_tokens=4096 #para el modelo gpt-5
            )
        else:
            completion = openai.chat.completions.create(
                model=self.model,
                messages=chat_template,
                max_tokens=4096,
                temperature=self.temperature, #el modelo gpt-5 solo admite temperatura de 1
            )

        response = completion.choices[0].message.content
        self.responses.append({"input": chat_template, "response": response})
        #print(response)

        return response

    def get_response(self):
        return self.responses

    def save_responses_to_json(self, outputpath):
        responses = [entry["response"].strip("`").strip() for entry in self.responses]
        with open(outputpath, "a", encoding="utf-8") as file:
            for response in responses:
                file.write(response + "\n")

    def clean_responses(self):
        self.responses = []

class LlmLabelInference(GptLLM):
    def __init__(self, model_name, temperature=0.5):
        super().__init__(model_name, temperature)
        self.prompt_path = "prompts/search_interpretation.txt"
        self.prompt = ""

    def set_prompt(self, label, concept_type):
        """
        Function to build the prompt
        """
        initial_prompt = read_txt_file_as_string(self.prompt_path)
        system_prompt = "As a biological science expert, I would like your assistance in interpreting a label associated with biological samples."
        initial_prompt = initial_prompt.format(label=label,concept_type=concept_type)
        conversation = []
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": initial_prompt})
        self.prompt = conversation

    def get_prompt(self):
        return self.prompt

    def clean_responses(self):
        super().clean_responses()

class LlmIRIGpt(GptLLM):
    def __init__(self, model_name, json_data, temperature=0.5):
        super().__init__(model_name, temperature)
        self.prompt_path = "prompts/IRIsearch_instructions_restrictions.txt"
        self.system_path = "system_prompt.txt"
        self.json_data = json_data
        self.prompt = ""

    def set_prompt(self, label, label_context,candidates,candidate_descriptions):
        """
        Function to build the prompt
        """
        initial_prompt = read_txt_file_as_string(self.prompt_path)
        system_prompt = read_txt_file_as_string(self.system_path)
        initial_prompt = initial_prompt.format(label=label,label_context=label_context, candidates=candidates,candidate_descriptions=candidate_descriptions)
        conversation = []
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": initial_prompt})
        self.prompt = conversation

    def get_prompt(self):
        return self.prompt

    def clean_responses(self):
        super().clean_responses()


def main(args):
    output_folder = args.output_folder
    dataset_path = args.dataset_path
    model_name = args.model_name

    dataframe = pd.read_csv(dataset_path, sep='\t', header=0)  # tsv file loaded
    #dataframe = pd.read_csv(dataset_path, sep=',', header=0)
    dataframe.columns = ['Label', 'CLO', 'CL', 'UBERON', 'BTO', 'Type']
    json_string = dataframe2prettyjson(dataframe)  # tsv2json
    json_data = json.loads(json_string)

    print('---------------------------------------------------')
    llm_IRI_search = LlmIRIGpt(model_name, json_data)  # Main task: Search identifiers from the synthesized JSON data

    #load ontologies in chroma
    for db_name, metadata in db_metadata.items():
        ontology = db_name
        persist_directory = metadata["persist_directory"]
        execute_if_not_exists(persist_directory,ontology)  # create the embeddings and store them in Chromadb
        subdirectory = metadata["output_path"]
        full_outputpath = os.path.join(output_folder, subdirectory)
        print(f"Output path: {full_outputpath}")

    results = []  # List to accumulate final results
    for key, value in json_data.items():
        label = value.get("Label")
        concept_type = value.get("Type")

        # ----------- INFER LABEL MEANING ------------
        llm_infer_info = LlmLabelInference("gpt-5-mini-2025-08-07")
        llm_infer_info.set_prompt(label, concept_type)
        result = llm_infer_info.run_inference(llm_infer_info.prompt)
        inference = parse_json_from_string(result)

        # Extract inferred values
        cell_line = inference["Cell Line"]
        cell_type = inference["Cell Type"]
        anatomical_structure = inference["Anatomical Structure"]
        additional_info = inference["Interpretation"]

        label_info_path = os.path.join(output_folder, "label_inferences_gpt5_mini.json")
        new_entry = {
            "Label": label,
            "Cell_Line": cell_line,
            "Cell_Type": cell_type,
            "Anatomical_Structure": anatomical_structure,
            "Additional_Info": additional_info
        }

        # Crea o actualiza archivo JSON
        if os.path.exists(label_info_path):
            with open(label_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append(new_entry)

        with open(label_info_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print("Label:",label,"\n Linea celular:",cell_line,"\n Tipo celular:",cell_type,"\n Organo:",anatomical_structure,"\n Info:",additional_info)
        # Initialize ID variables
        clo_id = "-"
        bto_id = "-"
        cl_id = "-"
        uberon_id = "-"

        # ---------- Search CLO and BTO using Cell Line ----------
        if concept_type == "CL":
            for ontology_key in ["CLO", "BTO"]:
                persist_directory = db_metadata[ontology_key]["persist_directory"]
                query_term = cell_line if cell_line != "-" else label

                candidates, candidates_descriptions = extract_iris_and_descriptions(query_term, persist_directory)
                print("\nCandidates for", query_term, ontology_key," ontology: \n", candidates)
                llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
                ontology_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

                if ontology_key == "CLO":
                    clo_id = ontology_id
                else:  # BTO
                    bto_id = ontology_id

            # ---------- Search CL using Cell Type (or label fallback) ----------
            query_term = cell_type if cell_type != "-" else label
            persist_directory = db_metadata["CL"]["persist_directory"]
            candidates, candidates_descriptions = extract_iris_and_descriptions(
                query_term, persist_directory
            )
            print("\nCandidates for", query_term," CL ontology: \n", candidates)
            llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
            cl_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

            # ---------- Search UBERON using Anatomical Structure (or label fallback) ----------
            query_term = anatomical_structure if anatomical_structure != "-" else label
            persist_directory = db_metadata["UBERON"]["persist_directory"]
            candidates, candidates_descriptions = extract_iris_and_descriptions(
                query_term, persist_directory
            )
            print("\nCandidates for", query_term," UBERON ontology: \n", candidates, )
            llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
            uberon_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

        elif concept_type == "CT":
            # ---------- Search CL and BTO using Cell Type ----------
            for ontology_key in ["CL", "BTO"]:
                persist_directory = db_metadata[ontology_key]["persist_directory"]
                query_term = cell_type if cell_type != "-" else label

                candidates, candidates_descriptions = extract_iris_and_descriptions(query_term, persist_directory)
                print("\nCandidates for", query_term, ontology_key, " ontology: \n", candidates)
                llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
                ontology_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

                if ontology_key == "CL":
                    cl_id = ontology_id
                else:  # BTO
                    bto_id = ontology_id

            # ---------- Search UBERON using Anatomical Structure (or label fallback) ----------
            query_term = anatomical_structure if anatomical_structure != "-" else label
            persist_directory = db_metadata["UBERON"]["persist_directory"]
            candidates, candidates_descriptions = extract_iris_and_descriptions(
                query_term, persist_directory
            )
            print("\nCandidates for", query_term, " UBERON ontology: \n", candidates)
            llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
            uberon_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

        elif concept_type == "A":
            # ---------- Search UBERON and BTO using Anatomical Structure ----------
            for ontology_key in ["UBERON", "BTO"]:
                persist_directory = db_metadata[ontology_key]["persist_directory"]
                query_term = anatomical_structure if anatomical_structure != "-" else label

                candidates, candidates_descriptions = extract_iris_and_descriptions(query_term, persist_directory)
                print("\nCandidates for", query_term, ontology_key, " ontology: \n", candidates)
                llm_IRI_search.set_prompt(query_term, additional_info, candidates, candidates_descriptions)
                ontology_id = llm_IRI_search.run_inference(llm_IRI_search.prompt)

                if ontology_key == "UBERON":
                    uberon_id = ontology_id
                else:  # BTO
                    bto_id = ontology_id

        print("Row saved:",label,concept_type,clo_id,cl_id,uberon_id,bto_id,"\n")
        # ---------- Add results to the list ----------
        results.append(
            {
                "Label": label,
                "Type": concept_type,
                "CLO_id": clo_id,
                "CL_id": cl_id,
                "UBERON_id": uberon_id,
                "BTO_id": bto_id,
            }
        )

    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_folder, "100_gpt_5_mini_inference_restrictions.csv")
    results_df.to_csv(output_csv, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output_folder",  type=str, required=True, help="Name of the folder in which the model response is stored.")
    parser.add_argument("-dp","--dataset_path", type=str, required=True, help="Path to the TSV file with the labels to be mapped.")
    parser.add_argument("-m","--model_name", type=str, required=True, help="GPT model to use.")

    
    args = parser.parse_args()
    main(args)
