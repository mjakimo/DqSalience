from info import QUESTIONS_PATH, ANNOTATION_SHEETS_PATH, MODEL_NAMES, CORPUS_INFO_PATH, CSV_PATH
import pandas as pd
import os
from lxml import etree # type: ignore
from itertools import combinations
import json
import random

def get_default_rating(question):
    if question.count('?') > 1:
        return 0
    elif 'unclear' in question:
        return 2
    else:
        return -1

def create_annotation_sheets_tool(category, models=MODEL_NAMES, save_dir=ANNOTATION_SHEETS_PATH, json_file_name=None):
    """
    Create annotation sheets for one file in the corpus.
    """

    # Create a directory for annotation sheets if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    parser = etree.XMLParser(remove_blank_text=True)

    # Find all the potential questions files for the given category
    potential_questions_files = [f"{category}_{model}.xml" for model in models]
    potential_questions_xml = [etree.parse(f"potential_questions/{file}", parser) for file in potential_questions_files]
    # Get the list of utterance IDs (<g id="4">) from one of the potential questions files
    if potential_questions_files:
        dialogue_json = []
        questions_json = []

        utterance_ids = potential_questions_xml[0].xpath('//g/@id')
        dialogue_file_name = f"{CSV_PATH}/{potential_questions_xml[0].xpath('//desc/@file')[0].split('/')[-1][:-4]}.csv"  # Remove .xml extension
        dialogue_df = pd.read_csv(dialogue_file_name)
        dialogue_json = [{"speaker": row['speaker'], "utterance": row['utterance']} for _, row in dialogue_df.iterrows()]

        ## Retrieve the questions
        groups = list(combinations(range(len(potential_questions_files)), len(models)))
        for i,group in enumerate(groups):
            models = [potential_questions_files[j].split('_')[-1][:-4] for j in group]
            xml_files = [potential_questions_xml[j] for j in group]
            print(f"Creating annotation sheet for group {i+1}: {', '.join(models)}")

            for utterance_id in utterance_ids:
                # For each utterance ID, retrieve the corresponding questions from the XML files
                all_questions = []
                utterance_full_id = ''
                for xml_tree in xml_files:
                    generation = xml_tree.xpath(f'//g[@id="{utterance_id}"]')
                    if generation:
                        questions = generation[0].find('clean_generation').text.split('\n')
                        for question in questions:
                            all_questions.append(question.strip())
                        utterance_full_id = generation[0].find('utterance').text.split(':')[0]
                if all_questions and utterance_full_id:
                    # If there are questions for this utterance ID, add them to the questions list
                    for question in all_questions:
                        # Add the question to questions
                        default_rating = get_default_rating(question)
                        if default_rating >= 0:
                            questions_json.append({
                                "id": len(questions_json),
                                "text": question,
                                "cutoff": utterance_id,
                                "default_rating": default_rating
                            })
                        else:
                            questions_json.append({
                                "id": len(questions_json),
                                "text": question,
                                "cutoff": utterance_id
                            })
                            
            # Save a JSON file of the form {"dialogue": dialogue, "questions": questions}
            if json_file_name is None:
                json_file_name = f'{save_dir}/{category}_{i+1}_annotations.json'
            elif len(groups) > 1:
                json_file_name = f'{save_dir}/{json_file_name[:-5] if json_file_name.endswith(".json") else json_file_name}_{i+1}.json'
            else:
                json_file_name = f'{save_dir}/{json_file_name}{"" if json_file_name.endswith(".json") else ".json"}'

            with open(json_file_name, 'w', encoding='utf-8') as f:
                json.dump({"dialogue": dialogue_json, "questions": questions_json}, f, ensure_ascii=False, indent=4)
    


if __name__ == "__main__":

    corpus = pd.read_csv('./corpus.csv', header=0)
    models = ["qwen3.5:27b", "qwq"]

    # Prepare the excel sheet where the progress of the annotations will be tracked
    todo_list = []
    
    with open(f'{ANNOTATION_SHEETS_PATH}/annotation_sheets_info.csv', 'w', encoding='utf-8') as sheet_info_file:
        sheet_info_file.write("Category,Models,FileNumber\n")

        # Create random order of the files to annotate
        annotation_order = list(range(len(corpus)))
        random.seed(42)
        random.shuffle(annotation_order)

        for index, row in corpus['category'].items():
            # Use the shuffled order to get a 1-based file number
            annotation_number = annotation_order[int(index)] + 1
    
            create_annotation_sheets_tool(f"{row}_{int(index)+1}", models=models, save_dir=ANNOTATION_SHEETS_PATH, json_file_name=f"sheet_{annotation_number}")
    
            sheet_info_file.write(f"{row}_{int(index)+1},{'|'.join(models)},{annotation_number}\n")
    
            todo_list.append([int(index)+1, '', ''])
        
        todo_df = pd.DataFrame(todo_list, columns=['FileNumber', 'Annotator', 'Comment'])
        todo_df.to_excel(f'{ANNOTATION_SHEETS_PATH}/ToDo.xlsx', index=False)

