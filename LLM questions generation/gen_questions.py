from info import MODEL_NAMES, PROMPT, QUESTIONS_PATH, CSV_PATH, CORPUS_INFO_PATH
from load_models import main
import re
import os
from lxml import etree
import tqdm
import time
import pandas as pd
import math

CLIENT = main()

def drop_reasoning(text: str) -> str:
    """
    Drop the reasoning part from the text if it is present.
    
    :param text: The text to process.
    :return: The text without the reasoning part.
    """
    match = re.match(r'<think>(?P<reasoning>.*?)</think>(?P<answer>.*)', text, re.DOTALL)
    if match:
        return match.group('answer').strip()
    return text.strip()

def generate_QUDs_from_dialogue(utterances: list[tuple[str, str]], 
                                model: str, 
                                file_name: str = "temp_generation", 
                                file_path: str = None) -> tuple[list[str], bool]: # type: ignore
    """
    Generate QUDs from a dialogue using the Ollama API.
    
    :param utterances: List of tuples containing speaker and utterance.
    :param model: The name of the model to use for generation.
    :param file_name: The name of the file to save the generated questions (without extension).
    :param file_path: The path to the original dialogue file (for reference in the XML).
    :return: List of generated results.
    """
    generated_questions = []

    full_save_file_name = f'{QUESTIONS_PATH}/{file_name}_{model}.xml'

    finished = True  # Flag to indicate if the generation was successful
    last_id = 2 # Start from the 4th utterance (index 2 + 2) as per the original code
    # If we have already generated questions, we will continue from there, incrementing 2 by 2.

    # If the file already exists, we will append to it,
    # otherwise we will create a new one with the header as an xml file
    if not os.path.exists(full_save_file_name):
        # Create the root element
        root = etree.Element('potentialQuestionsGen') # type: ignore

        # Make a new document tree
        doc = etree.ElementTree(root)

        # Add the subelements
        fileDesc = etree.SubElement(root, 'desc', 
                                            model=model,
                                            file=file_path if file_path is not None else file_name,) # type: ignore
        generation = etree.SubElement(root, 'generation') # type: ignore
    else:
        parser = etree.XMLParser(remove_blank_text=True)
        doc = etree.parse(full_save_file_name, parser)
        generation = doc.find('generation')  # type: ignore
        # find the last g in generation to continue from there
        if generation is None:
            raise ValueError("The XML file does not contain a 'generation' element.")
        else:
            last_g = generation.findall('g')[-1] if generation.findall('g') else None
            if last_g is not None:
                last_id = int(last_g.get('id', '0'))
            else:
                last_id = 2
        print(f"Continuing with {model} from last ID: {last_id}")

    try:
        # Iterate over the utterances with a progress bar
        # We probe every second utterance, from the 4th one
        for i, (speaker, utterance) in tqdm.tqdm(enumerate(utterances[last_id + 2::2]), desc=f"Generating with {model}", total=len(utterances[last_id + 2::2])):
            id_utterance = i * 2 + last_id + 2  # Adjust index to match the original utterances list
            context = '\n'.join([f"{s}: {u}" for s, u in utterances[max(0,id_utterance-20):id_utterance+1]])  # Context includes all previous utterances
            prompt = PROMPT.format(conversational_context=context, anchor_utterance=f"{speaker}: {utterance}")
            result = CLIENT.generate(model, prompt=prompt)
            generated_questions.append(result.response.strip())

            # Add to the XML document
            element = etree.SubElement(generation, 'g',
                                    id=str(id_utterance)) # type: ignore
            

            previous_utterance = etree.SubElement(element, 'previous_utterance') # type: ignore
            previous_utterance_element = utterances[id_utterance - 1]
            previous_utterance.text = f"{previous_utterance_element[0]}: {previous_utterance_element[1].strip()}"

            utterance = etree.SubElement(element, 'utterance') # type: ignore
            utterance_element = utterances[id_utterance]
            utterance.text = f"{utterance_element[0]}: {utterance_element[1].strip()}"

            full_generation = etree.SubElement(element, 'full_generation') # type: ignore
            full_generation.text = result.response.strip()
            
            clean_generation = etree.SubElement(element, 'clean_generation') # type: ignore
            clean_generation.text = drop_reasoning(result.response.strip())
    
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        finished = False
    
    finally:
        # Save to XML file
        with open(full_save_file_name, 'wb') as outFile:
            # Write the document tree to the file with UTF-16 encoding
            doc.write(outFile, xml_declaration=True, encoding='utf-16', pretty_print=True)  
    
    return generated_questions, finished  # Assuming the response is in the correct tuple format.

def main():
    corpus_df = pd.read_csv(CORPUS_INFO_PATH)

    files = {f"{row['category']}_{int(i) + 1}": (row['file_name'], row['end']) for i, row in corpus_df.iterrows()} # type: ignore
    done = {model: set() for model in MODEL_NAMES}  # Initialise done for each file key

    while not all([len(done[file_key]) == len(files) for file_key in done]):
        try:
            for model in MODEL_NAMES:
                for file_key, (file_path, end) in files.items():
                    if file_key not in done[model]:
                        csv_file = f"{CSV_PATH}/{file_path[:-4]}.csv"
                        utterances_df = pd.read_csv(csv_file)
                        utterances = list(zip(utterances_df['speaker'], utterances_df['utterance']))

                        if not math.isnan(end):
                            utterances = utterances[:int(end+1)]

                        quds, no_error = generate_QUDs_from_dialogue(utterances, model, file_name=file_key, file_path=file_path)

                        if no_error:
                            print(f"Successfully generated questions for {file_key} with model {model}.")
                            done[model].add(file_key)
                        else:
                            print(f"Error occurred while generating questions for {file_key} with model {model}. Retrying later")
                print()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 2 seconds...")
            time.sleep(2)



if __name__ == "__main__":
    main()