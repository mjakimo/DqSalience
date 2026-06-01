import os
import ast
import csv


input_file = "Hao/Nb_utterance_Corpus_Hao.txt" #Filepath



output_dir = "Speakers_Stats_CVS" #The directory, where to put the csv file
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue

    file_name, dict_part = line.split(":", 1)
   
    speaker_dict = ast.literal_eval(dict_part.strip())

    total_speakers = len(speaker_dict)
    total_utterances = sum(speaker_dict.values())

    csv_name = file_name.replace(".txt", ".csv")
    csv_path = os.path.join(output_dir, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["speaker", "nb_utterances"])
        
        for speaker, count in speaker_dict.items():
            writer.writerow([speaker, count])

print("Every files are computed")
