import pandas as pd
import os

# Liste tous les fichiers .xlsx du dossier
for file in os.listdir('.'):
    if file.endswith(".xlsx"):
        # Lecture du fichier Excel
        df = pd.read_excel(file)
        
        # Création du nouveau nom (remplacement de l'extension)
        new_name = file.replace('.xlsx', '.csv')
        
        # Export en vrai CSV (séparateur point-virgule souvent utilisé en France)
        df.to_csv(new_name, index=False, encoding='utf-8-sig', sep=';')
        
        print(f"Converti : {file} -> {new_name}")
