list_files_full = ['FL4clean.txt', 'FL7clean.txt', 'FLDclean.txt', 'FLHclean.txt', 'G3Uclean.txt', 'H5Gclean.txt', 'HDJclean.txt', 'HE8clean.txt', 'HEAclean.txt', 'HECclean.txt', 'HESclean.txt', 'HF3clean.txt', 'HV1clean.txt', 'HV5clean.txt', 'HV6clean.txt', 'J3Tclean.txt', 'J8Dclean.txt', 'J9Dclean.txt', 'JJ9clean.txt', 'JJAclean.txt', 'JJGclean.txt', 'JN7clean.txt', 'K69clean.txt', 'KBD_064002clean.txt', 'KBE_009101clean.txt', 'KBK_002334clean.txt', 'KBL_040501clean.txt', 'KC6_055705clean.txt', 'KC6_055905clean.txt', 'KCD_005214clean.txt', 'KCE_029701clean.txt', 'KD1_034501clean.txt', 'KD2_061610clean.txt', 'KDA_011605clean.txt', 'KDB_014817clean.txt', 'KDE_007113clean.txt', 'KDM_030302clean.txt', 'KE3_011211clean.txt', 'KE4_067607clean.txt', 'KGHclean.txt', 'KJSclean.txt', 'KLVclean.txt', 'KNV_137903clean.txt', 'KPW_133301clean.txt', 'KPY_136604clean.txt', 'KRF_fixed_1clean.txt', 'KRF_fixed_3clean.txt', 'KRG_fixed_13clean.txt', 'KRG_fixed_9clean.txt', 'KRH_fixed_16clean.txt', 'KRH_fixed_23clean.txt', 'KRH_fixed_25clean.txt', 'KRH_fixed_27clean.txt', 'KRH_fixed_34clean.txt', 'KRH_fixed_37clean.txt', 'KRL_fixed_3clean.txt', 'KRL_fixed_4clean.txt', 'KS0clean.txt', 'KS7_fixed_4clean.txt', 'KSN_135904clean.txt']
list_files_full_true = []
for str in list_files_full:
    str = str.replace("clean", "_clean")
    list_files_full_true.append(str)

def check_lengh_utterances(files): 
#Function that return 1 dict with the lengh (nb of character) for each utterance per speakers.
#The result is accurate, verify by hand (not for all the file)
#Take also in the final count for all the speaker the paranthesis that indicated something else, for example "(recorded jingle)"

    list_speakers = []
    list_len_utterance = []
    dict_speakers_len = {}
    with open(files) as f:
        file = f.readlines()
        for ligne in file:
            list_len_utterance.append(len(ligne))
            ligne = ligne.split()
            list_speakers.append(ligne[1])
        list_len_utterance.pop(0)
        list_speakers.pop(0)

        for i in range (0, len(list_speakers)):
            if list_speakers[i] in dict_speakers_len:
                dict_speakers_len[list_speakers[i]] = dict_speakers_len[list_speakers[i]] + list_len_utterance[i]
            else:
                dict_speakers_len[list_speakers[i]] = list_len_utterance[i]
    return(dict_speakers_len)

    

def write_result_csv(files_path):
    files_path.replace(".txt", ".csv")
    with open(files_path, "a") as f:
        f.write(check_lengh_utterances(files_path))
    print("file done succesfully !")


print(check_lengh_utterances("/Users/axel/Documents/Universite/Cours/Master_Traitement_Automatique_Des_langues/Master_1/Semester_8/Supervised_Project/Corpus/Sub_Corpus_For_Project/Divided_Corpus/Axel_Cleaned/KRH_fixed_27_clean.txt"))
