list_files = ["HEA_clean.txt", "HEC_clean.txt", "HV6_clean.txt", "J8D_clean.txt", "JJG_clean.txt", "KBL_040501_clean.txt", 
              "KD2_061610_clean.txt", "KDB_014817_clean.txt", "KE4_067607_clean.txt", "KGH_clean.txt", "KNV_137903_clean.txt",
              "KPW_133301_clean.txt", "KRF_fixed_3_clean.txt", "KRH_fixed_25_clean.txt", "KRH_fixed_27_clean.txt"]

List_Files_Jinane = ["G3U.xml", "J3T.xml","JJ9.xml","JJA.xml","KBE_009101.xml","KCD_005214.xml","KDE_007113.xml","KDM_030302.xml",
                     "KE3_011211.xml","KJS.xml","KPY_136604.xml","KRH_fixed_37.xml","KRL_fixed_4.xml","KS7_fixed_4.xml","KSN_135904.xml"]

list_files_jinane_clean = []
for files in List_Files_Jinane:
    files_clean = files.replace(".xml", "_clean.txt")
    list_files_jinane_clean.append(files_clean)

list_files_Hao = ["FL4_clean.txt", "FL7_clean.txt", "FLH_clean.txt", "H5G_clean.txt",
                  "HES_clean.txt", "HV1_clean.txt", "JN7_clean.txt", "KBD_064002_clean.txt",
                  "KBK_002334_clean.txt", "KC6_055705_clean.txt", "KC6_055905_clean.txt",
                  "KCE_029701_clean.txt", "KD1_034501_clean.txt", "KLV_clean.txt", "KRL_fixed_3_clean.txt"]


def count_utterance_Everybody(file_name):
    list_from_str = ""
    dict_utterance = {}
    with open(file_name) as f:
        file = f.readlines()
        for item in file:
            list_from_str = item.split(" ")
            if list_from_str[1] in dict_utterance:
                dict_utterance[list_from_str[1]] += 1
            else:
                dict_utterance[list_from_str[1]] = 1
    return(dict_utterance)


def count_utterance_Hao(file_name):
    list_from_str = ""
    dict_utterance = {}
    with open(file_name) as f:
        file = f.readlines()
        file.pop(0)
        file.pop(1)
        for item in file:
            list_from_str = item.split(" ")
            if list_from_str[1] in dict_utterance:
                dict_utterance[list_from_str[1]] += 1
            else:
                dict_utterance[list_from_str[1]] = 1
    return(dict_utterance)


for files in list_files_Hao:
    with open("Nb_utterance_Corpus_Hao.txt", "a") as f:
        f.write(files)
        f.write(":")
        f.write(str(count_utterance_Hao(files)))
        f.write("\n")
        f.write("\n")
