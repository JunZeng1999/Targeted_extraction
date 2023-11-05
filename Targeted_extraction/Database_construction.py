from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors


# Standardize the SMILES format of molecules
# csv_data1 = pd.read_csv('your path/BioTransformer(superbio)/C1/bio_1.csv')
# SMILES1 = csv_data1['SMILES'].tolist()
# SMILES1 = np.array(SMILES1)
# SMI1 = []
# location1 = []
# for i in range(len(SMILES1)):
#     mol1 = Chem.MolFromSmiles(SMILES1[i])
#     if mol1 is not None:
#         smi1 = Chem.MolToSmiles(mol1)
#     else:
#         smi1 = 0
#         location1.append(i)
#     SMI1.append(smi1)
# csv_data1['Unified_smiles'] = SMI1
# csv_data1 = csv_data1.drop(location1)
# csv_data1.to_csv("your path/C1/BIO_1.csv", index=False, encoding="utf-8")

# csv_data2 = pd.read_csv('your path/CTS/C1/cts_1.csv')
# SMILES2 = csv_data2['smiles'].tolist()
# SMILES2 = np.array(SMILES2)
# SMI2 = []
# location2 = []
# for j in range(len(SMILES2)):
#     mol2 = Chem.MolFromSmiles(SMILES2[j])
#     if mol2 is not None:
#         smi2 = Chem.MolToSmiles(mol2)
#     else:
#         smi2 = 0
#         location2.append(j)
#     SMI2.append(smi2)
# csv_data2['Unified_smiles'] = SMI2
# csv_data2 = csv_data2.drop(location2)
# csv_data2.to_csv("your path/C1/CTS_1.csv", index=False, encoding="utf-8")


# Integrate the predicted results from three tools for one component
# csv_CTS = pd.read_csv('your path/C1/CTS_1.csv')
# ID_mean1_1 = []
# ID_mean1 = csv_CTS['genKey'].tolist()
# smiles_mean1 = csv_CTS['Unified_smiles'].tolist()
# formula1 = csv_CTS['formula'].tolist()
# for i in range(len(ID_mean1)):
#     a = 'CTS1_' + ID_mean1[i]
#     ID_mean1_1.append(a)
#
# csv_BIO = pd.read_csv('your path/C1/BIO_1.csv')
# ID_mean2_2 = []
# ID_mean2 = csv_BIO['Result_ID'].tolist()
# smiles_mean2 = csv_BIO['Unified_smiles'].tolist()
# formula2 = csv_BIO['Molecular_formula'].tolist()
# for i in range(len(ID_mean2)):
#     s2 = str(ID_mean2[i])
#     a = 'bio1_' + s2
#     ID_mean2_2.append(a)
#
# csv_sygma = pd.read_csv('your path/C1/sygma1.csv')
# ID_mean3_3 = []
# ID_mean3 = csv_sygma['sygma_ID'].tolist()
# smiles_mean3 = csv_sygma['SyGMa_metabolite'].tolist()
# formula3 = []
# for i in range(len(ID_mean3)):
#     s3 = str(ID_mean3[i])
#     a = 'sygma14_' + s3
#     ID_mean3_3.append(a)
#     formula3.append('undetermined')
#
# ID_mean = ID_mean1_1 + ID_mean2_2 + ID_mean3_3
# smiles_mean = smiles_mean1 + smiles_mean2 + smiles_mean3
# formula_mean = formula1 + formula2 + formula3
#
# df = pd.DataFrame()
# df['ID'] = ID_mean
# df['SMILES'] = smiles_mean
# df['formula'] = formula_mean
# df.to_csv("your path/C1/C1.csv")


# Integrate all the predicted results of metabolites for each component into one list
# Calculate molecular weight
# Convert standardized SMILES to InChIKeys
# csv1 = pd.read_csv('your path/C1/C1.csv')
# ID_1 = csv1['ID'].tolist()
# smiles_1 = csv1['SMILES'].tolist()
# formula_1 = csv1['formula'].tolist()
#
# csv2 = pd.read_csv('your path/C2/C2.csv')
# ID_2 = csv2['ID'].tolist()
# smiles_2 = csv2['SMILES'].tolist()
# formula_2 = csv2['formula'].tolist()
#
# ID_mean = ID_1 + ID_2
# smiles_mean = smiles_1 + smiles_2
# formula_mean = formula_1 + formula_2
#
# location = []
# InchiKey = []
# Short_InChIKey = []
# aa = 0
# for k in range(len(smiles_mean)):
#     b = 0
#     mol = Chem.MolFromSmiles(smiles_mean[k])
#     location.append(Descriptors.ExactMolWt(mol))
#     a = Chem.MolToInchiKey(mol)
#     InchiKey.append(a)
#     for i in range(len(a)):
#         aa = i
#         if a[i] == '-':
#             b = b + 1
#             if b == 1:
#                 Short_InChIKey.append(a[0:aa])
# df = pd.DataFrame()
# df['ID'] = ID_mean
# df['SMILES'] = smiles_mean
# df['formula'] = formula_mean
# df['m/z'] = location
# df['InchIKey'] = InchiKey
# df['Short InChIKey'] = Short_InChIKey
# df.to_csv("your path/C1_C2.csv")


# Build a database based on the integrated results mentioned above, which can be used for MS-FINDER
# csv = pd.read_csv('your path/C1_C2.csv')
# ID_1 = csv['ID'].tolist()
# smiles_1 = csv['SMILES'].tolist()
# formula_1 = csv['formula'].tolist()
# InchIKey_1 = csv['InchIKey'].tolist()
# Short_InChIKey_1 = csv['Short InChIKey'].tolist()
# mz = csv['m/z'].tolist()
# mz = np.array(mz)
# mz_sort = mz.argsort()
# list1 = []
# list2 = []
# list3 = []
# list4 = []
# list5 = []
# list6 = []
# list7 = []
# list8 = []
# a = 0
# for i in range(len(mz_sort)):
#     list1.append(ID_1[mz_sort[i]])
#     list2.append(InchIKey_1[mz_sort[i]])
#     list3.append(Short_InChIKey_1[mz_sort[i]])
#     list4.append('-')
#     list5.append(mz[mz_sort[i]])
#     list6.append(formula_1[mz_sort[i]])
#     list7.append(smiles_1[mz_sort[i]])
#     list8.append('-')
# df = pd.DataFrame()
# df['Title'] = list1
# df['InChIKey'] = list2
# df['Short InChIKey'] = list3
# df['PubChem CID'] = list4
# df['Exact mass'] = list5
# df['Formula'] = list6
# df['SMILES'] = list7
# df['Database ID'] = list8
# df.to_csv("your path/C1_C2_database1.csv")


# Remove duplicate entries from the database
# csv = pd.read_csv('your path/C1_C2_database1.csv')
# smiles = csv['SMILES'].tolist()
# smiles = np.array(smiles)
# Title_1 = csv['Title'].tolist()
# PubChem_CID_1 = csv['PubChem CID'].tolist()
# Database_ID_1 = csv['Database ID'].tolist()
# Exact_mass_1 = csv['Exact mass'].tolist()
# formula_1 = csv['Formula'].tolist()
# InchIKey_1 = csv['InChIKey'].tolist()
# Short_InChIKey_1 = csv['Short InChIKey'].tolist()
# list_zong = []
# list1 = []
# list2 = []
# list3 = []
# list4 = []
# list5 = []
# list6 = []
# list7 = []
# list8 = []
# location = []
# for i in range(len(smiles)):
#     if smiles[i] not in list_zong:
#         list_zong.append(smiles[i])
#         list1.append(Title_1[i])
#         list2.append(InchIKey_1[i])
#         list3.append(Short_InChIKey_1[i])
#         list4.append(PubChem_CID_1[i])
#         list5.append(Exact_mass_1[i])
#         list6.append(formula_1[i])
#         list7.append(smiles[i])
#         list8.append(Database_ID_1[i])
#     else:
#         location.append(i)
# for j in range(len(location)):
#     value = list7.index(smiles[location[j]])
#     list1[value] = str(list1[value]) + ';' + Title_1[location[j]]
# df = pd.DataFrame()
# df['Title'] = list1
# df['InChIKey'] = list2
# df['Short InChIKey'] = list3
# df['PubChem CID'] = list4
# df['Exact mass'] = list5
# df['Formula'] = list6
# df['SMILES'] = list7
# df['Database ID'] = list8
# df.to_csv("your path/C1_C2_database2.csv")


# Identify the molecular formula of predicted metabolites
# Input SMILES into CTS
# csv1 = pd.read_csv('your path/C1_C2_database1.csv')
# Formula_2 = csv1['Formula'].tolist()
# Exact_mass_2 = csv1['Exact mass'].tolist()
# Exact_mass_2 = np.array(Exact_mass_2)
# csv = pd.read_csv('your path/C1_C2_database2.csv')
# smiles = csv['SMILES'].tolist()
# Title_1 = csv['Title'].tolist()
# PubChem_CID_1 = csv['PubChem CID'].tolist()
# Database_ID_1 = csv['Database ID'].tolist()
# Exact_mass_1 = csv['Exact mass'].tolist()
# Exact_mass_1 = np.array(Exact_mass_1)
# formula_1 = csv['Formula'].tolist()
# InchIKey_1 = csv['InChIKey'].tolist()
# Short_InChIKey_1 = csv['Short InChIKey'].tolist()
# for i in range(len(Exact_mass_1)):
#     if formula_1[i] == 'undetermined':
#         for j in range(len(Exact_mass_2)):
#             if Exact_mass_2[j] == Exact_mass_1[i] and Formula_2[j] != 'undetermined':
#                 formula_1[i] = Formula_2[j]
# df = pd.DataFrame()
# df['Title'] = Title_1
# df['InChIKey'] = InchIKey_1
# df['Short InChIKey'] = Short_InChIKey_1
# df['PubChem CID'] = PubChem_CID_1
# df['Exact mass'] = Exact_mass_1
# df['Formula'] = formula_1
# df['SMILES'] = smiles
# df['Database ID'] = Database_ID_1
# df.to_csv("your path/C1_C2_database3.csv")


# Establish a unified SMILES format for metabolites reported in the literature
# csv_data1 = pd.read_csv('your path/literature_database1.csv')
# SMILES1 = csv_data1['SMILES'].tolist()
# SMILES1 = np.array(SMILES1)
# SMI1 = []
# location1 = []
# for i in range(len(SMILES1)):
#     mol1 = Chem.MolFromSmiles(SMILES1[i])
#     if mol1 is not None:
#         smi1 = Chem.MolToSmiles(mol1)
#     else:
#         smi1 = 0
#         location1.append(i)
#     SMI1.append(smi1)
# csv_data1['Unified_smiles'] = SMI1
# csv_data1 = csv_data1.drop(location1)
# csv_data1.to_csv("your path/literature_database2.csv", index=False, encoding="utf-8")


# Add molecular weight and other information to the metabolite data obtained through literature retrieval
# csv = pd.read_csv('your path/literature_database2.csv')
# smiles = csv['Unified_smiles'].tolist()
# smiles = np.array(smiles)
# Title_1 = csv['Title'].tolist()
# PubChem_CID_1 = csv['PubChem CID'].tolist()
# Database_ID_1 = csv['Database ID'].tolist()
# formula_1 = csv['Formula'].tolist()
# list_zong = []
# list1 = []
# list2 = []
# list3 = []
# list4 = []
# list5 = []
# list6 = []
# list7 = []
# list8 = []
# location = []
# for i in range(len(smiles)):
#     mol = Chem.MolFromSmiles(smiles[i])
#     list5.append(Descriptors.ExactMolWt(mol))
#     list2.append(Chem.MolToInchiKey(mol))
#     list_zong.append(smiles[i])
#     list1.append(Title_1[i])
#     aa = 0
#     b = 0
#     a = Chem.MolToInchiKey(mol)
#     for j in range(len(a)):
#         aa = j
#         if a[j] == '-':
#             b = b + 1
#             if b == 1:
#                 list3.append(a[0:aa])
#     list4.append(PubChem_CID_1[i])
#     list6.append(formula_1[i])
#     list7.append(smiles[i])
#     list8.append(Database_ID_1[i])
# df = pd.DataFrame()
# df['Title'] = list1
# df['InChIKey'] = list2
# df['Short InChIKey'] = list3
# df['PubChem CID'] = list4
# df['Exact mass'] = list5
# df['Formula'] = list6
# df['SMILES'] = list7
# df['Database ID'] = list8
# df.to_csv("your path/literature_database3.csv")


# Integrate literature data with predicted metabolite data, 
# removing duplicate entries
# csv = pd.read_csv('your path/C1_C2_literature_database1.csv')
# smiles = csv['SMILES'].tolist()
# smiles = np.array(smiles)
# Title_1 = csv['Title'].tolist()
# PubChem_CID_1 = csv['PubChem CID'].tolist()
# Database_ID_1 = csv['Database ID'].tolist()
# Exact_mass_1 = csv['Exact mass'].tolist()
# formula_1 = csv['Formula'].tolist()
# InchIKey_1 = csv['InChIKey'].tolist()
# Short_InChIKey_1 = csv['Short InChIKey'].tolist()
# list_zong = []
# list1 = []
# list2 = []
# list3 = []
# list4 = []
# list5 = []
# list6 = []
# list7 = []
# list8 = []
# location = []
# for i in range(len(smiles)):
#     if smiles[i] not in list_zong:
#         list_zong.append(smiles[i])
#         list1.append(Title_1[i])
#         list2.append(InchIKey_1[i])
#         list3.append(Short_InChIKey_1[i])
#         list4.append(PubChem_CID_1[i])
#         list5.append(Exact_mass_1[i])
#         list6.append(formula_1[i])
#         list7.append(smiles[i])
#         list8.append(Database_ID_1[i])
#     else:
#         location.append(i)
# for j in range(len(location)):
#     value = list7.index(smiles[location[j]])
#     list1[value] = str(list1[value]) + ';' + Title_1[location[j]]
# df = pd.DataFrame()
# df['Title'] = list1
# df['InChIKey'] = list2
# df['Short InChIKey'] = list3
# df['PubChem CID'] = list4
# df['Exact mass'] = list5
# df['Formula'] = list6
# df['SMILES'] = list7
# df['Database ID'] = list8
# df.to_csv("your path/C1_C2_literature_database2.csv")


# Retrieve m/z information from the database in order to run “Targeted_extraction_main.py”
# csv1 = pd.read_csv('your path/C1_C2_literature_database2.csv')
# Formula_1 = csv1['Formula'].tolist()
# Exact_mass_1 = csv1['Exact mass'].tolist()
# Exact_mass_1 = np.array(Exact_mass_1)
# list_zong = []
# Formula_zong = []
# list_zong1 = []
# Formula_zong1 = []
# for i in range(len(Formula_1)):
#     if Formula_1[i] not in Formula_zong:
#         list_zong.append(Exact_mass_1[i])
#         Formula_zong.append(Formula_1[i])
# df1 = pd.DataFrame()
# df1['Exact mass'] = list_zong
# df1.to_csv("your path/mz.csv")
