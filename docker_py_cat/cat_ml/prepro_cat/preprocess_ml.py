import stanza
import pandas as pd
from tqdm import tqdm
import os 

stanza.download('es')
nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', batch_size=16)
#used the stanza library for spanish

print("Listing /app/datas contents:")
print(os.listdir("/app"))

def process_data(df, text_column_name, category_column_name):
    processed_data = []
    for _, row in tqdm(df.iterrows(), desc="Processing", total=len(df)):
        sentence = row[text_column_name].lower().strip()
        category = row[category_column_name]
        
        doc = nlp(sentence)
        keywords = []
        lemmas = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in ["PROPN", "NOUN", "ADJ"]:
                    keywords.append(word.text)
                    lemmas.append(word.lemma)

        joined_lemmas = " ".join(lemmas)
        processed_data.append({
            "Original": sentence, 
            "Keywords": keywords, 
            "Lemmas": joined_lemmas,
            "Category": category  
        })
    
    return pd.DataFrame(processed_data)

def read_and_combine(file_paths):
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

food_files = [
    'paths to multiple files'
]

health_files = [
    'paths to multiple files'
]

housing_files = [
    'paths to multiple files'
]

clothes_files = [
    'paths to multiple files'
]

EAC_files = [
    'paths to multiple files'
]


data_food_total = read_and_combine(food_files)
data_health_total = read_and_combine(health_files)
data_housing_total = read_and_combine(housing_files)
data_clothes_total = read_and_combine(clothes_files)
data_EAC_total = read_and_combine(EAC_files)

print("Processing Food")
processed_food = process_data(data_food_total, 'name', 'category')

print("Processing Health")
processed_health = process_data(data_health_total, 'name', 'category')

print("Processing Housing")
processed_housing = process_data(data_housing_total, 'name', 'category')

print("Processing Clothes")
processed_clothes = process_data(data_clothes_total, 'name', 'category')

print("Processing EAC")
processed_EAC = process_data(data_EAC_total, 'name', 'category')


processed_food.to_csv('/app/datos/processed_food.csv', index=False)
processed_health.to_csv('/app/datos/processed_health.csv', index=False)
processed_housing.to_csv('/app/datos/processed_housing.csv', index=False)
processed_clothes.to_csv('/app/datos/processed_clothes.csv', index=False)
processed_EAC.to_csv('/app/datos/processed_EAC.csv', index=False)

filename = 'paths'
n = 100  # every 100th line = 1% of the whole file
fields = ['name']
test_data = pd.read_csv(filename, header=0, skiprows=lambda i: i % n != 0, usecols=fields, sep=',', encoding='latin-1')

test_data['name'] = test_data['name'].astype(str)

nulls = test_data[test_data['name'].isnull()]

test_data["category"]=''

print(test_data.info())
print(test_data.head(2))

processed_test = process_data(test_data, 'name', 'category')

processed_test.to_csv('/app/datos/processed_test.csv', index=False)