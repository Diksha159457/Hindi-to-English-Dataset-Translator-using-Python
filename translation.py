import pandas as pd
import asyncio
from googletrans import Translator

# Read CSV file
data = pd.read_csv("/Users/dikshashahi/Downloads/hindi.csv")
print(data)

translator = Translator()
translations = {}

async def translate_data():
    for column in data.columns:
        unique = data[column].unique()

        for element in unique:
            result = await translator.translate(element)
            translations[element] = result.text

    for i in translations.items():
        print(i)

    data.replace(translations, inplace=True)
    print(data)

# run async function
asyncio.run(translate_data())
