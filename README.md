Hindi to English Dataset Translator
This project translates Hindi text present in a CSV dataset into English automatically using Python. The program reads the dataset, translates Hindi words or sentences into English, and saves the translated dataset into a new CSV file.
The project demonstrates basic data processing and language translation using Python.
Technologies Used
Python
pandas – for reading and processing CSV data
googletrans – for translating Hindi text into English

project structure:
translation-project
│
├── data
│   └── hindi.csv
│
├── output
│   └── output.csv
│
├── src
│   └── translator.py
│
├── requirements.txt
└── README.md
Features
Reads Hindi text from a CSV dataset
Automatically translates Hindi content into English
Replaces original Hindi values with translated English values
Saves the translated dataset as a new CSV file
How It Works
The program loads the CSV dataset using pandas.
It scans each column and finds unique Hindi text values.
Each value is sent to the translation API.
The translated English text replaces the original Hindi text.
The updated dataset is saved as a new CSV file.
