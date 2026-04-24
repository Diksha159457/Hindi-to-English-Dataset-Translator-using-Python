# Hindi to English Dataset Translator

A Python utility for translating Hindi text stored in CSV datasets into English. The script reads a source CSV, translates text columns with `googletrans`, and writes the translated dataset back to disk for downstream analysis or preprocessing.

## Highlights

- Reads Hindi text from CSV data
- Translates all text columns by default
- Supports selecting specific columns with a CLI flag
- Writes the translated dataset to a new CSV file

## Tech Stack

- Python
- Pandas
- googletrans

## Project Files

```text
Hindi-to-English-Dataset-Translator-using-Python/
├── hindi.csv
├── translation.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
git clone https://github.com/Diksha159457/Hindi-to-English-Dataset-Translator-using-Python.git
cd Hindi-to-English-Dataset-Translator-using-Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python translation.py --input hindi.csv --output translated_hindi.csv
```

Optional: translate only selected columns.

```bash
python translation.py --input hindi.csv --output translated_hindi.csv --columns sentence category
```

## Resume Value

This project demonstrates CSV processing, CLI design, lightweight NLP tooling, and preprocessing workflows for multilingual datasets.

## Future Improvements

- Add batch progress reporting for large datasets
- Add retry logic for translator failures
- Support other source and destination languages from the command line

## License

MIT. See [LICENSE](LICENSE).
