import argparse
from pathlib import Path

import pandas as pd
from googletrans import Translator


def translate_value(translator: Translator, value: str, cache: dict[str, str]) -> str:
    if value in cache:
        return cache[value]

    translated = translator.translate(value, src="hi", dest="en").text
    cache[value] = translated
    return translated


def translate_dataframe(dataframe: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    translator = Translator()
    cache: dict[str, str] = {}
    target_columns = columns or [name for name in dataframe.columns if dataframe[name].dtype == "object"]

    for column in target_columns:
        dataframe[column] = dataframe[column].apply(
            lambda value: value
            if pd.isna(value) or not str(value).strip()
            else translate_value(translator, str(value), cache)
        )

    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Hindi CSV content into English.")
    parser.add_argument(
        "--input",
        default="hindi.csv",
        help="Path to the input CSV file. Defaults to hindi.csv in the project root.",
    )
    parser.add_argument(
        "--output",
        default="translated_hindi.csv",
        help="Path to write the translated CSV output.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        help="Optional list of column names to translate. Defaults to all text columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    dataframe = pd.read_csv(input_path)
    translated = translate_dataframe(dataframe, args.columns)
    translated.to_csv(output_path, index=False)

    print(f"Translated dataset saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
