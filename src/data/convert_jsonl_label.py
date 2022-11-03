"""
Convert JSONL label file to CSV format.
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl_path")
    parser.add_argument("output_csv_path")
    args = parser.parse_args()

    input_jsonl_path: str = args.input_jsonl_path
    output_csv_path: str = args.output_csv_path

    labels = pd.read_json(
        input_jsonl_path,
        lines=True,
        dtype={"UID": str},  # type: ignore
    )
    print(f"Input: json-lines labels {input_jsonl_path}")
    print(f"Output: csv labels {output_csv_path}")
    print(f"Parsed {len(labels)} entries")

    if "true_label" in labels.keys():
        labels.to_csv(
            output_csv_path, columns=["true_label", "screen_name"], index=False
        )
    else:
        labels.to_csv(
            output_csv_path,
            columns=["classifier_label", "screen_name", "UID"],
            index=False,
        )


if __name__ == "__main__":
    main()
