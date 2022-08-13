from typing import List, Dict
import argparse

from .filter_label_file import get_user_id


def main():
    """
    Verify that a given list of UserIDs is included
    in the given label file. Write the selected user labels
    to the given output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("user_ids")
    parser.add_argument("user_labels")
    parser.add_argument("csv_output_path")

    args = parser.parse_args()
    user_ids_path: str = args.user_ids
    user_label_path: str = args.user_labels
    csv_output_path: str = args.csv_output_path

    with open(user_ids_path, "r") as user_ids_file:
        user_ids: List[str] = user_ids_file.readlines()[1:]

    with open(user_label_path, "r") as user_label_file:
        raw_label_lines: List[str] = user_label_file.readlines()

    user_labels: List[str] = raw_label_lines[1:]

    user_label_row_lookup: Dict[str, str] = {}
    for user_label_row in user_labels:
        user_id = get_user_id(user_label_row)
        user_label_row_lookup[user_id] = user_label_row

    selected_rows: List[str] = []
    user_ids_without_label = []
    for user_id_row in user_ids:
        user_id = get_user_id(user_id_row)
        user_label_row = user_label_row_lookup.get(user_id)

        if user_label_row:
            selected_rows.append(user_label_row)
        else:
            user_ids_without_label.append(user_id_row)

    print("Number of userIDs without label:", len(user_ids_without_label))
    print(user_ids_without_label)

    print(f"Writing {len(selected_rows)} rows of user labels to {csv_output_path}.")
    with open(csv_output_path, "w") as output_file:
        output_file.writelines([raw_label_lines[0]] + selected_rows)


if __name__ == "__main__":
    main()
