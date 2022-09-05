from typing import List, Dict
import argparse
import json

from .filter_label_file import get_user_id


def select_rows_from_label_file(
    user_label_row_lookup: Dict[str, str], user_ids: List[str]
) -> List[str]:
    """
    Select label rows matching user_ids.

    Args:
     label_rows_lookup: mapping from user_id to rows of the user label file.
     user_ids: list of user_id to select.

    Returns:
     List[str]: list of selected label file rows.
    """
    selected_rows: List[str] = []
    user_ids_without_label = []
    for user_id in user_ids:
        user_label_row = user_label_row_lookup.get(user_id)

        if user_label_row:
            selected_rows.append(user_label_row)
        else:
            user_ids_without_label.append(user_id)

    print("Number of userIDs without label:", len(user_ids_without_label))
    print(user_ids_without_label)

    return selected_rows


def main():
    """
    Verify that a given list of UserIDs is included
    in the given label file. Write the selected user labels
    to the given output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("user_ids")
    parser.add_argument("user_labels")
    parser.add_argument("csv_output_prefix")
    parser.add_argument("csv_output_suffix")
    parser.add_argument("--use_folds", required=False, default=False)

    args = parser.parse_args()
    user_ids_path: str = args.user_ids
    user_label_path: str = args.user_labels
    csv_output_prefix: str = args.csv_output_prefix
    csv_output_suffix: str = args.csv_output_suffix
    use_folds: bool = args.use_folds

    user_id_folds: Dict[str, List[str]]

    with open(user_ids_path, "r") as user_ids_file:
        if use_folds:
            with open(user_ids_path, "r") as user_ids_json_file:
                user_id_folds = json.load(user_ids_json_file)
        else:
            user_id_rows: List[str] = user_ids_file.readlines()[1:]
            user_ids: List[str] = list(map(get_user_id, user_id_rows))
            user_id_folds = {"": user_ids}

    with open(user_label_path, "r") as user_label_file:
        raw_label_lines: List[str] = user_label_file.readlines()

    user_labels: List[str] = raw_label_lines[1:]

    user_label_row_lookup: Dict[str, str] = {}
    for user_label_row in user_labels:
        user_id = get_user_id(user_label_row)
        user_label_row_lookup[user_id] = user_label_row

    for fold_name, user_ids in user_id_folds.items():
        selected_rows = select_rows_from_label_file(user_label_row_lookup, user_ids)

        csv_output_path = csv_output_prefix + fold_name + csv_output_suffix
        print(f"Writing {len(selected_rows)} rows of user labels to {csv_output_path}.")
        with open(csv_output_path, "w") as output_file:
            output_file.writelines([raw_label_lines[0]] + selected_rows)


if __name__ == "__main__":
    main()
