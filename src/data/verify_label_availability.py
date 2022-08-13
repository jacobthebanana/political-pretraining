from typing import List
import argparse

from .filter_label_file import get_user_id


def main():
    """
    Verify that a given list of UserIDs is included
    in the given label file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("user_ids")
    parser.add_argument("user_labels")

    args = parser.parse_args()
    user_ids_path: str = args.user_ids
    user_label_path: str = args.user_labels

    with open(user_ids_path, "r") as user_ids_file:
        user_ids: List[str] = user_ids_file.readlines()[1:]

    with open(user_label_path, "r") as user_label_file:
        user_labels: List[str] = user_label_file.readlines()[1:]

    labelled_user_ids = set()
    for user_label_row in user_labels:
        user_id = get_user_id(user_label_row)
        labelled_user_ids.add(user_id)

    user_ids_without_label = []
    for user_id_row in user_ids:
        user_id = get_user_id(user_id_row)
        if user_id not in labelled_user_ids:
            user_ids_without_label.append(user_id_row)

    print("Number of userIDs without label:", len(user_ids_without_label))
    print(user_ids_without_label)


if __name__ == "__main__":
    main()
