from typing import Dict, List
import argparse
import pickle
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_uid_pkl")
    parser.add_argument("output_uid_json")

    args = parser.parse_args()
    test_uid_pkl_path: str = args.test_uid_pkl
    output_uid_json_path: str = args.output_uid_json

    with open(test_uid_pkl_path, "br") as test_uid_pkl_file:
        test_uids: Dict[int, List[str]] = pickle.load(test_uid_pkl_file)

    with open(output_uid_json_path, "w") as output_uid_json_file:
        json.dump(test_uids, output_uid_json_file, indent=2)


if __name__ == "__main__":
    main()
