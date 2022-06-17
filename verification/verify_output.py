import sys
from typing import List, Dict, Tuple, Union


def read_all_data(filename_total, endian="big"):
    with open(filename_total, "rb") as fd:
        data = fd.read()

    files = dict()

    while len(data) > 0:
        # First four bytes are the filename
        filename_size = int.from_bytes(data[:4], endian)
        content_size = int.from_bytes(data[4:12], endian)

        filename = data[12:12 + filename_size]
        content = data[12:12 + filename_size: 12 +
                       filename_size + content_size]

        files[filename] = content
        data = data[12 + filename_size + content_size:]

    return files


def compare_files(reference: str, candidate: str) -> Tuple[Dict[str, int], List[str]]:
    """
    :param reference: the reference filename
    :param candidate: the candidate filename
    :return: (error_statistics, error_messages)
    """
    errors: Dict[str, int] = {}
    error_messages: List[str] = []

    try:
        ref_data = read_all_data(reference)
    except Exception as e:
        assert False, f"Cannot parse the reference file {reference}: {e}"

    try:
        cdt_data = read_all_data(candidate)
    except Exception as e:
        return {"fatal_out_parsing": 1}, [str(e)]

    # Verify the content size
    if len(ref_data) != len(cdt_data):
        errors["invalid_length"] = 1
        error_messages.append(
            f"The output does not contain the same number of files as expected ({len(ref_data)} vs {len(cdt_data)})")
        return errors, error_messages  # Useless to continue after that

    for filename, filecontent in ref_data.items():
        if filename not in cdt_data.keys():
            errors["not_found_file"] = errors.get("not_found_file", 0) + 1
            error_messages.append(
                f"Could not find the following file in the output: {filename}")
            continue
        cdt_filecontent = cdt_data[filename]
        if filecontent != cdt_filecontent:
            errors["not_same_output_file"] = errors.get(
                "not_same_output_file", 0) + 1
            error_messages.append(
                f"Not the expected output for the file: {filename}")

    return errors, error_messages


def check(reference_filename: str, candidate_filename: str) -> Dict[str, Union[int, List[str]]]:
    print("COMPARE", reference_filename, candidate_filename)
    try:
        status, error_messages = compare_files(
            reference_filename, candidate_filename)
    except Exception as e:
        status = {"misc": str(e)}
        error_messages = [f"{type(e)}: {str(e)}"]
    status["correct"] = False if len(status) > 0 else True
    status["error_messages"] = error_messages
    return status


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: verify_output.py <submission_file> <expected_file>",
              file=sys.stderr)
        exit(1)

    submission_file = sys.argv[1]
    expected_file = sys.argv[2]
    print(check(expected_file, submission_file))
