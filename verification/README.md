# Verification script

The `verify_output.py` file compares two output files of the FEC project. This is a slightly modified version of the script used to assess the correctness of the projects for the course LEPL1503 for the FEC project.

## Usage

```bash
python3 verify_output.py <path to correct output file> <path to the output file to be verified>
```

Both files must be computed using the `-f` flag. You may use the [`fec.py`](../main.py) file to compute the __correct output file__.

## Expected output

If both files contain the same content (and assuming that the expected file is correct), the program should output the following message

```bash
{'correct': True, 'error_messages': []}
```

If the `correct` key is set to `False`, the tested file is incorrect.