# lepl1503-2022-pyfec
Python implementation of the 2022 project of LEPL1503

## Make input

The instance files have been generated with the following commands:

```bash
python3 make_input.py input_txt/small.txt input_binary/small.bin -b 3 -w 3 -r 4 -s 42 -n --loss-pattern 3 -v

python3 make_input.py input_txt/medium.txt input_binary/medium.bin -b 10 -w 20 -r 2 -s 12345 -n

python3 make_input.py input_txt/big.txt input_binary/big.bin -b 50 -w 1000 -r 20 -s 1 -n

python3 make_input.py input_txt/africa.txt input_binary/africa.bin -b 50 -w 100 -r 20 -s 1 -n
```

## Usage

You may run the program with

```bash
python3 main.py input_binary/ -f output.txt
```