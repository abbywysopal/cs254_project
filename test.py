import os

for i in range(1,32):

    os.system(f'mkdir cpusim/tests/data/test{i}')
    os.system(f'mkdir cpusim/tests/data/test{i}/json')
    os.system(f'python cpusim_generate_basic_block.py cpusim/tests/data/test{i}/ 500 {i}')
    os.system(f'python cpusim/cpusim.py cpusim/tests/data/test{i}/')
    os.system(f'python create_json.py cpusim/tests/data/test{i}/')
    os.system(f'python cpusim/tokenize.py cpusim/tests/data/test{i}/')
