import os

os.system('rm -f -R cpusim/tests/data')
os.system('mkdir cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/train')
os.system('mkdir cpusim/tests/data/train/json')
os.system('python cpusim_generate_basic_block.py cpusim/tests/data/train/ 4000 0')
os.system('python cpusim/cpusim.py cpusim/tests/data/train/')
os.system('python create_json.py cpusim/tests/data/train/')
os.system('python cpusim/tokenize.py cpusim/tests/data/train/')