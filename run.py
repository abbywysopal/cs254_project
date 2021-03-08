import os

os.system('rm -f -R cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/test')
os.system('mkdir cpusim/tests/data/test/json')
os.system('python cpusim/cpusim_generate_basic_block.py cpusim/tests/data/test/ 50000')
os.system('python cpusim/cpusim.py cpusim/tests/data/test/')
os.system('python cpusim/create_json.py cpusim/tests/data/test/')
os.system('python cpusim/tokenize.py cpusim/tests/data/test/')