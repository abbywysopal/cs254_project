import os

os.system('rm -f -R cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/test')
os.system('mkdir cpusim/tests/data/test/json')
os.system('python cpusim_generate_basic_block.py cpusim/tests/data/test/ 5000')
os.system('python cpusim/cpusim.py cpusim/tests/data/test/')
os.system('python create_json.py cpusim/tests/data/test/')
os.system('python tokenize.py cpusim/tests/data/test/')