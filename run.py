import os

os.system('rm -f -R cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/')
os.system('mkdir cpusim/tests/data/test')
os.system('mkdir cpusim/tests/data/test/json')

# os.system('mkdir cpusim/tests/data/train')
# os.system('mkdir cpusim/tests/data/valid')

# os.system('mkdir cpusim/tests/data/train/json')
# os.system('mkdir cpusim/tests/data/valid/json')

os.system('python cpusim/cpusim_generate_basic_block.py cpusim/tests/data/test/ 10')
# os.system('python cpusim/cpusim_generator.py cpusim/tests/data/train/ 2000')
# os.system('python cpusim/cpusim_generator.py cpusim/tests/data/valid/ 200')

os.system('python cpusim/cpusim.py cpusim/tests/data/test/')
# os.system('python cpusim/cpusim.py cpusim/tests/data/train/')
# os.system('python cpusim/cpusim.py cpusim/tests/data/valid/')

os.system('python cpusim/create_json.py cpusim/tests/data/test/')
# os.system('python cpusim/create_json.py cpusim/tests/data/train/')
# os.system('python cpusim/create_json.py cpusim/tests/data/valid/')