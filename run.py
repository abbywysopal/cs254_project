import os

os.system('rm -f -R tests/data/')
os.system('mkdir tests/data/')
os.system('mkdir tests/data/test')
os.system('mkdir tests/data/train')
os.system('mkdir tests/data/valid')
os.system('mkdir tests/data/test/json')
os.system('mkdir tests/data/train/json')
os.system('mkdir tests/data/valid/json')

os.system('python cpusim_generate_basic_block.py tests/data/test/ 10')
# os.system('python cpusim_generator.py tests/data/train/ 2000')
# os.system('python cpusim_generator.py tests/data/valid/ 200')

os.system('python cpusim.py tests/data/test/')
# os.system('python cpusim.py tests/data/train/')
# os.system('python cpusim.py tests/data/valid/')

# os.system('python create_inst_json.py tests/data/test/')
# os.system('python create_inst_json.py tests/data/train/')
# os.system('python create_inst_json.py tests/data/valid/')