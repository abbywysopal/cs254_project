import os

os.system('python cpusim_generator.py tests/data/test/ 200')
os.system('python cpusim_generator.py tests/data/train/ 2000')
os.system('python cpusim_generator.py tests/data/valid/ 2000')

os.system('python cpusim.py tests/data/test/')
os.system('python cpusim.py tests/data/train/')
os.system('python cpusim.py tests/data/valid/')

os.system('python create_json.py tests/data/test/')
os.system('python create_json.py tests/data/train/')
os.system('python create_json.py tests/data/valid/')