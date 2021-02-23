from hwcounter import Timer, count, count_end
from time import sleep
import glob
files = glob.glob("data/binary*")

for filename in files:
    file = open(filename)
    # files = 
    # subprocess.run("binary1.txt")
    cycles = 0
    for line in file.readlines():
        # line = file.readline()
        # print(line)
        start = count()
        exec(line)
        elapsed = count_end() - start
        cycles += elapsed
    print(f'elapsed cycles: {cycles}')

    output_filename = filename[0 : filename.index("/") + 1] + "output_" + filename[filename.index("/") + 1 :]
    output_file = open(output_filename, 'x')
    output_file.write(str(cycles))