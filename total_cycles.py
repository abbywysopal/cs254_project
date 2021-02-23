import simpy
import random
import statistics
from hwcounter import Timer, count, count_end
from time import sleep
import glob
import _thread

def output_data():
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
        output_file = open(output_filename, 'w')
        output_file.write(str(cycles))

# # Set up the environment
# env = simpy.Environment()

# # Assume you've defined checkpoint_run() beforehand
# env.process(output_data())

# # Let's go!
# env.run(until=10)
output_data()

_thread.start_new_thread(output_data,())