import json
import glob
import os
import sys

if __name__ == '__main__':

    path = sys.argv[1]
    assembly_filenames = glob.glob(path + "*.ass")
    output_filenames = glob.glob(path + "output*")

    data = 0
    for filename in assembly_filenames:
        # print(filename)
        output_filename = path + "output_assembly" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        binary_filename = path + "binary" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        # print(output_filename)
        if output_filename in output_filenames:
            # print("FOUND")
            file = open(binary_filename)
            index = 0
            # print("LINES")
            ml_data = dict()
            for line in file.readlines():
                # print(str(index) + ": " + line)
                # dec = int(str(line), 16)
                string_num = line
                string_num = string_num[2:]
                ml_data[index] = float(int(string_num,2))
                index += 1

            while index < 32:
                ml_data[index] = 0
                index += 1

            output_file = open(output_filename)
            
            cycles = int(output_file.readline())
            cycles = format(cycles, '#016b')

            ml_data["cycles taken"] = cycles
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".json"
            # data_string = json.dumps(ml_data, indent=4)
            # print(data_string)
            with open(name, 'w') as outfile:
                json.dump(ml_data, outfile, indent=4)
            data += 1
        else:
            os.remove(filename)
