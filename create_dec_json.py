import json
import glob
import os
import sys

if __name__ == '__main__':
    MAX_VAL = 4294967295

    path = sys.argv[1]
    assembly_filenames = glob.glob(path + "*.ass")
    output_filenames = glob.glob(path + "output*")

    data = 0
    for filename in assembly_filenames:
        # print(filename)
        output_filename = path + "output_assembly" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        hex_filename = path + "hex" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        # print(output_filename)
        if output_filename in output_filenames:
            # print("FOUND")
            file = open(hex_filename)
            index = 0
            # print("LINES")
            ml_data = dict()
            for line in file.readlines():
                # print(str(index) + ": " + line)
                dec = int(str(line), 16)
                ml_data[index] = float(dec)
                index += 1

            while index < 32:
                ml_data[index] = 0
                index += 1

            output_file = open(output_filename)
            ml_data["cycles taken"] = int(output_file.readline())
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".json"
            # data_string = json.dumps(ml_data, indent=4)
            # print(data_string)
            with open(name, 'w') as outfile:
                json.dump(ml_data, outfile, indent=4)
            data += 1
        else:
            os.remove(filename)
