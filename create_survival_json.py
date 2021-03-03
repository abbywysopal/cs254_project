import json
import glob
import os
import sys

if __name__ == '__main__':

    path = sys.argv[1]
    assembly_filenames = glob.glob(path + "*.ass")
    output_filenames = glob.glob(path + "output*")

    data = 0
    # assembly = []
    for filename in assembly_filenames:
        # print(filename)
        output_filename = path + "output_assembly" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        # print(output_filename)
        if output_filename in output_filenames:
            file = open(filename)
            index = 0

            ml_data = dict()
            assembly = ""
            for line in file.readlines():
                assembly += line
                index += 1

            ml_data["inst"] = assembly
            output_file = open(output_filename)
            ml_data["duration"] = int(output_file.readline())
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".json"
            # data_string = json.dumps(ml_data, indent=4)
            # print(data_string)
            with open(name, 'w') as outfile:
                json.dump(ml_data, outfile, indent=4)
            data += 1
        else:
            os.remove(filename)
