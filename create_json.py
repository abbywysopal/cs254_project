import json
import glob
import os
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    assembly_filenames = glob.glob(path + "*.ass")
    output_filenames = glob.glob(path + "output*")
    ml_data = dict()

    data = 0
    for filename in assembly_filenames:
        output_filename = path + "output_assembly" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        if output_filename in output_filenames:
            file = open(filename)
            index = 0
            for line in file.readlines():
                ml_data[index] = line
                index += 1

            output_file = open(output_filename)
            ml_data["cycles taken"] = int(output_file.readline())
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".txt"
            with open(name, 'w') as outfile:
                json.dump(ml_data, outfile, indent=4)
            data += 1
        else:
            os.remove(filename)

    print(data)