import json
import glob
import os
import sys

def remove_commas(instruction):
    final_instr = instruction.lower()
    comma_found =  final_instr.find(',') != -1

    while comma_found:
        final_instr = final_instr[:final_instr.find(',')] + " " + final_instr[final_instr.find(',') + 1 :]
        comma_found =  final_instr.find(',') != -1

    return final_instr

if __name__ == '__main__':

    path = sys.argv[1]
    assembly_filenames = glob.glob(path + "*.ass")
    output_filenames = glob.glob(path + "output*")

    data = 0
    for filename in assembly_filenames:
        output_filename = path + "output_assembly" + filename[filename.index("y") + 1 :filename.index(".")] + ".txt"
        if output_filename in output_filenames:
            file = open(filename)
            index = 0
            ml_data = dict()
            ml_data["instr"] = []

            for line in file.readlines():
                line = line[:-1]
                #remove ','
                # print(remove_commas(line))
                ml_data["instr"].append(remove_commas(line))
                # ml_data[index] = line
                index += 1

            while index < 32:
                ml_data["instr"].append("")
                index += 1

            output_file = open(output_filename)
            ml_data["cycles"] = int(output_file.readline())
            # print(ml_data["cycles"])
            list = []
            list.append(ml_data)
            
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".txt"
            with open(name, 'w') as outfile:
                outfile.write('[')
                json.dump(ml_data, outfile, indent=4)
                outfile.write(']')

            data += 1
        else:
            os.remove(filename)