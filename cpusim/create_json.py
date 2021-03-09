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
            ml_data["instr_cycle"] = []
            ml_data["total_cycles"] = 0

            for line in file.readlines():
                line = line[:-1]
                #remove ','
                # print(remove_commas(line))
                ml_data["instr"].append(remove_commas(line))
                # ml_data[index] = line
                index += 1

            # while index < 32:
            #     ml_data["instr"].append("")
            #     index += 1

            output_file = open(output_filename)
            last_cycle = 0
            index = 0
            for line in output_file.readlines():
                count = int(line)
                if last_cycle != 0:
                    count -= last_cycle
                ml_data["instr_cycle"].append(count)
                ml_data["total_cycles"] = int(line)
                last_cycle = int(line)
                index += 1

            while index <= 32:
                ml_data["instr_cycle"].append(0)
                index += 1
            
            
            ml_data["instr_cycle"].pop()

            list = []
            list.append(ml_data)
            
            name = path + "json/json_assembly" + filename[filename.index("y") + 1 : filename.index(".")] + ".txt"
            with open(name, 'w') as outfile:
                outfile.write('[')
                json.dump(ml_data, outfile, indent=4)
                outfile.write(']')

            data += 1

            os.remove(output_filename)
        os.remove(filename)