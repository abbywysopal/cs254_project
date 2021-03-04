import json
import glob
import os
import sys

import json
import glob
import os
import sys

OPCODES = ['WRS', 'WR', 'LDI', 'ADDI', 'SUBI', 'MULI', "DIVI", 'STORE', 'LOAD', 'XOR', 'ADD', 'SUB', 'MUL', "DIV"]

if __name__ == '__main__':

    path = sys.argv[1]
    json_filenames = glob.glob(path + "/json/*")
    # print(json_filenames)

    for filename in json_filenames:
        # print(filename)
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            # print(data)
        
        tokenized = []
        for item in data:

            tokenized.append("<block>")
            instructions = item['instr']
            
            for inst in instructions:
                opcode = "" 
                src1 = ""
                src2 = ""
                dest = ""
                tokenized.append("<instr>")
                # print("INST", inst)

                tokenized.append("<opcode>")
                if(inst.find(" ") != -1):
                    opcode = inst[:inst.find(" ")]
                    # print("opc", opcode)
                    tokenized.append(opcode)
                    tokenized.append("</opcode>")


                    dest = inst[inst.find(" ") + 1:]
                    if(dest.find(" ") != -1):
                        src1 = dest[dest.find(" ") + 1 :]
                        dest = dest[:dest.find(" ")]
                        if(src1.find(" ") != -1):
                            src2 = src1[src1.find(" ") + 1:]
                            src1 = src1[:src1.find(" ")]
                            if(src2.find(" ") != -1):
                                src2 = src2[:src2.find(" ")]
                        
                    else:
                        src1 = dest
                        dest = ""

                    # print("dest", dest)
                    # print("src1", src1)
                    # print("src2", src2)

                    tokenized.append("<srcs>")
                    if(src1 != ""):
                        tokenized.append("<operand>")
                        tokenized.append(src1)
                        tokenized.append("</operand>")
                    
                    if(src2 != ""):
                        tokenized.append("<operand>")
                        tokenized.append(src2)
                        tokenized.append("</operand>")
                    tokenized.append("</srcs>")


                    tokenized.append("<dsts>")
                    if(dest != ""):
                        tokenized.append("<operand>")
                        tokenized.append(dest)
                        tokenized.append("</operand>")
                    tokenized.append("</dsts>")
                else:
                    opcode = inst
                    # print("opc", opcode)
                    tokenized.append(opcode)
                    tokenized.append("</opcode>")

                tokenized.append("</instr>")
            # for i in range(len(instructions)):

            tokenized.append("</block>")
            tokenized.append("<time>")
            tokenized.append(item['cycles'])
            tokenized.append("</time>")
            # print(tokenized)

            token = ""
            for tokens in tokenized:
                token += str(tokens)
            # print(token)
            item['xml'] = token

            with open(filename, 'w') as json_file:
                json.dump(item, json_file, indent=4)

  