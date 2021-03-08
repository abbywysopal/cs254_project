import json
import glob
import os
import sys

import json
import glob
import os
import sys

OPCODES = dict(LDI='0110111', LOAD='0000011', ADDI='0010011', MULI='0010011', DIVI='0010011', 
    MUL='0110011', DIV='0110011', ADD='0110011', XOR='0110011', SUB='0110011', SUBI='0010011', 
    STORE='0100011', WR='0110111', WRS='0110111')


FUNCT_CODES = dict(WRS='000001', WR='000010', LDI='000011', ADDI='000100', SUBI='000101', MULI='000110',
            DIVI='000111', STORE='001000', LOAD='001001', XOR='001010', ADD='001011', SUB='001100', MUL='001101',
            DIV='001110')

FUNCT3 = dict(ADDI='000', SUBI='000', MULI='000', DIVI='100', STORE='010', LOAD= '011',XOR='100', ADD='000', SUB='000', MUL='000', DIV='100')
FUNCT7 = dict(XOR='0000000', ADD='0000000', SUB='0100000', MUL='0000001', DIV='0000001')

if __name__ == '__main__':

    path = sys.argv[1]
    json_filenames = glob.glob(path + "/json/*")
    # print(json_filenames)
    one_hot_map = []

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
                    # tokenized.append(int(OPCODES[opcode.upper()],2))
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
                        # if(src1.find("r") != -1):
                        #     src1 = src1[src1.find("r") + 1:]
                        tokenized.append(src1)
                        tokenized.append("</operand>")
                    
                    if(src2 != ""):
                        tokenized.append("<operand>")
                        # if(src2.find("r") != -1):
                        #     src2 = src1[src2.find("r") + 1:]
                        tokenized.append(src2)
                        tokenized.append("</operand>")
                    tokenized.append("</srcs>")


                    tokenized.append("<dsts>")
                    if(dest != ""):
                        tokenized.append("<operand>")
                        # if(dest.find("r") != -1):
                        #     dest = dest[dest.find("r") + 1:]
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
            # tokenized.append("<time>")
            # tokenized.append(item['cycles'])
            # tokenized.append("</time>")
            # print(tokenized)

            token = ""
            for tokens in tokenized:
                token += str(tokens)
            # print(token)
            item['xml'] = token

        token_to_hot_idx = {}
        hot_idx_to_token = {}
        hexmap = {}

        def hot_idxify(elem):
            # print(elem)
            if elem not in token_to_hot_idx:
                token_to_hot_idx[elem] = len(token_to_hot_idx)
                hot_idx_to_token[token_to_hot_idx[elem]] = elem
            # print(token_to_hot_idx[elem])
            return token_to_hot_idx[elem]


        map_item = list(map(hot_idxify, tokenized))
        item['nmap'] = map_item
        with open(filename, 'w') as json_file:
            json.dump(item, json_file, indent=4)

        one_hot_map.append(map_item)
    
    # print(one_hot_map)

'''

  Canonicalization -> token layer -> instruction layer

  token layer, which maps a given token to an embedding. 
  We implement the token layer by mapping each token in the sequence ß
  to an n-dimensional vector by learning a linear transformation of the 
  one-hot token vectors (this is equivalent to learning a lookup table).

    instruction layer. Because each instruction can have a variable number of 
    tokens depending on its number of source and destination operands, the size 
    of the input to the embedding stage is variable. We therefore implement the 
    instruction layer with a sequential Recurrent Neural Network (RNN) 
    architecture with Long Short Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997) cells.
  '''
