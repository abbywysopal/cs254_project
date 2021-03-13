import json
import glob
import os
import sys

import json
import glob
import os
import sys

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

if __name__ == '__main__':

    path = sys.argv[1]
    json_filenames = glob.glob(path + "/json/*")
    one_hot_map = []

    for filename in json_filenames:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        
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

                tokenized.append("<opcode>")
                if(inst.find(" ") != -1):
                    opcode = inst[:inst.find(" ")]
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
                    tokenized.append(opcode)
                    tokenized.append("</opcode>")

                tokenized.append("</instr>")

            tokenized.append("</block>")

            token = ""
            for tokens in tokenized:
                token += str(tokens)
            item['xml'] = token

        token_to_hot_idx = {}
        hot_idx_to_token = {}
        hexmap = {}

        def hot_idxify(elem):
            if elem not in token_to_hot_idx:
                token_to_hot_idx[elem] = len(token_to_hot_idx)
                hot_idx_to_token[token_to_hot_idx[elem]] = elem
            return token_to_hot_idx[elem]


        map_item = list(map(hot_idxify, tokenized))
        item['nmap'] = map_item
        with open(filename, 'w') as json_file:
            json.dump(item, json_file, indent=4)

        one_hot_map.append(map_item)
