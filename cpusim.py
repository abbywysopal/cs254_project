#!/usr/bin/env python

import re
import argparse
import glob
from instruction import *
from fetchunit import *
from decunit import *
from execution_unit import *
from writeback_unit import *
from registerfile import *
from pipeline import *
import gv
import json

debug = True
debug = False


class Computor:
    def __init__(self, program):
        self._program = program
        self.fetchunit = FetchUnit(program)
        gv.fu = self.fetchunit
        self.decodeunit = DecUnit()
        self.execunit = ExecUnit()
        self.wbunit = WBUnit()
        gv.R = RegisterFile(48)
        self.clock_cnt = 0

    def run_non_pipelined(self, filename):
        if debug:
            print("RUNNING NON-PIPELINED")
        last_instr = getNOP()
        while not isinstance(last_instr, HALTInstruction):
            if debug:
                print("\n\nSTART")
                print("Before anything:", str(gv.pipeline), "clk", self.clock_cnt)
            # fetch
            self.fetchunit.fetch(1)
            self.clock_cnt += 1
            if debug:
                print("After fetch:", str(gv.pipeline), "clk", self.clock_cnt)
            gv.pipeline.advance()

            # decode
            self.decodeunit.decode()
            self.clock_cnt += 1
            if debug:
                print("After decode:", str(gv.pipeline), "clk", self.clock_cnt)
            gv.pipeline.advance()

            # execute
            self.execunit.execute()
            self.clock_cnt += 1
            if debug:
                print("After execute:", str(gv.pipeline), "clk", self.clock_cnt)
            gv.pipeline.advance()

            # writeback
            if debug:
                print("Before writeback:", str(gv.pipeline), "clk", self.clock_cnt)
            last_instr = self.wbunit.writeback()
            self.clock_cnt += 1
            if debug:
                print("After writeback:", str(gv.pipeline), "clk", self.clock_cnt)
            gv.pipeline.advance()

            if debug:
                print("END")

        # if debug:
        output_filename = "tests/data/output_assembly" + filename[filename.index("y") + 1 :]
        output_file = open(output_filename, 'w')
        output_file.write(str(self.clock_cnt))
        print("Cycles taken:", self.clock_cnt)

    def run_pipelined(self, file_name, ml_data):
        if debug:
            print("RUNNING PIPELINED")
        last_instr = getNOP()
        while not isinstance(last_instr, HALTInstruction):
            self.clock_cnt += 1

            if debug:
                print("\n\nSTART")
                print("Before anything:", str(gv.pipeline), "clk", self.clock_cnt)

            last_instr = self.wbunit.writeback()
            if debug:
                print("After writeback:", str(gv.pipeline), "clk", self.clock_cnt)

            self.execunit.execute()
            if debug:
                print("After execute:", str(gv.pipeline), "clk", self.clock_cnt)

            self.decodeunit.decode()
            if debug:
                print("After decode:", str(gv.pipeline), "clk", self.clock_cnt)


            self.fetchunit.fetch(1)
            if debug:
                print("After fetch:", str(gv.pipeline), "clk", self.clock_cnt)

            gv.pipeline.advance()

            if debug:
                print("END")
        #
        # if debug:
        # output_filename = "tests/data/output_assembly" + filename[filename.index("y") + 1 :]
        # output_file = open(output_filename, 'w')
        # output_file.write(str(self.clock_cnt))
        print("Cycles taken:", self.clock_cnt)
        ml_data["cycles"] = self.clock_cnt
        # print(ml_data)
        


def assemble(asm, program, ml_data):
    label_targets = {}
    same_line_no = []
    addr = 0
    num_labels = 0

    ml_data["instr"] = []

    for line_no in range(len(asm)):
        line = asm[line_no].strip()
        ml_data["instr"].append(line)
        if ':' in line and "DATA" not in line:
            same_line_no.append(line[:-1])
        elif 'DATA' not in line:
            if same_line_no:
                num_labels += len(same_line_no)
                for label in same_line_no:
                    label_targets[label] = line_no - num_labels
            same_line_no = []

        else:
            gv.data_mem.append(int(re.search("\d+", line).group()))
            addr += 1

    for i in range(len(asm)):
        line = asm[i].strip()

        opcode = line.split(' ')[0]

        if opcode == 'JMP':
            line = "JMP " + str(label_targets[line.split(' ')[1]])

        if opcode in ['BGEZ', 'BNEZ', 'BLTZ', 'BEQZ']:
            operands = line.split(' ')[1]
            line = opcode + " " + operands .split(',')[0] + "," + str(label_targets[operands .split(',')[1]])

        if 'IADDR' in line:
            dest_reg, label = line.split(' ')[1].split(',')
            line = "LDI " + dest_reg + "," + str(label_targets[label])

        if 'DATA' not in line and ":" not in line:
            instr = get_instruction(line)
            program.append(instr)


def print_data_mem():
    # print("\n+    0 1 2 3"),
    print("")
    for idx, word in enumerate(gv.data_mem):
        print(idx, word)
        # if idx % 4 == 0:
        #     print "\n" + (str(idx) + ".").ljust(4),
        #
        # if word < 256:
        #     to_print = chr(word)
        #     print to_print if to_print.isalnum() else " ",
        # else:
        #     print word,


def main(args):

    list = []
    for i in range(4):
        input_filename = 'tests/assembly/test{}.ass'.format(i+1)

        with open(input_filename, 'r') as ass_file:
            asm = ass_file.readlines()

        ml_data = dict()

        program = []
        assemble(asm, program, ml_data)

        gv.pipeline = Pipeline()
        pc3000 = Computor(program)

        if args.pipelined:
            gv.is_pipelined = True
            pc3000.run_pipelined(file_name='ml_data', ml_data=ml_data)
        else:
            gv.is_pipelined = False
            pc3000.run_non_pipelined()

        print(ml_data)
        list.append(ml_data)

    with open('{}.txt'.format('ml_data'), 'w') as outfile:
        outfile.write('[')
        for item in list:
            outfile.write("%s,\n" % item)
        outfile.write(']')
        # json.dump(ml_data, outfile, indent=4)

    # if debug:
    # print_data_mem()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--file', required=True, help='Input .ass file')
    parser.add_argument('--pipelined', required=False, default=0, type=int, choices={0, 1}, help='Run in pipelined mode?')

    args = parser.parse_args()

    main(args)
