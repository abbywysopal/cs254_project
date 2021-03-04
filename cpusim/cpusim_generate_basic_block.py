#   ======================= A RISCV RANDOM TEST CASES GENERATOR ======================    #
#   Program takes as input number of test cases to produce, number of instructions, and   #
#   number of registers to use. Generates the instructions in binary, hex, and assembly.  #
#                           Authored by Amr Mohamed                                       #

import random
import numpy as np
import re
import argparse
import sys

# Function to reverse a dictionary keys with values
def reverse_dict_with_iterable(dictionary):
    rev = {}
    for key, value in dictionary.items():
        for item in value:
            rev[item] = key
    return rev

# Converting a 32 bit binary string instruction to a hexadecimal one
# def convert_to_hex(binary_instruction):
#     # return hex(int(binary_instruction[::-1], 2))[2:]
#     return format(int(binary_instruction, 2), '08x')

# XOR x : XOR R1,R1,R1
# WRS x : WRS 6
# WR R2
# STORE x: STORE R2,R3,0
# BGEZ x: BGEZ R2, L1
# BLTZ x: BLTZ R2, L1
# BEQZ x: BEQZ R2, L1
# BNEZ x: BNEZ R2, L1
# JMP (label) x: JMP L1
# JUMP (reg) x: JUMP R1
# IADDR R2, L1
# LOAD x: LOAD R2,R3,0
# LDI x: LDI R2, 20
# ADDI x ADDI r1, r1, 100
# SUBI x
# MULI x
# DIVI x
# ADD x
# SUB x
# MUL x
# DIV x


# Appending Instruction in corresponding lists
def add_instructions(assembly, binary):
    instructions_list_assembly.append(assembly)
    instructions_list_binary.append(binary)


# Function to generate an R-Type instruction
def generate_r(name):
    # print('Generating R')
    opcode_instruction = OPCODES[name]
    func_instruction = FUNCT_CODES[name]

    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rs1_binary = "{0:05b}".format(rs1_decimal)

    rs2_decimal = random.choice(REGISTERS_TO_USE)
    rs2_binary = "{0:05b}".format(rs2_decimal)

    rd_decimal = random.choice(REGISTERS_TO_USE)
    rd_binary = "{0:05b}".format(rd_decimal)

    instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + ",R" + str(
        rs2_decimal)

    instruction_binary = '0000000' + rs2_binary + rs1_binary + func_instruction + rd_binary + opcode_instruction

    add_instructions(instruction_assembly, instruction_binary)

# ADDI x ADDI r1, r1, 100
# Function to generate an I-Type instruction
def generate_i(name):
    # print('Generating I')
    opcode_instruction = OPCODES[name]
    func_instruction = FUNCT_CODES[name]

    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rs1_binary = "{0:05b}".format(rs1_decimal)

    rd_decimal = random.choice(REGISTERS_TO_USE)
    rd_binary = "{0:05b}".format(rd_decimal)

    if name in LOAD_INSTRUCTION_NAMES:
        rs1_decimal = 0
        imm_decimal = random.choice(STORED_MEMORY_LOCATIONS)  # Choose from stored in locations
        instruction_assembly = name + " R" + str(rd_decimal) + "," + str(imm_decimal)
    else:
        imm_decimal = np.random.randint(0, 480)
        # imm_decimal = 0
        instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + "," + str(
            imm_decimal)
    
    imm_binary = "{0:012b}".format(imm_decimal)
    instruction_binary = imm_binary + rs1_binary + func_instruction + rd_binary + opcode_instruction

    add_instructions(instruction_assembly, instruction_binary)

# Function to generate a U-Type instruction
# WRS 6
# WR R2
def generate_u(name):
    # print('Generating U')
    opcode_instruction = OPCODES[name]
    rd_decimal = random.choice(REGISTERS_TO_USE)
    rd_binary = "{0:05b}".format(rd_decimal)

    #TODO: change to a random number
    imm_decimal = np.random.randint(0, 480)
    # imm_decimal = 0
    imm_binary = "{0:020b}".format(imm_decimal)

    if name == 'WRS':
        instruction_assembly = name + " " + str(imm_decimal)
    else:
        instruction_assembly = name + " R" + str(rd_decimal)
        
    instruction_binary = imm_binary + rd_binary + opcode_instruction
    add_instructions(instruction_assembly, instruction_binary)


# Instruction generation wrapper
def generate_instruction(name):
    if INSTRUCTION_TO_TYPE[name] == 'R_TYPE':
        generate_r(name)
    elif INSTRUCTION_TO_TYPE[name] == 'I_TYPE':
        generate_i(name)
    elif INSTRUCTION_TO_TYPE[name] == 'U_TYPE':
        generate_u(name)

if __name__ == '__main__':

    path = str(sys.argv[1])

    TYPES_TO_INSTRUCTION = dict(U_TYPE={'WRS', 'WR'},
                                I_TYPE={'LDI', 'ADDI', 'SUBI', 'MULI', "DIVI", 'STORE', 'LOAD'},
                                R_TYPE={'XOR', 'ADD', 'SUB', 'MUL', "DIV"})

    OPCODES = dict(WRS='000001', WR='0000010', LDI='0001010', ADDI='0001011', SUBI='0001100', MULI='0001101',
               DIVI='0001110', STORE='0001111', LOAD='0010000', XOR='0010001', ADD='0010010', SUB='0010011', MUL='0010100',
               DIV='0010101')

    FUNCT_CODES = dict(WRS='001', WR='010', LDI='010', ADDI='011', SUBI='100', MULI='101',
               DIVI='110', STORE='111', LOAD='010', XOR='0001', ADD='010', SUB='011', MUL='100',
               DIV='101')

    LOAD_INSTRUCTION_NAMES = {'LDI'}

    STORE_INSTRUCTION_NAMES = {'STORE', 'WR', "WRS"}

    M_EXTENSION_NAMES = {'MUL', 'MULI', 'DIV', 'DIVI'}

    # Reversing the instructions table to correlate each instruction with its type directly
    INSTRUCTION_TO_TYPE = reverse_dict_with_iterable(TYPES_TO_INSTRUCTION)

    TEST_CASES_NUMBER = 0
    # Validaing Input
    while int(TEST_CASES_NUMBER) < 1:
        TEST_CASES_NUMBER = int(sys.argv[2])

    for test_case in range(int(TEST_CASES_NUMBER)):
        # Initializing all variables
        REGISTERS_NUMBER = 48
        # Instructions_Number = 15
        # Instructions_Number = random.randint(1,29)
        Instructions_Number = 31
        INSTRUCTION_CURRENT = 0
        STORED_MEMORY_LOCATIONS = []
        instructions_list_assembly = []
        instructions_list_binary = []

        # Random Registers to use
        REGISTERS_TO_USE = np.random.randint(0, 47, int(REGISTERS_NUMBER))
        Instructions_Number = int(Instructions_Number)

        # Generating instructions
        for instruction in range(Instructions_Number):
            INSTRUCTION_CURRENT = instruction
            instruction_name = random.choice(list(INSTRUCTION_TO_TYPE.keys()))

            # Check for load instruction with no prior store
            while instruction_name in LOAD_INSTRUCTION_NAMES and len(STORED_MEMORY_LOCATIONS) == 0:
                instruction_name = random.choice(list(INSTRUCTION_TO_TYPE.keys()))

            generate_instruction(instruction_name)

        # Writing and formatting ouput files
        assembly_file = open(path + "assembly" + str(test_case + 1) + ".ass", "w")
        #assembly_text = open(path + "assembly" + str(test_case + 1) + ".txt", "w")
        #binary_file = open(path + "binary" + str(test_case + 1) + ".txt", "w")

        #binary_file.write('0b00000000000000000000000000000000' + '\n')
        #assembly_text.write('L1:\n')
        for i in range(Instructions_Number):
            assembly_file.write(instructions_list_assembly[i] + "\n")
            #binary_file.write("0b" + instructions_list_binary[i] + "\n")
            #assembly_text.write(instructions_list_assembly[i] + "\n")
    
        imm_decimal = Instructions_Number * 4
        imm_binary = "{0:032b}".format(imm_decimal)
        #binary_file.write("0b" + imm_binary + '\n')
        #assembly_text.write('L' + str(lastlabel) + ':\n')    
        assembly_file.write('HALT\n')
        #binary_file.write('0b11111111111111111111111111111111' + '\n')
        assembly_file.close()
        #binary_file.close()
        #assembly_text.write('HALT\n')
        #assembly_text.close()
