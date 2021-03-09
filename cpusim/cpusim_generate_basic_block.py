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
def convert_to_hex(binary_instruction):
    # return hex(int(binary_instruction[::-1], 2))[2:]
    return format(int(binary_instruction, 2), '08x')

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
def add_instructions(assembly, binary, hex):
    instructions_list_assembly.append(assembly)
    instructions_list_binary.append(binary)
    instructions_list_hex.append(hex)

# Function to generate an R-Type instruction
'''
funct 7         rs2            rs1         funct3        rd            opcode
(31,25)=7     (24, 20)=5    (19,15)=5      (14, 12)=3    (11, 7)=5     (6,0)=7
'''
def generate_r(name):
    # #print('Generating R')
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

    instruction_binary = FUNCT7[name] + rs2_binary + rs1_binary + FUNCT3[name] + rd_binary + opcode_instruction

    add_instructions(instruction_assembly, instruction_binary, convert_to_hex(instruction_binary))
    # if(len(instruction_binary) > 32):
        #print(instruction_binary, str(len(instruction_binary)))
        #print("TOOO BIG R-type")
        #print(str(len(FUNCT7[name])), str(len(rs2_binary)), str(len(rs1_binary)), str(len(FUNCT3[name])), str(len(rd_binary)), str(len(opcode_instruction)))

# ADDI x ADDI r1, r1, 100
# Function to generate an I-Type instruction
'''
    imm          rs1         funct3      rd          opcode
    (32,20)     (19,15)     (14, 12)    (11, 7)     (6,0)
'''
def generate_i(name):
    # #print('Generating I')
    opcode_instruction = OPCODES[name]
    func_instruction = FUNCT_CODES[name]

    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rs1_binary = "{0:05b}".format(rs1_decimal)

    rd_decimal = random.choice(REGISTERS_TO_USE)
    rd_binary = "{0:05b}".format(rd_decimal)

    if name in LOAD_INSTRUCTION_NAMES:
        '''
    imm         rd         opcode
    (31,12)    (11, 7)     (6,0)
    '''
        rs1_decimal = 0
        imm_decimal = random.choice(STORED_MEMORY_LOCATIONS)  # Choose from stored in locations
        instruction_assembly = name + " R" + str(rd_decimal) + "," + str(imm_decimal)
        imm_binary = "{0:020b}".format(imm_decimal)
        instruction_binary = imm_binary + rd_binary + opcode_instruction
        # if(len(instruction_binary) > 32):
            #print(instruction_binary, str(len(instruction_binary)))
            #print("TOOO BIG I-type")
            #print(str(len(imm_binary)), str(len(rd_binary)), str(len(opcode_instruction)))

    else:
        imm_decimal = np.random.randint(0, 480)
        # imm_decimal = 0
        instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + "," + str(
            imm_decimal)
    
        imm_binary = "{0:012b}".format(imm_decimal)
        instruction_binary = imm_binary + rs1_binary + FUNCT3[name] + rd_binary + opcode_instruction
        # if(len(instruction_binary) > 32):
            #print(instruction_binary, str(len(instruction_binary)))
            #print("TOOO BIG I-type")
            #print(str(len(imm_binary)), str(len(rs1_binary)), str(len(FUNCT3[name])), str(len(opcode_instruction)))

    add_instructions(instruction_assembly, instruction_binary, convert_to_hex(instruction_binary))

# Function to generate a U-Type instruction
# WRS 6
# WR R2
'''
imm         rd         opcode
(31,12)    (11, 7)     (6,0)
'''
def generate_u(name):
    # #print('Generating U')
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
    add_instructions(instruction_assembly, instruction_binary, convert_to_hex(instruction_binary))
    # if(len(instruction_binary) > 32):
        #print(instruction_binary, str(len(instruction_binary)))
        #print("TOOO BIG U-type")
        #print(str(len(imm_binary)), str(len(rd_binary)), str(len(opcode_instruction)))


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

    OPCODES = dict(LDI='0110111', LOAD='0000011', ADDI='0010011', MULI='0010011', DIVI='0010011', 
        MUL='0110011', DIV='0110011', ADD='0110011', XOR='0110011', SUB='0110011', SUBI='0010011', 
        STORE='0100011', WR='0110111', WRS='0110111')


    FUNCT_CODES = dict(WRS='000001', WR='000010', LDI='000011', ADDI='000100', SUBI='000101', MULI='000110',
               DIVI='000111', STORE='001000', LOAD='001001', XOR='001010', ADD='001011', SUB='001100', MUL='001101',
               DIV='001110')

    FUNCT3 = dict(ADDI='000', SUBI='000', MULI='000', DIVI='100', STORE='010', LOAD= '011',XOR='100', ADD='000', SUB='000', MUL='000', DIV='100')
    FUNCT7 = dict(XOR='0000000', ADD='0000000', SUB='0100000', MUL='0000001', DIV='0000001')

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
        REGISTERS_NUMBER = 32
        # Instructions_Number = 15
        Instructions_Number = random.randint(1,31)
        # Instructions_Number = 31
        INSTRUCTION_CURRENT = 0
        STORED_MEMORY_LOCATIONS = []
        instructions_list_assembly = []
        instructions_list_binary = []
        instructions_list_hex = []

        # Random Registers to use
        REGISTERS_TO_USE = np.random.randint(0, 31, int(REGISTERS_NUMBER))
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
        # assembly_text = open(path + "assembly" + str(test_case + 1) + ".txt", "w")
        #binary_file = open(path + "binary" + str(test_case + 1) + ".txt", "w")
        #hex_file = open(path + "hex" + str(test_case + 1) + ".v", "w")

        for i in range(Instructions_Number):
            assembly_file.write(instructions_list_assembly[i] + "\n")
            #binary_file.write("0b" + instructions_list_binary[i] + "\n")
            #hex_file.write("0x" + instructions_list_hex[i] + "\n")
            #assembly_text.write(instructions_list_assembly[i] + "\n")
    
        bin = "00000000000000000000000000110000"
        #binary_file.write("0b" + bin)

        assembly_file.write('HALT\n')
        #hex_file.write("0x00000030")
        assembly_file.close()
        #binary_file.close()
        #hex_file.close()
