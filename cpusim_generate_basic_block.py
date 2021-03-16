# inspired by Amr Mohamed's RISCV random test case generator
# https://github.com/Amrsaeed/riscv_test_generator

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
    return format(int(binary_instruction, 2), '08x')

# Appending Instruction in corresponding lists
def add_instructions(assembly):
    instructions_list_assembly.append(assembly)

# Function to generate an R-Type instruction
def generate_r(name):
    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rs2_decimal = random.choice(REGISTERS_TO_USE)
    rd_decimal = random.choice(REGISTERS_TO_USE)

    instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + ",R" + str(
        rs2_decimal)

    add_instructions(instruction_assembly)

# ADDI x ADDI r1, r1, 100
# Function to generate an I-Type instruction
def generate_i(name):
    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rd_decimal = random.choice(REGISTERS_TO_USE)

    if name in LOAD_INSTRUCTION_NAMES:
        rs1_decimal = 0
        imm_decimal = random.choice(STORED_MEMORY_LOCATIONS)  # Choose from stored in locations
        instruction_assembly = name + " R" + str(rd_decimal) + "," + str(imm_decimal)

    else:
        imm_decimal = np.random.randint(0, 480)
        instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + "," + str(
            imm_decimal)

    add_instructions(instruction_assembly)

# Function to generate a U-Type instruction
def generate_u(name):
    rd_decimal = random.choice(REGISTERS_TO_USE)
    imm_decimal = np.random.randint(0, 480)

    if name == 'WRS':
        instruction_assembly = name + " " + str(imm_decimal)
    else:
        instruction_assembly = name + " R" + str(rd_decimal)

    add_instructions(instruction_assembly)


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
        # Instructions_Number = random.randint(1,31)
        Instructions_Number = int(sys.argv[3])
        if(Instructions_Number == 0):
            Instructions_Number = random.randint(1,31)
        INSTRUCTION_CURRENT = 0
        STORED_MEMORY_LOCATIONS = []
        instructions_list_assembly = []

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

        for i in range(Instructions_Number):
            assembly_file.write(instructions_list_assembly[i] + "\n")

        assembly_file.write('HALT\n')
        assembly_file.close()
