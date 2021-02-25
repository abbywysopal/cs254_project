#   ======================= A RISCV RANDOM TEST CASES GENERATOR ======================    #
#   Program takes as input number of test cases to produce, number of instructions, and   #
#   number of registers to use. Generates the instructions in binary, hex, and assembly.  #
#                           Authored by Amr Mohamed                                       #

import random
import numpy as np
import re


# Function to reverse a dictionary keys with values
def reverse_dict_with_iterable(dictionary):
    rev = {}
    for key, value in dictionary.items():
        for item in value:
            rev[item] = key
    return rev

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


TYPES_TO_INSTRUCTION = dict(U_TYPE={'WRS', 'WR'}, UJ_TYPE={'JMP', 'JUMP', 'IADDR'},
                            SB_TYPE={'BGEZ', 'BLTZ', 'BEQZ', 'BNEZ'},
                            I_TYPE={'LDI', 'ADDI', 'SUBI', 'MULI', "DIVI", 'STORE', 'LOAD'},
                            R_TYPE={'XOR', 'ADD', 'SUB', 'MUL', "DIV"}, L_TYPE = {'L'})

LOAD_INSTRUCTION_NAMES = {'LDI'}

STORE_INSTRUCTION_NAMES = {'STORE', 'WR', "WRS"}

M_EXTENSION_NAMES = {'MUL', 'MULI', 'DIV', 'DIVI'}

# Reversing the instructions table to correlate each instruction with its type directly
INSTRUCTION_TO_TYPE = reverse_dict_with_iterable(TYPES_TO_INSTRUCTION)

TEST_CASES_NUMBER = 0

# Appending Instruction in corresponding lists
def add_instructions(assembly):
    instructions_list_assembly.append(assembly)


# Function to generate an R-Type instruction
def generate_r(name):
    # print('Generating R')
    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rs2_decimal = random.choice(REGISTERS_TO_USE)
    rd_decimal = random.choice(REGISTERS_TO_USE)

    instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + ",R" + str(
        rs2_decimal)

    add_instructions(instruction_assembly)

# ADDI x ADDI r1, r1, 100
# Function to generate an I-Type instruction
def generate_i(name):
    # print('Generating I')
    rs1_decimal = random.choice(REGISTERS_TO_USE)
    rd_decimal = random.choice(REGISTERS_TO_USE)

    if name in LOAD_INSTRUCTION_NAMES:
        rs1_decimal = 0
        imm_decimal = random.choice(STORED_MEMORY_LOCATIONS)  # Choose from stored in locations
        instruction_assembly = name + " R" + str(rd_decimal) + "," + str(imm_decimal * 4)
    else:
        imm_decimal = np.random.randint(0, 4095)
        instruction_assembly = name + " R" + str(rd_decimal) + ",R" + str(rs1_decimal) + "," + str(
            imm_decimal)

    add_instructions(instruction_assembly)

# Function to generate an SB-Type instruction
# BGEZ x: BGEZ R2, L1
# BLTZ x: BLTZ R2, L1
# BEQZ x: BEQZ R2, L1
# BNEZ x: BNEZ R2, L1
def generate_sb(name):
    # print('Generating SB')
    rs1_decimal = random.choice(REGISTERS_TO_USE)

    label = STORED_LABELS.pop() + 1
    STORED_LABELS.append(label - 1)

    instruction_assembly = name + " R" + str(rs1_decimal) + ",L" + str(label) 

    add_instructions(instruction_assembly)


# Function to generate a U-Type instruction
# WRS 6
# WR R2
def generate_u(name):
    # print('Generating U')
    rd_decimal = random.choice(REGISTERS_TO_USE)
    imm_decimal = np.random.randint(0, 1048575)

    if name == 'WRS':
        instruction_assembly = name + " " + str(imm_decimal)
    else:
        instruction_assembly = name + " R" + str(rd_decimal)
        
    add_instructions(instruction_assembly)


# Function to generate an UJ-Type instruction
# JMP (label) x: JMP L1
# JUMP (reg) x: JUMP R1
# IADDR R2, L1
def generate_uj(name):
    # print('Generating UJ')
    rd_decimal = random.choice(REGISTERS_TO_USE)
    imm_decimal = INSTRUCTION_CURRENT * 4

    # If address is current one, regenerate another.
    while imm_decimal == INSTRUCTION_CURRENT * 4:
        imm_decimal = 2 * np.random.randint(0, Instructions_Number * 2)

    label = STORED_LABELS.pop() + 1
    STORED_LABELS.append(label - 1)

    if name == 'JUMP':
        instruction_assembly = name + " R" + str(rd_decimal)
    elif name == 'JMP':
        instruction_assembly = name + " L" + str(label)
    else:
        instruction_assembly = name + " R" + str(rd_decimal) + ",L" + str(label)

    add_instructions(instruction_assembly)

def generate_label(name):
    label = STORED_LABELS.pop() + 1
    STORED_LABELS.append(label)
    instruction_assembly = "L" + str(label) + ":"
    add_instructions(instruction_assembly)


# Instruction generation wrapper
def generate_instruction(name):
    if INSTRUCTION_TO_TYPE[name] == 'R_TYPE':
        generate_r(name)
    elif INSTRUCTION_TO_TYPE[name] == 'I_TYPE':
        generate_i(name)
    elif INSTRUCTION_TO_TYPE[name] == 'SB_TYPE':
        generate_sb(name)
    elif INSTRUCTION_TO_TYPE[name] == 'U_TYPE':
        generate_u(name)
    elif INSTRUCTION_TO_TYPE[name] == 'UJ_TYPE':
        generate_uj(name)
    elif INSTRUCTION_TO_TYPE[name] == 'L_TYPE':
        generate_label(name)


# Validaing Input
while int(TEST_CASES_NUMBER) < 1:
    TEST_CASES_NUMBER = 500

for test_case in range(int(TEST_CASES_NUMBER)):
    # Initializing all variables
    REGISTERS_NUMBER = 48
    # Instructions_Number = 15
    Instructions_Number = random.randint(1,30)
    INSTRUCTION_CURRENT = 0
    STORED_MEMORY_LOCATIONS = []
    STORED_LABELS = [1]
    instructions_list_assembly = []

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
    assembly_file = open("tests/data/assembly" + str(test_case + 1) + ".ass", "w")
    assembly_text = open("tests/data/assembly" + str(test_case + 1) + ".txt", "w")

    assembly_file.write('L1:\n')
    assembly_text.write('L1:\n')
    for i in range(Instructions_Number):
        assembly_file.write(instructions_list_assembly[i] + "\n")
        assembly_text.write(instructions_list_assembly[i] + "\n")

    lastlabel = STORED_LABELS.pop() + 1
    assembly_file.write('L' + str(lastlabel) + ':\n')
    assembly_text.write('L' + str(lastlabel) + ':\n')    
    assembly_file.write('HALT\n')
    assembly_file.close()
    assembly_text.write('HALT\n')
    assembly_text.close()

