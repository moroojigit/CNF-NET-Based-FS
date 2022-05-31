import argparse
import itertools
import numpy as np 

import copy


class wire:
    def __init__(self, name, type, operands, logic_value, logic_level, prob0, prob1, absprob,
                 fanout, mainout, tag, index):
        self.name = name
        self.type = type
        self.operands = operands
        self.logic_value = logic_value
        self.logic_level = logic_level
        self.prob0 = prob0
        self.prob1 = prob1
        self.absprob = absprob
        self.fanout = fanout
        self.mainout = mainout
        self.tag = tag
        self.index = index


def findsubsets(S, m):
    return set(itertools.combinations(S, m))


def sub_lists(list1):
    # store all the sublists
    sublist = [[]]

    # first loop
    for i in range(len(list1) + 1):

        # second loop
        for j in range(i + 1, len(list1) + 1):
            # slice the subarray
            sub = list1[i:j]
            sublist.append(sub)

    return sublist


def wire_print(wire_in):
    print("w_name: ", wire_in.name)
    print("w_type: ", wire_in.type)
    for i in range(0, len(wire_in.operands)):
        print("w_opr", i, ": ", wire_in.operands[i].name)
    print("w_valu: ", wire_in.logic_value)
    print("w_glev: ", wire_in.logic_level)
    print("w_prob0:", wire_in.prob0)
    print("w_prob1:", wire_in.prob1)
    print("w_abs_prob:", wire_in.absprob)
    print("w_fanout:", wire_in.fanout)
    print("w_mainout:", wire_in.mainout)
    print("index:", wire_in.index, "\n")


def wire_fanin_cone(wire_in, cone_size):
    index_traverse = 0
    if wire_in.type == "inp":
        return []
    else:
        fanin_cone = [wire_in]
        cone_size = cone_size - 1
        temp = copy.deepcopy(fanin_cone[index_traverse].operands)
        while cone_size != 0:
            added_bef = 0
            if len(temp) == 0:
                index_traverse = index_traverse + 1
                if index_traverse >= len(fanin_cone):
                    cone_size = 0
                else:
                    temp = copy.deepcopy(fanin_cone[index_traverse].operands)
            elif temp[0].type == "inp":
                temp.remove(temp[0])
            else:
                for i in range(0, len(fanin_cone)):
                    if fanin_cone[i].name == temp[0].name:
                        added_bef = 1
                if added_bef == 0:
                    fanin_cone.append(temp[0])
                    cone_size = cone_size - 1
                temp.remove(temp[0])
    return fanin_cone


def get_unique_fanin_cone(wire_in):
    return uniquify_wire_list(get_fanin_cone(wire_in))


def get_fanin_cone(wire_in):
    if wire_in.type == "inp":
        return []
    else:
        fanin_cone = []
        fanin_cone.append(wire_in)
        for i in range(len(wire_in.operands)):
            fanin_cone += get_fanin_cone(wire_in.operands[i])
        return fanin_cone


def get_fanin_cone2(wire_in, fanin_cone):
    if wire_in.type != "inp":
        fanin_cone.add(wire_in)
        for i in range(len(wire_in.operands)):
            get_fanin_cone2(wire_in.operands[i], fanin_cone)


def uniquify_wire_list(wire_list):
    seen = set()
    unique = []
    for obj in wire_list:
        if obj.name not in seen:
            unique.append(obj)
            seen.add(obj.name)
    return unique


def wire_dep(benchmark_address, args):
    inputs = []
    outputs = []
    wires = []
    input_array = []
    temp = []
    bench_file = open(benchmark_address)
    index = 1

    for line in bench_file:
        if "INPUT" in line:
            inputs.append(line[line.find("(") + 1:line.find(")")])
            input_array.append(line[line.find("(") + 1:line.find(")")])
            wires.append(wire(line[line.find("(") + 1:line.find(")")], "inp", [], "1", 0, 0.5, 0.5, 0, 0, 0, 0, index))
            input_array = []
            index += 1
        elif "OUTPUT" in line:
            out_name = line[line.find("(") + 1:line.find(")")]
            outputs.append(out_name)
        elif " = " in line:
            gate_out = line[0: line.find(" =")]
            gate_type = line[line.find("= ") + 2: line.find("(")]
            gate_list_inputs = line[line.find("(") + 1:line.find(")")]
            gate_oprs = gate_list_inputs.split(", ")
            for i in range(0, len(gate_oprs)):
                found = False
                for j in range(0, len(wires)):
                    if wires[j].name == gate_oprs[i]:
                        found = True
                        temp.append(wires[j])
                        break
                if not found:
                    print("ERROR")
                    print(gate_out, gate_oprs[i])
                    exit()
            max_level = 0
            for i in range(0, len(temp)):
                if temp[i].logic_level > max_level:
                    max_level = temp[i].logic_level

            temp_prob0 = 0.25
            temp_prob1 = 0.25
            for i in range(0, len(temp)):
                if len(temp) == 1:
                    if gate_type == "NOT" or gate_type == "not":
                        temp_prob = temp[0].prob0
                        temp_prob0 = temp[0].prob1
                        temp_prob1 = temp_prob
                    elif gate_type == "BUFF" or gate_type == "buff":
                        temp_prob0 = temp[0].prob0
                        temp_prob1 = temp[0].prob1
                else:
                    if gate_type == "NAND" or gate_type == "nand":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob0 + temp_prob1 * temp[i].prob0 + temp_prob0 * temp[
                                i].prob1
                            temp_prob1 = temp_prob1 * temp[i].prob1
                    elif gate_type == "AND" or gate_type == "and":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob0 + temp_prob1 * temp[i].prob0 + temp_prob0 * temp[
                                i].prob1
                            temp_prob1 = temp_prob1 * temp[i].prob1
                    elif gate_type == "NOR" or gate_type == "nor":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob0
                            temp_prob1 = 1 - temp_prob0
                    elif gate_type == "OR" or gate_type == "or":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob0
                            temp_prob1 = 1 - temp_prob0
                    elif gate_type == "XNOR" or gate_type == "xnor":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob0 + temp_prob1 * temp[i].prob1
                            temp_prob1 = 1 - temp_prob0
                    elif gate_type == "XOR" or gate_type == "xor":
                        if i == 0:
                            temp_prob0 = temp[i].prob0
                            temp_prob1 = temp[i].prob1
                        else:
                            temp_prob0 = temp_prob0 * temp[i].prob1 + temp_prob1 * temp[i].prob0
                            temp_prob1 = 1 - temp_prob0

            if gate_type == "NAND" or gate_type == "nand":
                temp_prob = temp_prob0
                temp_prob0 = temp_prob1
                temp_prob1 = temp_prob

            if gate_type == "NOR" or gate_type == "nor":
                temp_prob = temp_prob0
                temp_prob0 = temp_prob1
                temp_prob1 = temp_prob

            wires.append(wire(gate_out, gate_type, temp, "1", max_level + 1, temp_prob0, temp_prob1,
                              abs(temp_prob0 - temp_prob1), 0, 0, 0, index))
            temp = []
            index += 1

    bench_file.close()

    # just check for correctness
    for i in range(0, len(wires)):
        for j in range(i + 1, len(wires)):
            if wires[i].name == wires[j].name:
                print(wires[i].name, wires[j].name)
                print("ERROR in reading wires")
                exit()

    for i in range(0, len(wires)):
        fanout_temp = 0
        for j in range(0, len(wires)):
            if i != j:
                for k in range(0, len(wires[j].operands)):
                    if wires[i].name == wires[j].operands[k].name:
                        fanout_temp = fanout_temp + 1
        wires[i].fanout = fanout_temp

    return wires


def simple_read_bench(benchmark_address):
    wires = []
    temp = []
    bench_file = open(benchmark_address)
    index = 1

    for line in bench_file:
        if "#" in line:
            continue
        elif "INPUT" in line:
            wires.append(wire(line[line.find("(") + 1:line.find(")")], "inp", [], "1", 0, 0, 0, 0, 0, 0, 0, index))
        elif "OUTPUT" in line:
            wires.append(wire(line[line.find("(") + 1:line.find(")")], "out", [], "1", 0, 0, 0, 0, 0, 0, 0, index))
        elif " = " in line:
            gate_out = line[0: line.find(" =")]
            gate_type = line[line.find("= ") + 2: line.find("(")].lower()
            gate_list_inputs = line[line.find("(") + 1:line.find(")")]
            gate_oprs = gate_list_inputs.split(",")
            gate_oprs = [x.strip(' ') for x in gate_oprs]
            for i in range(0, len(gate_oprs)):
                found = False
                for j in range(0, len(wires)):
                    if wires[j].name == gate_oprs[i]:
                        found = True
                        temp.append(wires[j])
                        break
                if not found:
                    # print gate_out, gate_oprs[i]
                    temp.append(wire(gate_oprs[i], "dummy", [], "1", 0, 0, 0, 0, 0, 0, 0, 0))
            wires.append(wire(gate_out, gate_type, temp, "1", 0, 0, 0, 0, 0, 0, 0, index))
        else:
            continue

        temp = []
        index += 1

    bench_file.close()

    for i in range(0, len(wires)):
        for j in range(0, len(wires[i].operands)):
            if wires[i].operands[j].type == "dummy":
                found = False
                for k in range(0, len(wires)):
                    if wires[k].name == wires[i].operands[j].name:
                        found = True
                        wires[i].operands[j] = wires[k]
                        break
                if not found:
                    print(wires[i].operands[j].name)
                    print("ERROR1 in read_circuit()")
                    exit()

    # just to be sure!
    for i in range(len(wires)):
        if wires[i].name != wires[wires[i].index - 1].name:
            print(wires[i].name)
            print("ERROR2 in read_circuit()")
            exit()

    return wires


def tseytin_t(wires_in):
    cnf_clause_count = 0
    cnf_clause_list = ""

    for i in range(len(wires_in)):
        if wires_in[i].type == "NOT" or wires_in[i].type == "not":
            cnf_clause_list += str(wires_in[i].index) + " " + str(wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_list += "-" + str(wires_in[i].index) + " -" + str(wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_count += 2

        elif wires_in[i].type == "BUFF" or wires_in[i].type == "buff":
            cnf_clause_list += str(wires_in[i].index) + " -" + str(wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_list += "-" + str(wires_in[i].index) + str(wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_count += 2

        elif wires_in[i].type == "AND" or wires_in[i].type == "and":
            cnf_clause_list += str(wires_in[i].index)
            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += " -" + str(wires_in[i].operands[j].index)
            cnf_clause_list += " 0\n"

            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += "-" + str(wires_in[i].index) + " " + str(wires_in[i].operands[j].index) + " 0\n"

            cnf_clause_count += len(wires_in[i].operands) + 1

        elif wires_in[i].type == "NAND" or wires_in[i].type == "nand":
            cnf_clause_list += "-" + str(wires_in[i].index)
            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += " -" + str(wires_in[i].operands[j].index)
            cnf_clause_list += " 0\n"

            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += str(wires_in[i].index) + " " + str(wires_in[i].operands[j].index) + " 0\n"

            cnf_clause_count += len(wires_in[i].operands) + 1

        elif wires_in[i].type == "OR" or wires_in[i].type == "or":
            cnf_clause_list += "-" + str(wires_in[i].index)
            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += " " + str(wires_in[i].operands[j].index)
            cnf_clause_list += " 0\n"

            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += str(wires_in[i].index) + " -" + str(wires_in[i].operands[j].index) + " 0\n"

            cnf_clause_count += len(wires_in[i].operands) + 1

        elif wires_in[i].type == "NOR" or wires_in[i].type == "nor":
            cnf_clause_list += str(wires_in[i].index)
            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += " " + str(wires_in[i].operands[j].index)
            cnf_clause_list += " 0\n"

            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += "-" + str(wires_in[i].index) + " -" + str(wires_in[i].operands[j].index) + " 0\n"

            cnf_clause_count += len(wires_in[i].operands) + 1

        elif wires_in[i].type == "XOR" or wires_in[i].type == "xor":
            operand_subsets = sub_lists(wires_in[i].operands)
            for j in range(0, len(operand_subsets)):
                if len(operand_subsets[j]) % 2 == 0:
                    cnf_clause_list += "-" + str(wires_in[i].index)
                    for k in range(0, len(wires_in[i].operands)):
                        if wires_in[i].operands[k] not in operand_subsets[j]:
                            cnf_clause_list += " " + str(wires_in[i].operands[k].index)
                        else:
                            cnf_clause_list += " -" + str(wires_in[i].operands[k].index)
                    cnf_clause_list += " 0\n"
                    cnf_clause_count += 1

                elif len(operand_subsets[j]) % 2 == 1 and len(operand_subsets[j]) != 0:
                    cnf_clause_list += str(wires_in[i].index)
                    for k in range(0, len(wires_in[i].operands)):
                        if wires_in[i].operands[k] not in operand_subsets[j]:
                            cnf_clause_list += " " + str(wires_in[i].operands[k].index)
                        else:
                            cnf_clause_list += " -" + str(wires_in[i].operands[k].index)
                    cnf_clause_list += " 0\n"
                    cnf_clause_count += 1

        elif wires_in[i].type == "XNOR" or wires_in[i].type == "xnor":
            operand_subsets = sub_lists(wires_in[i].operands)
            for j in range(0, len(operand_subsets)):
                if len(operand_subsets[j]) % 2 == 1 and len(operand_subsets[j]) != 0:
                    cnf_clause_list += "-" + str(wires_in[i].index)
                    for k in range(0, len(wires_in[i].operands)):
                        if wires_in[i].operands[k] not in operand_subsets[j]:
                            cnf_clause_list += " " + str(wires_in[i].operands[k].index)
                        else:
                            cnf_clause_list += " -" + str(wires_in[i].operands[k].index)
                    cnf_clause_list += " 0\n"
                    cnf_clause_count += 1

                elif len(operand_subsets[j]) % 2 == 0 and len(operand_subsets[j]) != 0:
                    cnf_clause_list += str(wires_in[i].index)
                    for k in range(0, len(wires_in[i].operands)):
                        if wires_in[i].operands[k] not in operand_subsets[j]:
                            cnf_clause_list += " " + str(wires_in[i].operands[k].index)
                        else:
                            cnf_clause_list += " -" + str(wires_in[i].operands[k].index)
                    cnf_clause_list += " 0\n"
                    cnf_clause_count += 1

            cnf_clause_list += str(wires_in[i].index)
            for j in range(0, len(wires_in[i].operands)):
                cnf_clause_list += " " + str(wires_in[i].operands[j].index)
            cnf_clause_list += " 0\n"
            cnf_clause_count += 1

        elif wires_in[i].type == "MUX" or wires_in[i].type == "mux":
            cnf_clause_list += str(wires_in[i].index) + " -" + str(wires_in[i].operands[1].index) + " " + str(
                wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_list += str(wires_in[i].index) + " -" + str(wires_in[i].operands[2].index) + " -" + str(
                wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_list += "-" + str(wires_in[i].index) + " " + str(wires_in[i].operands[1].index) + " " + str(
                wires_in[i].operands[0].index) + " 0\n"
            cnf_clause_list += "-" + str(wires_in[i].index) + " " + str(wires_in[i].operands[2].index) + " -" + str(
                wires_in[i].operands[0].index) + " 0\n"

            cnf_clause_count += 4

    return cnf_clause_count, cnf_clause_list

def cnf_to_matrix(cnf_str):
    """Convert cnf file to bipartite graph matrix (num_clauses,num_clauses)

    Args:
        fpath: The path to the cnf file.
            The first two lines of the file are notes.
            The clauses start from the 3rd line.

    Returns:
        A numpy 2-dimensional array.
        The rows are for clauses.
        The columns are for literals.
        example:

        array([[ 0,  0,  0,  0, -1,  1],
               [ 0,  0,  0,  1,  0, -1],
               [ 0,  0,  0,  0,  1, -1]])

        The example denotes a cnf file with 3 clauses and 6 parameters.
        0 denotes that the literal in this column didn't show up in the
            cluase of this row.
        1 denotes the positive literal exists in this clause.
        -1 denotes the negative literal exists in this clause.

    """
    # lines = open(fpath).readlines()[2:]
    lines = cnf_str.split("\n")

    literals = {}
    clauses = []
    for l in lines:
        for ele in l.strip().split(" ")[:-1]:
            literals[abs(int(ele))] = 0
    num_literals = len(literals)
    num_clauses = len(lines)
    literals_list = list(literals)
    l_to_pos = {}
    pos_init = 0
    for l in literals_list:
        l_to_pos[l] = pos_init
        pos_init += 1

    bi_g = np.zeros((num_clauses, num_literals), dtype=np.int)

    for i in range(num_clauses):
        for ele in lines[i].strip().split(" ")[:-1]:
            literal = int(ele)
            pos = l_to_pos[abs(literal)]
            if literal > 0:
                bi_g[i, pos] = 1
            else:
                bi_g[i, pos] = -1
    return bi_g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ISCAS-85 bench to CNF Converter v1.0 (c) Hadi Mardani Kamali.')
    parser.add_argument("-b", action="store", required=True, type=str, help="benchmark path")
    parser.add_argument("-c", action="store", required=True, type=str, help="cnf path")

    args = parser.parse_args()

    bench_address = args.b
    cnf_address = args.c

    wires = simple_read_bench(bench_address)

    cnf_variable_counts = len(wires)

    cnf_clause_count, cnf_content = tseytin_t(wires)

    cnf_file_content = "Circuit-CNF generated by BENCH2CNFv1.0 (c) Hadi Mardani Kamali\n"
    cnf_file_content += "p cnf " + str(cnf_variable_counts) + " " + str(cnf_clause_count) + "\n" + cnf_content

    cnf_file = open(cnf_address, 'w')
    cnf_file.write(cnf_file_content)
    cnf_file.close()

    print("cnf has been successfully written in", cnf_address)
