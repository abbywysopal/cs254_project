import collections

import gv

Stages = collections.OrderedDict([
        ("FETCH", 0),
        ("DECODE", 1),
        ("EXECUTE", 2),
        ("WRITEBACK", 3)])

State = {
    "OK":     0,
    "STALL":  1,
    "WHOOPS": 2
}

class Pipeline:
    def __init__(self):
        self.NUM_STAGES = 4
        self.pipe = [None] * self.NUM_STAGES
        gv.unit_statuses = ["READY"] * self.NUM_STAGES

    def __str__(self):
        string = ""
        for idx, instr in enumerate(self.pipe):
            string += "("  + list(Stages.keys())[idx][0] + ": " + str(instr) + ") "

        return string

    def advance(self):
        if self.pipe == [None] * self.NUM_STAGES:
            return 2

        if self.pipe[-1]:
            return 1
        else:
            for i in reversed(range(1, self.NUM_STAGES)):
                if gv.unit_statuses[i] == "READY" and gv.unit_statuses[i - 1] == "READY":
                    self.pipe[i] = self.pipe[i - 1]
                    self.pipe[i - 1] = None
            return 0

    def push(self, instr):
        self.pipe[0] = instr

