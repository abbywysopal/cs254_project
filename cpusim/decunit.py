import gv
from pipeline import *
import instruction


class DecUnit:
    LAST_dest = " "

    def decode(self):
        hazard = False
        instr = gv.pipeline.pipe[Stages["DECODE"]]

        EXECUTE_INST = gv.pipeline.pipe[Stages["EXECUTE"]]
        gv.unit_statuses[Stages["DECODE"]] = "BUSY"

        if instr:
            instr.decode()

            # check for jump?
            if instr.isUncondBranch:
                gv.fu.jump(instr.target)
                gv.pipeline.pipe[Stages["DECODE"]] = instruction.getNOP()

            if EXECUTE_INST is not None:
                srcs = instr.src

                if EXECUTE_INST.dest in srcs:
                    hazard = True

        # check for hazard? -- ??
        # if hazard
            # save instruction
            # feed NOP(s?)
            # next time, feed actual instruction
        # self.saved_instr

        gv.unit_statuses[Stages["DECODE"]] = "READY"


        return hazard
