import os
import sys

sys.path.insert(0, os.curdir)

from blueprinting.torch import TraceTensorMode

from .mocks import *


def execute_megatron_worker():
    sys.argv = sys.argv[2:]
    with open(sys.argv[0]) as f:
        src = f.read()

    print(f"loading megatron script: {sys.argv[0]}")
    code = compile(src, sys.argv[0], mode="exec")

    with TraceTensorMode():
        args = [
            x
            for arg in sys.argv
            for x in (
                (
                    "\\\n\t",
                    arg,
                )
                if arg.startswith("--")
                else (arg,)
            )
        ]
        print(f"executing megatron script with: \n{' '.join(args)}")
        import __main__

        eval(code, __main__.__dict__)
