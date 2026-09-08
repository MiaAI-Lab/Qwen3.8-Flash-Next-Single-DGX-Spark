#!/usr/bin/env python3
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "weight_utils_patched.py.orig")
OUT = os.path.join(HERE, "weight_utils_patched.py")

def patch() -> None:
    src = open(ORIG).read()
    old_code = (
        "                    param = f.get_tensor(name)\n"
        "                    yield name, param\n"
    )
    new_code = (
        "                    param = f.get_tensor(name)\n"
        "                    yield name, param.clone()\n"
    )
    if src.count(old_code) != 1:
        raise AssertionError(f"weight_utils: anchor missing or not unique (count={src.count(old_code)})")
    src = src.replace(old_code, new_code, 1)
    open(OUT, "w").write(src)
    print("ok", OUT)

if __name__ == "__main__":
    if not os.path.isfile(ORIG):
        print(f"ERROR: missing {ORIG}", file=sys.stderr)
        sys.exit(1)
    patch()
