"""Check that data/ holds the exact files used in the paper (SHA-256). Files may sit in any sub-folder of data/."""
import sys, os, hashlib
J = sys.argv[1] if len(sys.argv) > 1 else "."
found = {}
for root, _, fs in os.walk(os.path.join(J, "data")):
    for f in fs: found.setdefault(f, os.path.join(root, f))
ok = True
for line in open(os.path.join(J, "data", "checksums.sha256")):
    want, name = line.split(maxsplit=1); name = name.strip()
    if name not in found: print(f"MISSING  {name}"); ok = False; continue
    h = hashlib.sha256(open(found[name], "rb").read()).hexdigest()
    print(("OK       " if h == want else "CHANGED  ") + name); ok &= (h == want)
print("all files match" if ok else "some files are missing or differ: results may not match the paper")
