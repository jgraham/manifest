import manifest
import sys

if len(sys.argv) > 1:
    os = sys.argv[1]
else:
    os = "osx"

with open("example.manifest") as f:
    tree = manifest.compile(f, {"os":os})
    print manifest.to_test_manifest(tree)
