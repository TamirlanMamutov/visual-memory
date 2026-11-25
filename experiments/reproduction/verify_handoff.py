"""Verify a privately shared handoff against its path-independent SHA256 manifest."""
from pathlib import Path
import argparse,hashlib,json
def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'MANIFEST.json').read_text());count=0
    for relative,expected in manifest['files'].items():
        p=(root/relative).resolve()
        if root not in p.parents:raise ValueError('Unsafe manifest path')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+relative)
        count+=1
    return count
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print('HANDOFF_VERIFIED',verify(a.root),'files')
if __name__=='__main__':main()
