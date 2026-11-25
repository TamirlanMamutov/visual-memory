"""Download the frozen MediaPipe hand bundle, validating its known SHA256."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
URL='https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full/resolve/main/hand_landmarker.task'
EXPECTED='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    if not a.out.exists():
        tmp=a.out.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=90) as src,tmp.open('wb') as out:
            while True:
                b=src.read(1024**2)
                if not b:break
                out.write(b)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=EXPECTED:tmp.unlink();raise ValueError('Upstream model changed; obtain the recorded revision rather than silently changing the experiment')
        tmp.replace(a.out)
    if hashlib.sha256(a.out.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Existing model does not match recorded checksum')
    print('HAND_MODEL_VERIFIED',a.out)
if __name__=='__main__':main()
