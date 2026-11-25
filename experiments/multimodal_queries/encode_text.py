"""Encode new descriptions/transcripts against cached CLIP candidate vectors."""
from pathlib import Path
import argparse,json
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','texts','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--field',choices=['description_score','speech_score','name_score'],default='description_score')
    p.add_argument('--model',required=True,help='Same local CLIP model directory or HF model ID used for candidate vectors')
    p.add_argument('--local-files-only',action='store_true');p.add_argument('--device',default='cpu')
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());supplied=json.loads(a.texts.read_text())
    texts={q['key']:q['text'] for q in supplied['queries']}
    if len(texts)!=len(supplied['queries']) or set(texts)!={q['key'] for q in bundle['queries']}:raise ValueError('Exact text/query coverage required')
    if bundle.get('appearance_model')!=a.model:raise ValueError('Text and image vectors must use the identical CLIP model identifier/path and weights')
    import torch
    from transformers import CLIPModel,CLIPProcessor
    torch.set_num_threads(1);model=CLIPModel.from_pretrained(a.model,local_files_only=a.local_files_only).to(a.device).eval();processor=CLIPProcessor.from_pretrained(a.model,local_files_only=a.local_files_only)
    if bundle.get('appearance_revision') and getattr(model.config,'_commit_hash',None)!=bundle['appearance_revision']:raise ValueError('CLIP revision differs from image-vector export')
    for q in bundle['queries']:
        with torch.inference_mode():
            x=processor(text=[texts[q['key']]],return_tensors='pt',truncation=True).to(a.device)
            v=model.text_projection(model.text_model(**x,return_dict=True).pooler_output)
            v=torch.nn.functional.normalize(v,dim=-1)[0].cpu().numpy()
        for c in q['candidates']:
            import numpy as np
            image=np.asarray(c['appearance_embedding'])
            if image.shape!=v.shape:raise ValueError('Embedding dimension mismatch')
            c[a.field]=float(image@v)
            # Previously frozen cached combined scores are no longer applicable.
            c.pop('modality_scores',None)
    bundle['extra_text_input']=dict(field=a.field,provenance=supplied.get('provenance','unspecified'),semantics='Typed description or ASR transcript, not independent acoustic-event evidence')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(bundle,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
