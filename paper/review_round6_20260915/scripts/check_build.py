"""Check compiled manuscript and arXiv-tree equivalence."""
from pathlib import Path
import fitz, re, json
root=Path(__file__).resolve().parents[1]
pdf=fitz.open(root/'generated/ccf0_round6_20260915.pdf')
arxiv=fitz.open(root/'generated/build/arxiv_source/main.pdf')
assert len(pdf)==len(arxiv)==5
texts=[p.get_text() for p in pdf]
assert 'REFERENCES' in texts[4] and 'CONCLUSION' in texts[3]
for token in ['1817e166','6a52e8f3','ccf0-round4','Frozen-archive','Figure 3','released five-budget','four-setting sweep is released','portable check']:
 assert token not in '\n'.join(texts),token
assert 'Table 6:' in texts[3] and 'Figure 2:' in texts[2]
assert 'IDEA' in texts[0] and 'IDEA' in texts[4]
assert 'Nonzero-threshold sensitivity' in texts[3]
assert 'Budget endpoints' in texts[2]
assert '254807' not in '\n'.join(texts)
assert 'Qihang Wu' in texts[0] and 'Hao Li' not in texts[0]
for tree in ['source','arxiv_source']:
 log=(root/f'generated/build/{tree}/main.log').read_text()
 for pattern in ['Overfull', 'undefined references', 'Citation .* undefined', 'LaTeX Error']:
  assert not re.search(pattern,log),pattern
fonts=(root/'generated/pdffonts.txt').read_text().splitlines()[2:]
assert fonts and all(re.search(r'\byes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',f) and 'Type 3' not in f for f in fonts)
bounds=[]
for i,(p,q) in enumerate(zip(pdf,arxiv)):
 assert p.rect.width==612 and p.rect.height==792
 assert p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).samples==q.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).samples
 spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
 bound=[min(s['bbox'][0] for s in spans),min(s['bbox'][1] for s in spans),max(s['bbox'][2] for s in spans),max(s['bbox'][3] for s in spans)]
 assert bound[0]>50 and bound[2]<563 and bound[1]>70 and bound[3]<721,(i,bound)
 bounds.append(bound)
for p in (root/'source').rglob('*'):
 if p.is_file() and p.suffix in ['.tex','.sty','.pdf']:
  assert (root/'arxiv_source'/p.relative_to(root/'source')).read_bytes()==p.read_bytes()
report={'pages':5,'technical_pages':4,'references_page':5,'font_resources':len(fonts),'all_fonts_embedded_subset':True,'type3_fonts':False,'undefined_references':False,'overfull_boxes':False,'source_arxiv_render_identical':True,'page_text_bounds_pt':bounds,'note':'Glyph bounds allow small font-metric protrusions; visual inspection is recorded separately.'}
(root/'generated/build_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
