"""Prepare ordinary submission files from the compiled manuscript; never submit."""
from pathlib import Path
import re,json,csv,shutil,hashlib
root=Path(__file__).resolve().parents[1];out=root/'submission';out.mkdir(exist_ok=True)
tex=(root/'source/main.tex').read_text()
title=re.search(r'\\title\{([^}]+)\}',tex).group(1).upper()
abstract=tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
abstract=abstract.replace(r'$\epsilon=.02$','epsilon = 0.02').replace(r'\%','%').replace('--','-');abstract=' '.join(abstract.split())
assert abstract.isascii() and 100<=len(abstract.split())<=150
keywords='CLIP, explanation auditing, class specificity, constrained region selection'
for name,content in [('title.txt',title),('abstract.txt',abstract),('keywords.txt',keywords)]: (out/name).write_text(content+'\n',encoding='ascii')
aff1='Taizhou Institute of Science and Technology, Nanjing University of Science and Technology';a1='Taizhou 225300, Jiangsu, China'
authors=[]
for n,e,a,addr,oid,corr in [('Kaixin Liu','24107880127@nustti.edu.cn',aff1,a1,'https://orcid.org/0009-0005-5213-8081',False),('Zhipeng Ye','zhipengye@nustti.edu.cn',aff1,a1,'https://orcid.org/0000-0002-3384-2779',True),('Feng Jiang','jf@nustti.edu.cn',aff1,a1,'https://orcid.org/0000-0001-5362-3234',False),('Zhenghao Wang','wangzhenghao2002@outlook.com',aff1,a1,'https://orcid.org/0009-0002-8768-4937',False),('Qihang Wu','24107880128@nustti.edu.cn',aff1,a1,'https://orcid.org/0009-0009-6082-0223',False)]:
 authors.append(dict(order=len(authors)+1,name=n,email=e,affiliation=a,mailing_address=addr,orcid=oid,corresponding=corr))
with (out/'authors.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(authors[0]),lineterminator="\n");w.writeheader();w.writerows(authors)
metadata=dict(title=title,abstract=abstract,abstract_word_count=len(abstract.split()),keywords=keywords.split(', '),authors=authors,ready_to_submit=False,pending=['Author-confirmed conflicts-of-interest statement to insert on page 5','Author confirmation of ethical-compliance statement; final author approval'],review_category=None,funding='This work was supported by the Young Scientific and Technological Talent Support Program under the Taizhou Fengcheng Talent Plan.',conflicts_of_interest=None,acknowledgments_in_pdf=False)
(out/'submission_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
shutil.copy2(root/'generated/ccf0_round8_20260915.pdf',out/'liu.pdf')
shutil.copy2(out/'liu.pdf',out/'liu_five_authors_20260923.pdf')
files=[p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt']
(out/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(files)))
print('Prepared liu.pdf and form fields; missing author-supplied information remains explicit.')
