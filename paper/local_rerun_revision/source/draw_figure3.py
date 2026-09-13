"""Preserve photographic pixels and region overlays; replace only raster labels."""
from pathlib import Path
import io, os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
im=Image.open(ROOT/'figures'/'qualitative_cases_compact_labels.png').convert('RGB')
W=86/25.4*72
scale=(W-18)/1635
top=im.crop((125,83,1760,600));bottom=im.crop((125,666,1760,1160))
h1=top.height*scale;h2=bottom.height*scale
y1=13;y2=y1+h1+5;H=y2+h2+1
fig=plt.figure(figsize=(W/72,H/72))
for crop,y,hh in [(top,y1,h1),(bottom,y2,h2)]:
    ax=fig.add_axes([16/W,1-(y+hh)/H,(W-18)/W,hh/H]);ax.imshow(crop,interpolation='none');ax.axis('off')
for cx,label in [(382.5,'(a)'),(956,'(b)'),(1529.5,'(c)')]:
    fig.text((16+(cx-125)*scale)/W,1-6.5/H,label,ha='center',va='center',fontsize=9,color='#303E4A')
for y,hh,label in [(y1,h1,'CCI'),(y2,h2,'WF')]:
    fig.text(6/W,1-(y+hh/2)/H,label,ha='center',va='center',rotation=90,fontsize=9,color='#303E4A')
for ext in ['pdf','svg','png']:
    buffer=io.BytesIO()
    fig.savefig(buffer,format=ext,dpi=260,facecolor='white')
    target=ROOT/'figures'/f'figure3_qualitative_final.{ext}'
    temporary=target.with_suffix(target.suffix+'.tmp')
    with temporary.open('wb') as f:
        f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    temporary.replace(target)
plt.close(fig)
