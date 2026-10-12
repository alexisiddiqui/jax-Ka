from pathlib import Path
import json,re,hashlib
out=Path('outputs'); path=out/'atlas-alignment-report.html'
s=path.read_text()
points=json.loads((out/'alignment-phate-points.json').read_text())
compact=[{k:p.get(k) for k in ['id','dataset','split']+[f'{v}-k{n}' for v in ('whole','interface') for n in (5,15,30)]} for p in points]
block='''<section id="dataset-comparison"><h2>Compare datasets on the same map</h2>
<p>Each panel highlights one dataset. Faint grey points show the complete plotted reference population. Coordinates and axis limits are identical across the four panels.</p>
<div class="compare-controls"><label>Features <select id="compare-feature"><option value="whole">Whole sequences</option><option value="interface">Interface residues</option></select></label><label>Neighbours <select id="compare-k"><option value="15">15</option><option value="5">5</option><option value="30">30</option></select></label><label>Highlight <select id="compare-split"><option value="all">All splits</option><option value="val">Validation only</option><option value="train">Training only</option><option value="test">Test only</option></select></label></div>
<p class="muted">Circles: training · Diamonds: validation · Crosses: test. Counts refer to displayed points; sampling differs between cohorts. Switch to validation only for a clearer view of the held-out sets.</p>
<div class="compare-grid">'''+''.join(f'<div class="compare-card"><h3 style="color:{c}">{d}</h3><p id="compare-count-{i}" class="muted"></p><div id="compare-map-{i}" class="compare-map"></div></div>' for i,(d,c) in enumerate([('ATLAS','#0072B2'),('PINDER','#009E73'),('pKPDB','#D55E00'),('Benchmark','#882E72')]))+'''</div><p class="note">These panels use BLOSUM62 alignment distances. Each feature/neighbor setting is a separate joint PHATE fit; the frozen split is unchanged. Whole-sequence and interface maps have different coordinate systems. Visual overlap does not replace the alignment-based leakage checks.</p></section>
<style>.compare-controls{display:flex;gap:24px;flex-wrap:wrap}.compare-controls label{font-weight:650}.compare-controls select{margin-left:8px;padding:9px;border:1px solid #bbc6d3;border-radius:6px;font:inherit;background:white}.compare-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:20px}.compare-card{border:1px solid #dbe2eb;border-radius:10px;padding:12px;min-width:0}.compare-card h3{font-size:23px;margin:0}.compare-card p{font-size:13px;margin:3px 0}.compare-map{height:390px;width:100%}@media(max-width:850px){.compare-grid{grid-template-columns:1fr}}</style>
<script>
(function(){
const points=POINT_DATA;
const datasets=['ATLAS','PINDER','pKPDB','Benchmark'];
const colors=['#0072B2','#009E73','#D55E00','#882E72'];
const symbols={train:'circle',val:'diamond',test:'x'};
function render(){
 const feature=document.getElementById('compare-feature').value+'-k'+document.getElementById('compare-k').value;
 const split=document.getElementById('compare-split').value;
 const all=points.filter(p=>p[feature]!==null);
 const xx=all.map(p=>p[feature][0]),yy=all.map(p=>p[feature][1]);
 const xmin=Math.min(...xx),xmax=Math.max(...xx),ymin=Math.min(...yy),ymax=Math.max(...yy);
 const pad=.07*Math.max(xmax-xmin,ymax-ymin);
 datasets.forEach((ds,i)=>{
  const selected=all.filter(p=>p.dataset===ds&&(split==='all'||p.split===split));
  const counts=['train','val','test'].map(sp=>[sp,selected.filter(p=>p.split===sp).length]);
  document.getElementById('compare-count-'+i).textContent=selected.length.toLocaleString()+' highlighted · '+counts.filter(x=>x[1]).map(x=>x[1].toLocaleString()+' '+x[0]).join(' / ');
  const traces=[{type:'scattergl',mode:'markers',x:xx,y:yy,marker:{color:'#9eabb9',size:3,opacity:.12},hoverinfo:'skip',showlegend:false}];
  for(const sp of ['train','val','test']){
   const pp=selected.filter(p=>p.split===sp);if(!pp.length)continue;
   traces.push({type:'scattergl',mode:'markers',x:pp.map(p=>p[feature][0]),y:pp.map(p=>p[feature][1]),text:pp.map(p=>p.id),name:sp,showlegend:false,marker:{color:colors[i],symbol:symbols[sp],size:sp==='train'?5:7,opacity:sp==='train'?.65:.88},hovertemplate:'%{text}<br>'+ds+' '+sp+'<extra></extra>'});
  }
  Plotly.react('compare-map-'+i,traces,{template:'plotly_white',margin:{l:48,r:15,t:15,b:45},xaxis:{title:'PHATE 1',range:[xmin-pad,xmax+pad],zeroline:false},yaxis:{title:'PHATE 2',range:[ymin-pad,ymax+pad],zeroline:false},paper_bgcolor:'white',plot_bgcolor:'white',annotations:selected.length?[]:[{text:'No plotted entries for this selection',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,bgcolor:'white'}]}, {responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'atlas-'+feature+'-'+ds+'-'+split}});
 });
}
document.getElementById('compare-feature').addEventListener('change',render);
document.getElementById('compare-k').addEventListener('change',render);
document.getElementById('compare-split').addEventListener('change',render);
render();
})();
</script>
'''
block=block.replace('POINT_DATA',json.dumps(compact,separators=(',',':')))
scripts=re.findall(r'<script[^>]*>(.*?)</script>',s,flags=re.S)
library=max(scripts,key=len)
assert len(library)>1000000 and 'plotly' in library[:1000].lower()
intro=s[:s.index('<section id=')]
intro=re.sub(r'<p>Jump to:.*?</p>','',intro,flags=re.S)
html=intro+'<script>'+library+'</script>'+block+'</main></html>'
path.write_text(html)
verification=out/'alignment-export-verification.json'
v=json.loads(verification.read_text());v['sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest();v['interactive_layout']='Responsive per-dataset panels; feature, k and split selectors';verification.write_text(json.dumps(v,indent=2)+'\n')
print('Updated interactive alignment report:',path.resolve())
