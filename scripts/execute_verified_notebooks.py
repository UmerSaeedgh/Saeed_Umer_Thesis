"""Clean-process notebook execution for socket-restricted environments.

Only the %matplotlib inline display directive is replaced by explicit figure capture;
all analysis statements execute in source order in a fresh interpreter per notebook.
"""
from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parent.parent
if len(sys.argv)==1:
 reports=[]
 for p in sorted((ROOT/'distribution_analysis_v1').glob('0*.ipynb')):
  subprocess.run([sys.executable,__file__,str(p)],check=True)
  reports.append(json.loads((ROOT/'verification'/f'{p.stem}_execution.json').read_text()))
 (ROOT/'verification'/'execution_report.json').write_text(json.dumps(reports,indent=2))
else:
 import nbformat
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from IPython.core.interactiveshell import InteractiveShell
 from IPython.utils.capture import capture_output
 from IPython.display import display,Image
 import io
 p=Path(sys.argv[1]).resolve();os.chdir(p.parent);nb=nbformat.read(p,as_version=4)
 shell=InteractiveShell.instance()
 def show(*a,**kw):
  for number in plt.get_fignums():
   fig=plt.figure(number);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130,bbox_inches='tight');display(Image(data=buf.getvalue()))
  plt.close('all')
 plt.show=show;start=time.time();count=0
 for i,c in enumerate(nb.cells):
  if c.cell_type!='code':continue
  count+=1;c.outputs=[];c.execution_count=None
  source='\n'.join(line for line in c.source.splitlines() if line.strip()!='%matplotlib inline')
  with capture_output() as cap:result=shell.run_cell(source,store_history=True)
  if cap.stdout:c.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
  if cap.stderr:c.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
  for item in cap.outputs:c.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
  if result.error_before_exec or result.error_in_exec:
   nbformat.write(nb,p);raise RuntimeError(f'{p.name} cell {i}: {result.error_before_exec or result.error_in_exec}')
  c.execution_count=count;c.metadata={'execution_method':'Fresh Python process, IPython execution; matplotlib inline replaced only for figure capture.'}
  print('PASS',p.name,'cell',i,'count',count,flush=True)
 # Export actual in-memory results for thesis reconciliation.
 out=ROOT/'verification'/'tables';out.mkdir(exist_ok=True)
 import pandas as pd
 for name,value in shell.user_ns.copy().items():
  if isinstance(value,pd.DataFrame) and not name.startswith('_'):
   value.to_csv(out/f'{p.stem}__{name}.csv',index=True)
 nb.metadata['clean_execution']={'method':'Fresh interpreter per notebook, all analysis code in order','code_cells':count,'errors':0,'seconds':round(time.time()-start,1)}
 nbformat.validate(nb);nbformat.write(nb,p)
 (ROOT/'verification'/f'{p.stem}_execution.json').write_text(json.dumps(dict(notebook=p.name,**nb.metadata['clean_execution']),indent=2))
 print('COMPLETE',p.name,count,flush=True)
