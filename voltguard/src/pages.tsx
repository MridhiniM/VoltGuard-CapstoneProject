import {ReactNode,useMemo,useState} from 'react'
import {Download,Pause,Play} from 'lucide-react'
import {DEFAULTS,M,META,Metric,Settings,testConn,validate} from './core'
import {useApp} from './store'
import {TodayPanel,WeekPanel} from './forecast'
import {EventList,Empty,Meter,Panel,RG,Range,SC,SL,SimTag,Spark,Tag,TrendChart,fmtD,fmtT} from './ui'

function Source(){const A=useApp();return A.settings.mode==='sim'?<SimTag/>:<Tag color="#34d399">LIVE · NODE-RED</Tag>}
function Banner(){const A=useApp();const l=A.latest,c=SC[A.status]
 const why=A.status==='offline'?(A.settings.mode==='sim'?'No data':A.conn==='connecting'?'Connecting to Node-RED…':A.err||'No fresh readings from Node-RED'):l?.breaches.length?l.breaches.map(m=>META[m].label).join(', ')+' outside configured limits':l?.anomaly?.alert?'LSTM anomaly alert (advisory): readings are unlike normal behaviour':l?.prediction&&l.prediction.label.toLowerCase()!=='normal'?`ML anomaly (unverified): “${l.prediction.label}”`:'All readings inside configured limits'
 return <div className="flex flex-wrap items-center gap-4 rounded-[28px] border px-6 py-4" style={{borderColor:c+'66',background:c+'14'}} role="status">
  <span className="h-4 w-4 rounded-full" style={{background:c,boxShadow:`0 0 0 6px ${c}33`}}/>
  <div className="min-w-0 flex-1"><div className="text-3xl font-semibold" style={{color:c}}>{SL[A.status]}</div><div className="text-sm text-[#cbd5e1]">{why}</div></div>
  <div className="text-right text-sm"><Source/><div className="lbl mt-1">{l?`Last reading ${fmtT(l.ts)}`:'No readings yet'}</div></div></div>}
function Channel({m,big}:{m:Metric;big?:boolean}){const A=useApp();const l=A.latest,p=A.readings.at(-2),x=META[m],lim=A.settings.limits[m]
 const out=!!l?.breaches.includes(m),d=l&&p?l[m]-p[m]:null
 return <div className="grid grid-cols-[6px_1fr] gap-4 border-t border-ln py-4 first:border-0 first:pt-0 last:pb-0">
  <div className="rounded-full" style={{background:x.color}}/>
  <div className="min-w-0"><div className="flex justify-between gap-2"><span className="lbl">{x.label}</span><span className="lbl">allowed {lim.min}–{lim.max} {x.unit}</span></div>
   <div className="my-2 flex items-end justify-between gap-4"><div>
    <span className={`num ${big?'text-6xl sm:text-7xl':'text-5xl'}`} style={{color:out?(l?.status==='fault'?SC.fault:SC.warning):'#f1f5f9'}}>{l?l[m].toFixed(x.dp):'—'}</span>
    <span className="ml-1 text-lg text-mute">{x.unit}</span>
    <div className="lbl mt-1">{d==null?'No previous reading':`${d>=0?'+':''}${d.toFixed(x.dp)} ${x.unit} vs previous`}</div></div>
    <div className={big?'w-48':'w-28'}><Spark d={A.view} m={m}/></div></div>
   <Meter v={l?.[m]} lim={lim} color={x.color}/></div></div>}
function Trend({pause,h=300,def='15m'}:{pause?:boolean;h?:number;def?:string}){const A=useApp();const [r,setR]=useState(def)
 const to=A.paused?(A.view.at(-1)?.ts??A.now):A.now
 return <Panel title="Trend" right={<div className="flex items-center gap-2">{pause&&<button className="btn" onClick={A.togglePause}>{A.paused?<Play size={14}/>:<Pause size={14}/>}{A.paused?'Resume chart':'Pause chart'}</button>}<Range v={r} set={setR}/></div>}>
  <TrendChart data={A.view} from={to-RG[r]} to={to} h={h}/>{A.paused&&<p className="lbl mt-2">Chart paused. Readings are still being received.</p>}</Panel>}
export function PredBody(){const A=useApp();const l=A.latest,p=l?.prediction
 if(!l)return <Empty text="Waiting for the first reading" busy={A.conn==='connecting'}/>
 if(!p)return <div><p className="text-xl font-semibold">Prediction unavailable</p><p className="mt-1 text-sm text-mute">{A.settings.predictUrl?'The prediction endpoint returned no usable label for the latest reading.':'Model not connected. Add a prediction endpoint in Settings, or include a “prediction” object in the Node-RED payload.'}</p></div>
 const bad=p.label.toLowerCase()!=='normal',row=(k:string,v:ReactNode)=><div className="flex justify-between gap-4 border-t border-ln py-2 text-sm first:border-0"><span className="text-mute">{k}</span><span className="text-right">{v}</span></div>
 return <div><div className="mb-2 flex items-center gap-2">{l.sim&&<SimTag/>}<span className="lbl">{fmtD(l.ts)}</span></div>
  {row('Predicted condition',bad?'ML anomaly (unverified)':'Normal operation')}{row('Fault class',p.label)}
  {row('Confidence',p.confidence!=null?(p.confidence*100).toFixed(1)+'%':'Not supplied by the model')}{row('Safety thresholds',l.breaches.length?'Outside limits':'Within limits')}{row('Severity',l.status==='fault'?'Critical (threshold)':l.status==='warning'?'Review':'None')}
  <p className="mt-3 text-sm text-mute">{l.sim?'Simulated label from a fixed rule, not a trained model. Warnings caught only by thresholds show as “normal” here on purpose.':'Label comes from your model via Node-RED. ML predictions are advisory. Safety thresholds are evaluated independently; this model was trained on a low-current load and may not generalize to other loads.'}</p></div>}
const ASTATE:Record<string,string>={ok:'Scoring',insufficient_data:'Collecting window',sensor_error:'Sensor error (temperature unavailable)',off:'Supply off',unavailable:'Detector not loaded on server',inference_error:'Detector error'}
export function AnomalyBody(){const A=useApp();const l=A.latest,a=l?.anomaly
 if(!l)return <Empty text="Waiting for the first reading" busy={A.conn==='connecting'}/>
 if(!a)return <div><p className="text-xl font-semibold">Anomaly detector not reporting</p><p className="mt-1 text-sm text-mute">{l.sim?'The simulator does not run the LSTM detector.':'The backend payload has no “anomaly” object. Check the backend /api/status for lstm_loaded.'}</p></div>
 const ok=a.state==='ok',row=(k:string,v:ReactNode)=><div className="flex justify-between gap-4 border-t border-ln py-2 text-sm first:border-0"><span className="text-mute">{k}</span><span className="text-right">{v}</span></div>
 const ratio=ok&&a.score!=null&&a.threshold?a.score/a.threshold:null
 return <div>
  {row('Detector state',ASTATE[a.state]||a.state)}
  {ok&&row('Current window',<span style={{color:a.isAnomaly?SC.warning:SC.normal}}>{a.isAnomaly?'Unlike normal':'Looks normal'}</span>)}
  {ok&&row('Alert (3 windows in a row)',<span style={{color:a.alert?SC.warning:SC.normal}}>{a.alert?'ALERT':'None'}</span>)}
  {ok&&row('Reconstruction error',`${a.score?.toFixed(3)} (threshold ${a.threshold?.toFixed(3)}${ratio!=null?`, ${ratio.toFixed(2)}×`:''})`)}
  {a.state==='insufficient_data'&&row('Window',`${a.window??0} / ${a.needed??30} samples`)}
  {ok&&ratio!=null&&<div className="mt-2 h-1.5 rounded-full bg-ln"><div className="h-full rounded-full" style={{width:`${Math.min(100,ratio*50)}%`,background:a.isAnomaly?SC.warning:SC.normal}}/></div>}
  <p className="mt-3 text-sm text-mute">An LSTM autoencoder trained only on normal readings from one low-current load. It reports “unlike normal”, not the fault type or a forecast. Advisory only; about 1% of normal windows are flagged in testing.</p></div>}
export function Overview(){const A=useApp()
 return <div className="grid gap-4 xl:grid-cols-12"><div className="xl:col-span-12"><Banner/></div>
  <Panel className="xl:col-span-5" title="Channels">{M.map(m=><Channel key={m} m={m}/>)}</Panel>
  <div className="xl:col-span-7"><Trend/></div>
  <Panel className="xl:col-span-5" title="Fault prediction"><PredBody/></Panel>
  <Panel className="xl:col-span-7" title="Anomaly detection (LSTM)"><AnomalyBody/></Panel>
  <div className="xl:col-span-12"><TodayPanel/></div>
  <div className="xl:col-span-12"><WeekPanel/></div>
  <Panel className="xl:col-span-12" title="Recent events"><EventList list={A.events}/></Panel></div>}
export function Live(){const A=useApp();const f=Math.min(1,A.age/Math.max(5000,(A.settings.mode==='sim'?A.settings.simMs:A.settings.pollMs)*3))
 return <div className="grid gap-4"><Banner/>
  <Panel title={A.settings.mode==='sim'?'Simulated readings, generated in this browser':'Readings received from Node-RED'} right={<span className="lbl">Connection: {A.settings.mode==='sim'?'local simulator':A.conn}</span>}>
   <div className="grid gap-0 lg:grid-cols-3 lg:gap-8 [&>div]:lg:border-0 [&>div]:lg:pt-0">{M.map(m=><Channel key={m} m={m} big/>)}</div>
   <div className="mt-5"><div className="flex justify-between text-sm"><span className="lbl">Data freshness</span><span style={{color:A.stale?SC.warning:SC.normal}}>{A.latest?(A.stale?'Stale · ':'Fresh · ')+(A.age/1000).toFixed(0)+' s old':'No data'}</span></div>
    <div className="mt-1 h-1.5 rounded-full bg-ln"><div className="h-full rounded-full" style={{width:`${(1-f)*100}%`,background:A.stale?SC.warning:SC.normal}}/></div></div></Panel>
  <Trend pause h={340} def="5m"/>
  {A.settings.mode==='sim'&&<Demo/>}</div>}
function Demo(){const A=useApp();return <Panel title="Demonstration controls" right={<SimTag/>}><div className="flex flex-wrap items-center gap-3">
 <button className="btn" onClick={()=>A.trigger(false)}>Trigger demo warning</button><button className="btn" onClick={()=>A.trigger(true)}>Trigger demo fault</button>
 <span className="lbl">Injects an abnormal simulated reading. These are not hardware events.</span></div></Panel>}
export function Faults(){const A=useApp();const l=A.latest,lim=A.settings.limits,hist=A.events.filter(e=>e.kind==='limit'||e.kind==='prediction'||e.kind==='anomaly')
 return <div className="grid gap-4 lg:grid-cols-2">
  <Panel title="Model classification" right={<Tag color="#8296b3">from the ML model</Tag>}><PredBody/></Panel>
  <Panel title="Anomaly detection (LSTM)" right={<Tag color="#8296b3">from the ML model</Tag>}><AnomalyBody/></Panel>
  <Panel title="Threshold check" right={<Tag color="#8296b3">calculated by VoltGuard</Tag>}>{!l?<Empty text="Waiting for the first reading"/>:<>
   {M.map(m=>{const x=META[m],v=l[m],{min,max}=lim[m],s=v<min?`${(min-v).toFixed(x.dp)} ${x.unit} below minimum`:v>max?`${(v-max).toFixed(x.dp)} ${x.unit} above maximum`:'Within range'
    return <div key={m} className="flex flex-wrap justify-between gap-2 border-t border-ln py-2 text-sm first:border-0"><span>{x.label} {v.toFixed(x.dp)} {x.unit}<span className="lbl ml-2">limits {min}–{max}</span></span><span style={{color:s==='Within range'?SC.normal:SC.warning}}>{s}</span></div>})}
   <p className="mt-3 text-sm text-mute">{l.breaches.length?`Flagged because ${l.breaches.map(m=>META[m].label.toLowerCase()).join(' and ')} left the configured range.${l.status==='fault'&&l.breaches.length?' Readings more than 15% of the range beyond a limit, or a non-normal model class, are shown as a fault.':''}`:'No reading is outside its configured limits.'}</p></>}</Panel>
  <Panel title="Inputs used for the latest prediction">{l?<div className="grid grid-cols-3 gap-3">{M.map(m=><div key={m}><div className="lbl">{META[m].label}</div><div className="num text-3xl">{l[m].toFixed(META[m].dp)}<span className="text-base text-mute"> {META[m].unit}</span></div></div>)}<div className="col-span-3 lbl">Reading time {fmtD(l.ts)}</div></div>:<Empty text="No reading yet"/>}</Panel>
  <Panel title="Fault history"><EventList list={hist} max={20}/></Panel>
  {A.settings.mode==='sim'&&<div className="lg:col-span-2"><Demo/></div>}</div>}
export function HistoryPage(){const A=useApp();const rs=A.readings,end=rs.at(-1)?.ts??A.now
 const [from,setFrom]=useState(''),[to,setTo]=useState(''),[pg,setPg]=useState(0)
 const pf=Date.parse(from),pt=Date.parse(to),f=Number.isFinite(pf)?pf:(rs[0]?.ts??end-36e5),t=Number.isFinite(pt)?pt:end
 const rows=useMemo(()=>rs.filter(r=>r.ts>=f&&r.ts<=t),[rs,f,t])
 const marks=A.events.filter(e=>e.sev!=='info'&&e.kind!=='disconnect'&&e.ts>=f&&e.ts<=t).map(e=>e.ts)
 const per=20,pages=Math.max(1,Math.ceil(rows.length/per)),p=Math.min(pg,pages-1),page=rows.slice().reverse().slice(p*per,p*per+per)
 const csv=()=>{const q=(s:string)=>`"${s.replace(/"/g,'""')}"`
  const body=rows.map(r=>[new Date(r.ts).toISOString(),r.voltage,r.current,r.temperature,r.status,q(r.prediction?.label??''),r.prediction?.confidence??'',r.sim?'simulation':'live'].join(','))
  const b=new Blob([['timestamp,voltage_V,current_A,temperature_C,status,prediction_label,prediction_confidence,source',...body].join('\n')],{type:'text/csv'})
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`voltguard-${A.settings.mode==='sim'?'simulated-':''}readings.csv`;a.click();URL.revokeObjectURL(a.href)}
 return <div className="grid gap-4"><Panel title="Filters" right={<Source/>}><div className="flex flex-wrap items-end gap-3">
  <label className="text-sm"><span className="lbl block">From</span><input className="inp" type="datetime-local" value={from} onChange={e=>{setFrom(e.target.value);setPg(0)}}/></label>
  <label className="text-sm"><span className="lbl block">To</span><input className="inp" type="datetime-local" value={to} onChange={e=>{setTo(e.target.value);setPg(0)}}/></label>
  <button className="btn" onClick={()=>{setFrom('');setTo('')}}>Reset range</button>
  <button className="btn btn-acc" disabled={!rows.length} onClick={csv}><Download size={14}/>Export CSV ({rows.length})</button></div>
  <p className="lbl mt-3">{A.settings.mode==='sim'?'Simulated history, generated locally. Not hardware data.':'Session buffer: readings this browser received from Node-RED since connecting. VoltGuard does not assume Node-RED stores history.'}</p></Panel>
  <Panel title="Readings and fault markers"><TrendChart data={rs} from={f} to={t} marks={marks}/><p className="lbl mt-2">Dashed red lines mark warnings and faults.</p></Panel>
  <Panel title="Readings table">{!rows.length?<Empty text="No readings in this range. Widen the dates or wait for data."/>:<>
   <div className="overflow-x-auto"><table className="w-full min-w-[560px] text-sm"><thead className="text-left text-mute"><tr>{['Time','Voltage (V)','Current (A)','Temp (°C)','Status','Prediction'].map(h=><th key={h} className="py-2 pr-4 font-normal">{h}</th>)}</tr></thead>
   <tbody>{page.map(r=><tr key={r.ts} className="border-t border-ln"><td className="py-1.5 pr-4">{fmtD(r.ts)}</td><td>{r.voltage.toFixed(1)}</td><td>{r.current.toFixed(2)}</td><td>{r.temperature.toFixed(1)}</td><td style={{color:SC[r.status]}}>{SL[r.status]}</td><td>{r.prediction?.label??'—'}</td></tr>)}</tbody></table></div>
   <div className="mt-3 flex items-center justify-end gap-3 text-sm"><button className="btn" disabled={p===0} onClick={()=>setPg(p-1)}>Previous</button><span className="lbl">Page {p+1} of {pages}</span><button className="btn" disabled={p>=pages-1} onClick={()=>setPg(p+1)}>Next</button></div></>}</Panel></div>}
function F({label,children,hint}:{label:string;children:ReactNode;hint?:string}){return <label className="block text-sm"><span className="lbl block">{label}</span>{children}{hint&&<span className="lbl">{hint}</span>}</label>}
export function SettingsPage(){const A=useApp();const [d,setD]=useState<Settings>(A.settings);const [res,setRes]=useState<{ok:boolean;msg:string}|null>(null),[busy,setBusy]=useState(false),[saved,setSaved]=useState(false)
 const errs=validate(d),set=(k:keyof Settings,v:any)=>{setD({...d,[k]:v});setSaved(false)}
 const lim=(m:Metric,k:'min'|'max',v:number)=>{setD({...d,limits:{...d.limits,[m]:{...d.limits[m],[k]:v}}});setSaved(false)}
 const num=(v:number)=>Number.isNaN(v)?'':v
 const save=()=>{if(d.mode!==A.settings.mode&&!window.confirm(d.mode==='live'?'Switch to live data? Simulated data will be cleared and no simulation fallback is used.':'Switch to simulation? Live buffer will be cleared.'))return;A.save(d);setSaved(true)}
 return <div className="grid gap-4 lg:grid-cols-2">
  <Panel title="Node-RED connection"><div className="grid gap-3">
   <F label="Data source"><select className="inp" value={d.mode} onChange={e=>set('mode',e.target.value)}><option value="sim">Simulation</option><option value="live">Live (VoltGuard gateway / Node-RED)</option></select></F>
   <F label="Transport"><select className="inp" value={d.transport} onChange={e=>set('transport',e.target.value)}><option value="http">HTTP polling</option><option value="ws">WebSocket</option></select></F>
   <F label="Node-RED base URL"><input className="inp" value={d.baseUrl} onChange={e=>set('baseUrl',e.target.value)}/></F>
   <F label="Latest-readings path" hint="Combined with the base URL for polling"><input className="inp" value={d.latestPath} onChange={e=>set('latestPath',e.target.value)}/></F>
   <F label="WebSocket URL"><input className="inp" value={d.wsUrl} onChange={e=>set('wsUrl',e.target.value)}/></F>
   <F label="Polling interval (ms)"><input className="inp" type="number" value={num(d.pollMs)} onChange={e=>set('pollMs',e.target.valueAsNumber)}/></F>
   <div className="flex flex-wrap items-center gap-3"><button className="btn" disabled={busy} onClick={async()=>{setBusy(true);setRes(null);setRes(await testConn(d));setBusy(false)}}>{busy?'Testing…':'Test connection'}</button>
    {res&&<span className="text-sm" style={{color:res.ok?SC.normal:SC.fault}}>{res.msg}</span>}</div>
   <p className="lbl">No passwords or tokens are stored. Only these non-secret settings are kept in this browser.</p></div></Panel>
  <Panel title="Prediction model"><div className="grid gap-3">
   <F label="Prediction endpoint (optional)" hint="POST {voltage,current,temperature}. Leave blank if Node-RED sends the prediction itself."><input className="inp" value={d.predictUrl} onChange={e=>set('predictUrl',e.target.value)} placeholder="http://localhost:1880/api/predict"/></F>
   <F label="Label field in response"><input className="inp" value={d.labelKey} onChange={e=>set('labelKey',e.target.value)}/></F>
   <F label="Confidence field in response"><input className="inp" value={d.confKey} onChange={e=>set('confKey',e.target.value)}/></F></div></Panel>
  <Panel title="Operating limits"><div className="grid gap-3">{M.map(m=><div key={m} className="grid grid-cols-[1fr_1fr_1fr] items-end gap-3"><span className="pb-2 text-sm">{META[m].label} ({META[m].unit})</span>
   <F label="Minimum"><input className="inp" type="number" value={num(d.limits[m].min)} onChange={e=>lim(m,'min',e.target.valueAsNumber)}/></F><F label="Maximum"><input className="inp" type="number" value={num(d.limits[m].max)} onChange={e=>lim(m,'max',e.target.valueAsNumber)}/></F></div>)}</div></Panel>
  <Panel title="Simulation and charts"><div className="grid gap-3">
   <F label="Simulation update interval (ms)"><input className="inp" type="number" value={num(d.simMs)} onChange={e=>set('simMs',e.target.valueAsNumber)}/></F>
   <F label="Chart history length (readings)"><input className="inp" type="number" value={num(d.historyLen)} onChange={e=>set('historyLen',e.target.valueAsNumber)}/></F></div></Panel>
  <div className="lg:col-span-2 flex flex-wrap items-center gap-3">
   <button className="btn btn-acc" disabled={errs.length>0} onClick={save}>Save settings</button>
   <button className="btn" onClick={()=>{setD(DEFAULTS);setSaved(false)}}>Restore defaults</button>
   {saved&&<span className="text-sm" style={{color:SC.normal}}>Settings saved</span>}
   {errs.length>0&&<ul className="text-sm" style={{color:SC.fault}}>{errs.map(e=><li key={e}>{e}</li>)}</ul>}</div></div>}
