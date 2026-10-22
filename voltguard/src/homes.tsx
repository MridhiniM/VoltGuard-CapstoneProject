import {useEffect,useRef,useState} from 'react'
import {M,META,Reading,Status,evaluate,normalize,predict} from './core'
import {useApp} from './store'
import {Empty,Meter,Panel,SC,SL,Spark,Tag,fmtT} from './ui'
interface Home{id:number;name:string;voltage:number;current:number;temperature:number;status:'normal'|'warning'|'fault';prediction:{label:string;confidence:number};rule_breaches:string[];relay:string;injected_fault:string;fault_active_s:number|null;detections:number}
const FAULTS=['voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'] as const
export function Homes(){
 const A=useApp(),base=(import.meta.env.VITE_GATEWAY_URL||'http://localhost:5001').replace(/\/$/,'')
 const [homes,setHomes]=useState<Home[]>([]),[err,setErr]=useState(''),[pick,setPick]=useState<Record<number,string>>({})
 useEffect(()=>{let stop=false
  const poll=async()=>{try{const r=await fetch(base+'/api/homes');if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();if(!stop){setHomes(j.homes);setErr('')}}catch(e:any){if(!stop)setErr(e.message==='Failed to fetch'?'Gateway unreachable at '+base:e.message)}}
  poll();const t=setInterval(poll,1000);return()=>{stop=true;clearInterval(t)}},[base])
 const post=(path:string,body?:object)=>fetch(base+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body||{})}).catch(e=>setErr(String(e.message)))
 const faulty=homes.filter(h=>h.status!=='normal').length
 return <div className="grid gap-4">
  <Panel title="Fleet of 10 simulated homes" right={<div className="flex items-center gap-2"><Tag color="#38bdf8">SIMULATION</Tag><button className="btn" onClick={()=>post('/api/homes/reset')}>Clear all faults</button></div>}>
   <div className="text-sm text-[#cbd5e1]">{err?<span style={{color:SC.fault}}>{err}</span>:`${faulty} of ${homes.length||10} homes flagged. Faults are injected into simulated telemetry; each home is checked independently by threshold rules and the Random Forest.`}</div></Panel>
  {homes.length?<div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{homes.map(h=>{const c=SC[h.status],sel=pick[h.id]||FAULTS[0]
   const why=h.rule_breaches.length?h.rule_breaches.map(m=>META[m as keyof typeof META].label).join(', ')+' outside limits':h.status==='warning'?`ML (unverified): ${h.prediction.label}`:'Within limits'
   return <section key={h.id} className="panel" style={{borderColor:h.status==='normal'?undefined:c+'88'}}>
    <header className="mb-2 flex items-center justify-between"><h3 className="font-semibold">{h.name}</h3><Tag color={c}>{SL[h.status]}</Tag></header>
    <div className="grid grid-cols-3 gap-2">{M.map(m=>{const l=A.settings.limits[m],out=h[m]<l.min||h[m]>l.max
     return <div key={m}><div className="lbl">{META[m].label}</div><div className="num text-xl" style={{color:out?SC.fault:'#f1f5f9'}}>{h[m].toFixed(META[m].dp)}<span className="ml-0.5 text-xs text-mute">{META[m].unit}</span></div></div>})}</div>
    <div className="mt-2 text-sm text-[#cbd5e1]">{why}</div>
    <div className="lbl mt-1">ML: {h.prediction.label} ({Math.round(h.prediction.confidence*100)}%) · {h.relay}{h.fault_active_s!=null?` · flagged ${h.fault_active_s}s ago`:''}</div>
    <div className="mt-3 flex gap-2"><select aria-label={`Fault for ${h.name}`} className="btn flex-1" value={sel} onChange={e=>setPick(p=>({...p,[h.id]:e.target.value}))}>{FAULTS.map(f=><option key={f} value={f}>{f.replace('_',' ')}</option>)}</select>
     <button className="btn" onClick={()=>post(`/api/homes/${h.id}/fault`,{scenario:sel,duration_seconds:30})}>Inject 30s</button></div>
    {h.injected_fault!=='normal'&&<div className="lbl mt-1">Injected: {h.injected_fault}</div>}</section>})}</div>:!err&&<Empty busy text="Loading homes…"/>}</div>}

/** One real home: polls the configured live endpoint (Node-RED / ESP32) regardless of the dashboard's sim/live mode. */
export function LiveHome(){
 const A=useApp(),s=A.settings,url=s.baseUrl.replace(/\/$/,'')+s.latestPath
 const [r,setR]=useState<Reading>(),[hist,setHist]=useState<Reading[]>([]),[src,setSrc]=useState<{sim:boolean;source:string;mode:string}>(),[err,setErr]=useState(''),[now,setNow]=useState(Date.now())
 useEffect(()=>{const t=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(t)},[])
 useEffect(()=>{let stop=false,timer=0
  const poll=async()=>{const ac=new AbortController(),to=setTimeout(()=>ac.abort(),5000)
   try{const res=await fetch(url,{signal:ac.signal,cache:'no-store'})
    if(!res.ok)throw new Error(res.status===503?'No fresh hardware telemetry (HTTP 503)':'HTTP '+res.status)
    const raw=await res.json(),c=sRef.current,x=normalize(raw,Date.now(),c.limits,c.labelKey,c.confKey)
    if(!x)throw new Error('Payload lacks numeric voltage, current and temperature')
    if(!x.prediction&&c.predictUrl){const p=await predict(c,x);if(p)Object.assign(x,{prediction:p},evaluate(x,c.limits,p))}
    if(!stop){setErr('');setSrc({sim:raw.sim!==false,source:String(raw.source??'unknown'),mode:String(raw.input_mode??'')})
     setR(x);setHist(h=>h.length&&x.ts<=h[h.length-1].ts?h:[...h,x].slice(-120))}}
   catch(e:any){if(!stop)setErr(e?.name==='AbortError'?'Request timed out':e.message==='Failed to fetch'?'Unreachable at '+url:e.message)}
   finally{clearTimeout(to);if(!stop)timer=window.setTimeout(poll,s.pollMs)}}
  poll();return()=>{stop=true;clearTimeout(timer)}},[url,s.pollMs])
 const sRef=useRef(s);sRef.current=s
 const age=r?now-r.ts:Infinity,stale=age>Math.max(5000,s.pollMs*3),online=!err&&!!r&&!stale
 const status:Status=online?r!.status:'offline',c=SC[status]
 const real=online&&src?.sim===false
 return <div className="grid gap-4">
  <Panel title="Live home (real sensor feed)" right={<Tag color={real?'#34d399':src?.sim?'#38bdf8':SC.offline}>{real?'LIVE HARDWARE':src?.sim?'SIMULATED SOURCE':'NO DATA'}</Tag>}>
   <div className="text-sm text-[#cbd5e1]">{err?<span style={{color:SC.fault}}>{err}</span>:stale?<span style={{color:SC.warning}}>Last reading is {Math.round(age/1000)}s old</span>:src?.sim?'The endpoint is returning simulated readings, not hardware. Switch Node-RED to hardware mode to see the real home.':`Source: ${src?.source}`}</div>
   <div className="lbl mt-1">Polling {url} every {s.pollMs} ms. Independent of the dashboard's Live/Simulation switch.</div></Panel>
  <section className="panel" style={{borderColor:status==='normal'?undefined:c+'88'}}>
   <header className="mb-2 flex items-center justify-between"><h3 className="font-semibold">Home 1 (live)</h3><Tag color={c}>{SL[status]}</Tag></header>
   {r?<>
    <div className="grid grid-cols-3 gap-4">{M.map(m=>{const l=s.limits[m],out=r.breaches.includes(m)
     return <div key={m}><div className="lbl">{META[m].label}</div><div className="num text-3xl" style={{color:out?SC.fault:online?'#f1f5f9':SC.offline}}>{r[m].toFixed(META[m].dp)}<span className="ml-1 text-sm text-mute">{META[m].unit}</span></div>
      <Meter v={r[m]} lim={l} color={META[m].color}/><Spark d={hist} m={m}/></div>})}</div>
    <div className="mt-2 text-sm text-[#cbd5e1]">{r.breaches.length?r.breaches.map(m=>META[m].label).join(', ')+' outside limits':r.prediction&&r.prediction.label.toLowerCase()!=='normal'?`ML (unverified): ${r.prediction.label}`:'Within limits'}</div>
    <div className="lbl mt-1">ML: {r.prediction?`${r.prediction.label}${r.prediction.confidence!=null?` (${Math.round(r.prediction.confidence*100)}%)`:''}`:'no prediction'} · updated {fmtT(r.ts)}</div></>
   :!err&&<Empty busy text="Waiting for first reading…"/>}</section></div>}
