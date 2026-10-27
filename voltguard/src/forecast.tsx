import {useEffect,useState} from 'react'
import {Empty,Panel,SC,Tag} from './ui'

/** Weather-aware forecasts from the Flask gateway (/api/predict/today, /api/predict/weekly-outlook, /api/outage/last).
 *  Additive to the existing dashboard: nothing here changes readings, limits or fault status. */
const BASE=(import.meta.env.VITE_GATEWAY_URL||'http://127.0.0.1:5001').replace(/\/$/,'')
const RISK:Record<string,string>={low:SC.normal,normal:SC.normal,elevated:SC.warning,high:SC.fault,unknown:SC.offline}
const pct=(p:number|null|undefined)=>p==null?'—':p<0.001?'<0.1%':`${(p*100).toFixed(1)}%`

function useJson(path:string,everyMs=600000){
 const [d,setD]=useState<any>(null),[err,setErr]=useState('')
 useEffect(()=>{let stop=false
  const load=async()=>{try{const r=await fetch(BASE+path,{cache:'no-store'});const j=await r.json()
    if(!stop){if(r.ok&&j.state==='ok'){setD(j);setErr('')}else setErr(j?.reason||j?.error||'HTTP '+r.status)}}
   catch{if(!stop)setErr('Gateway unreachable at '+BASE)}}
  load();const t=setInterval(load,everyMs);return()=>{stop=true;clearInterval(t)}},[path,everyMs])
 return{d,err}}

const Updated=({d}:{d:any})=>d?<span className="lbl">{d.location?.name} · Open-Meteo {new Date(d.forecast_fetched_at).toLocaleTimeString([],{hour12:false,hour:'2-digit',minute:'2-digit'})}{d.forecast_stale?' (stale)':''}</span>:null
const why=(r:string[])=>r.length?` due to ${r.join(', ')}`:''

function LastOutage(){const {d}=useJson('/api/outage/last',60000)
 if(!d)return null
 const f=(iso:string)=>{const t=new Date(iso);return `${t.toLocaleDateString([],{day:'numeric',month:'short',year:'numeric'})} ${t.toLocaleTimeString([],{hour12:false,hour:'2-digit',minute:'2-digit'})}`}
 const me=d.this_site
 return <div className="mt-3 grid gap-1 text-sm">
  <div><span className="lbl">Last fault at your site (VoltGuard hardware): </span>{me?(me.ongoing?<span style={{color:SC.fault}}>supply lost since {f(me.start)}</span>
   :<span>{me.kind==='supply_lost'?'supply lost':'no telemetry (possible outage)'} {f(me.start)}, {me.minutes} min</span>):<span className="text-mute">none recorded yet (logging live ESP32 readings)</span>}</div>
  {d.site&&<div><span className="lbl">Last recorded outage in Bengaluru (ESMI, data ends 2018): </span>{d.site}, {f(d.start)}, {d.minutes} min{d.weather?` · ${d.weather.temperature_c} °C, rain ${d.weather.precip_mm} mm/h`:''}</div>}</div>}

function Stat({v,l,sub}:{v:string;l:string;sub?:string}){return <div className="rounded-xl bg-[#0a1424] px-4 py-3">
 <div className="num text-3xl" style={{color:'#2dd4bf'}}>{v}</div><div className="mt-1 text-sm">{l}</div>{sub&&<div className="lbl mt-0.5">{sub}</div>}</div>}

export function ModelCard(){const {d,err}=useJson('/api/outage/model-card',3600000)
 if(!d)return <Panel title="Outage model performance">{err?<p className="text-sm text-mute">Unavailable: {err}</p>:<Empty busy text="Loading…"/>}</Panel>
 const f=d.flagged,a=d.accuracy,r=d.roc_auc,rain=d.rain_effect,P=(x:number)=>`${(x*100).toFixed(1)}%`
 return <Panel title="Outage model performance" right={<span className="lbl">LSTM · tested on unseen {d.test_period.slice(3,7)} data</span>}>
  <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
   <Stat v={`${d.trained_on.outages.toLocaleString()}`} l="real outages learned from" sub={`${d.trained_on.sites} Bengaluru sites, 2014–2018`}/>
   <Stat v={`${f.lift.toFixed(1)}×`} l="more outages in flagged hours" sub={`${P(f.observed_onset_rate)} vs ${P(f.unflagged_onset_rate)} in other hours`}/>
   <Stat v={`${(rain.heavy/rain.dry).toFixed(1)}×`} l="outage rate in heavy rain" sub={`${P(rain.heavy)} vs ${P(rain.dry)} when dry (observed)`}/>
   <Stat v={r.lstm_weather.toFixed(3)} l="ROC-AUC (ranking skill)" sub={`no weather ${r.lstm_no_weather.toFixed(3)} · usual-rate baseline ${r.usual_rate_baseline.toFixed(3)}`}/></div>
  <p className="lbl mt-3">Accuracy is {P(a['model_at_0.5'])}, but always predicting “no outage” also scores {P(a.always_no_outage_baseline)} because outages start in only {P(1-a.always_no_outage_baseline)} of home-hours, so we report lift and ROC-AUC instead. Most outages (planned works, equipment faults) are not weather-driven; the model highlights the {P(f.share_of_hours)} of hours where risk is genuinely higher.</p></Panel>}

export function TodayPanel(){const {d,err}=useJson('/api/predict/today')
 const lstm=d?.outage_source==='lstm'
 return <Panel title="Today: power-outage forecast & load temperature" right={<Updated d={d}/>}>
  {!d?(err?<p className="text-sm text-mute">Unavailable: {err}. No forecast is shown without real weather data.</p>:<Empty busy text="Loading forecast…"/>):<>
   <div className="grid gap-4 md:grid-cols-2">
    <div><div className="lbl mb-1">Fault (power outage) expected {lstm?'· LSTM model':'· weather rules (model not loaded)'}</div>
     {d.outage_windows.length?d.outage_windows.map((w:any)=><div key={w.start} className="mb-1">
       <div className="text-lg font-semibold" style={{color:RISK[w.level]}}>{w.start}–{w.end}{why(w.reasons)}</div>
       {lstm&&<div className="lbl">{w.level} risk · {pct(w.peak_p)} chance of an outage starting per home-hour (usual {pct(w.typical_p)})</div>}</div>)
      :<div className="text-lg" style={{color:SC.normal}}>No elevated outage risk for the rest of today</div>}</div>
    <div><div className="lbl mb-1">Load predicted above {d.threshold_c} °C (ML)</div>
     {d.load_over_threshold_windows.length?d.load_over_threshold_windows.map((w:any)=><div key={w.start} className="text-lg" style={{color:SC.warning}}>{w.start}–{w.end}</div>)
      :<div className="text-lg" style={{color:SC.normal}}>{d.hours.some((h:any)=>h.load_temp_c!=null)?'Not expected today':'Prediction unavailable'}</div>}</div></div>
   <div className="mt-3 flex gap-1 overflow-x-auto pb-1">{d.hours.map((h:any)=><div key={h.time} className="min-w-[60px] flex-1 rounded-lg bg-[#0a1424] px-1 py-2 text-center text-xs" title={h.reasons.join(', ')||undefined}>
    <div className="text-mute">{h.time}</div>
    <div className="mx-auto my-1 h-1.5 w-6 rounded-full" style={{background:RISK[h.outage_risk]}}/>
    {lstm&&<div style={{color:RISK[h.outage_risk]}}>{pct(h.outage_p)}</div>}
    <div>{h.ambient_c!=null?`${Math.round(h.ambient_c)}°`:'—'}{h.precip_mm>0?` · ${h.precip_mm}mm`:''}</div>
    <div className="num mt-0.5 text-sm" style={{color:h.load_over_threshold?SC.warning:undefined}}>{h.load_temp_c!=null?`${h.load_temp_c.toFixed(1)}°`:'—'}</div></div>)}</div>
   <p className="lbl mt-2">{lstm?`Per hour: outage risk (colour) and the LSTM probability that an outage starts at a typical Bengaluru home, from forecast rain, temperature, wind and time of day; trained on real Bengaluru outages 2014–2018 (ESMI). Amber = elevated: probability ≥ ${pct(d.outage_model?.elevated_threshold)}; in the unseen 2018 data, such hours had about 2× more outages than the rest (model ROC-AUC ${d.outage_model?.test_roc_auc}). `:'Outage risk from weather rules. '}Bottom: ML-predicted load temperature (low confidence).</p>
   <LastOutage/></>}</Panel>}

export function WeekPanel(){const {d,err}=useJson('/api/predict/weekly-outlook')
 return <Panel title="Next 7 days (ML + weather)" right={<Updated d={d}/>}>
  {!d?(err?<p className="text-sm text-mute">Unavailable: {err}.</p>:<Empty busy text="Loading forecast…"/>):<>
   <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">{d.days.map((x:any)=>{const hot=x.indicator==='elevated',w=x.outage_window
    return <div key={x.date} className="rounded-xl bg-[#0a1424] px-2 py-3 text-center" title={[x.note,...(x.outage_reasons||[])].filter(Boolean).join(' · ')||undefined}>
     <div className="text-sm font-semibold">{x.weekday}</div><div className="lbl">{x.date.slice(5)}</div>
     <div className="mt-2 text-sm">{x.ambient_max_c!=null?`${Math.round(x.ambient_max_c)}°`:'—'}<span className="text-mute"> / {x.ambient_min_c!=null?`${Math.round(x.ambient_min_c)}°`:'—'}</span></div>
     <div className="num mt-1 text-lg" style={{color:hot?SC.warning:undefined}}>{x.equipment_max_c!=null?`${x.equipment_max_c.toFixed(1)}°`:'—'}</div>
     <div className="mt-1"><Tag color={RISK[x.outage_risk||'unknown']}>{x.outage_risk==='low'||x.outage_risk==='normal'?'outage risk normal':x.outage_risk==='unknown'?'risk n/a':`outage ${x.outage_risk}`}</Tag></div>
     {w&&<div className="mt-1 text-xs" style={{color:RISK[w.level]}}>{w.start}–{w.end}{why(w.reasons.filter((x:string)=>!x.startsWith('the usual')))}</div>}</div>})}</div>
   <p className="lbl mt-2">Ambient max / min (forecast) · predicted load peak (ML) · outage risk and the riskiest window from the outage LSTM. Hover a day for details.</p></>}</Panel>}

export function ForecastPage(){return <div className="grid gap-4"><TodayPanel/><WeekPanel/><ModelCard/></div>}
