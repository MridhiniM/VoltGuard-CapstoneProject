export type Status='normal'|'warning'|'fault'|'offline'
export type Metric='voltage'|'current'|'temperature'
export type Conn='connected'|'connecting'|'offline'
export interface Lim{min:number;max:number}
export interface Prediction{label:string;confidence?:number}
export interface Anomaly{state:string;score?:number;threshold?:number;isAnomaly?:boolean;alert?:boolean;window?:number;needed?:number;reason?:string}
export interface Reading{ts:number;voltage:number;current:number;temperature:number;status:Status;breaches:Metric[];prediction?:Prediction;anomaly?:Anomaly;sim:boolean}
export interface Ev{id:number;ts:number;kind:'limit'|'prediction'|'anomaly'|'disconnect'|'restore';sev:'info'|'warning'|'critical';text:string;sim:boolean}
export interface Settings{mode:'sim'|'live';transport:'http'|'ws';baseUrl:string;latestPath:string;wsUrl:string;pollMs:number;predictUrl:string;labelKey:string;confKey:string;simMs:number;historyLen:number;limits:Record<Metric,Lim>}
export const M:Metric[]=['voltage','current','temperature']
export const META:Record<Metric,{label:string;unit:string;color:string;dp:number}>={
 voltage:{label:'Voltage',unit:'V',color:'#22d3ee',dp:1},
 current:{label:'Current',unit:'A',color:'#60a5fa',dp:2},
 temperature:{label:'Temperature',unit:'°C',color:'#a3e635',dp:1}}
export const DEFAULTS:Settings={mode:'live',transport:'http',baseUrl:import.meta.env.VITE_NODERED_URL||'http://localhost:5000',latestPath:'/api/readings/latest',wsUrl:'ws://localhost:1880/ws/readings',pollMs:2000,predictUrl:'',labelKey:'label',confKey:'confidence',simMs:1000,historyLen:5000,
 limits:{voltage:{min:207,max:253},current:{min:0,max:5},temperature:{min:15,max:30}}}
let eid=0;export const nid=()=>++eid
const KEY='voltguard.settings.v2' // non-secret settings only
export function loadSettings():Settings{try{const s=JSON.parse(localStorage.getItem(KEY)||'{}');return{...DEFAULTS,...s,limits:{...DEFAULTS.limits,...(s.limits||{})}}}catch{return DEFAULTS}}
export const saveSettings=(s:Settings)=>{try{localStorage.setItem(KEY,JSON.stringify(s))}catch{}}
export function validate(s:Settings):string[]{const e:string[]=[];const ok=(x:number,a:number,b:number)=>Number.isFinite(x)&&x>=a&&x<=b
 for(const m of M){const l=s.limits[m];if(!Number.isFinite(l.min)||!Number.isFinite(l.max))e.push(`${META[m].label}: enter both limits`);else if(l.min>=l.max)e.push(`${META[m].label}: minimum must be below maximum`)}
 if(!ok(s.simMs,200,60000))e.push('Simulation interval must be 200–60000 ms')
 if(!ok(s.pollMs,500,60000))e.push('Polling interval must be 500–60000 ms')
 if(!ok(s.historyLen,100,100000))e.push('Chart history must be 100–100000 readings')
 if(!/^https?:\/\/.+/.test(s.baseUrl))e.push('Node-RED base URL must start with http:// or https://')
 if(!/^wss?:\/\/.+/.test(s.wsUrl))e.push('WebSocket URL must start with ws:// or wss://')
 if(s.predictUrl&&!/^https?:\/\/.+/.test(s.predictUrl))e.push('Prediction URL must start with http:// or https://')
 if(!s.latestPath.startsWith('/'))e.push('Latest-readings path must start with /')
 return e}
/** Threshold check. Beyond 15% of the limit span outside the range = critical. */
export function evaluate(r:Record<Metric,number>,lim:Settings['limits'],pred?:Prediction){
 const breaches:Metric[]=[];let crit=false
 for(const m of M){const{min,max}=lim[m],x=r[m],sp=max-min;if(x<min||x>max){breaches.push(m);if(x<min-sp*.15||x>max+sp*.15)crit=true}}
 const bad=!!pred&&pred.label.toLowerCase()!=='normal'
 const status:Status=crit?'fault':breaches.length?'warning':bad?'warning':'normal'
 return{breaches,status}}
/** Validate + normalize an untrusted payload. Returns null when unusable. */
export function normalize(raw:any,now:number,lim:Settings['limits'],labelKey='label',confKey='confidence'):Reading|null{
 if(!raw||typeof raw!=='object')return null
 const n=(x:any)=>{const v=typeof x==='string'&&x.trim()!==''?Number(x):x;return typeof v==='number'&&Number.isFinite(v)?v:null}
 const voltage=n(raw.voltage),current=n(raw.current),temperature=n(raw.temperature)
 if(voltage==null||current==null||temperature==null)return null
 let ts=now // missing/malformed timestamp -> arrival time
 if(raw.timestamp!=null){const p=typeof raw.timestamp==='number'?raw.timestamp:Date.parse(raw.timestamp);if(Number.isFinite(p))ts=p<1e12?p*1000:p}
 let prediction:Prediction|undefined;const p=raw.prediction
 if(p&&typeof p==='object'&&typeof p[labelKey]==='string'){const c=n(p[confKey]);prediction={label:p[labelKey],confidence:c!=null&&c>=0&&c<=1?c:undefined}}
 let anomaly:Anomaly|undefined;const a=raw.anomaly
 if(a&&typeof a==='object'&&typeof a.state==='string'){const o=(x:any)=>typeof x==='number'&&Number.isFinite(x)?x:undefined
  anomaly={state:a.state,score:o(a.score),threshold:o(a.threshold),isAnomaly:typeof a.is_anomaly==='boolean'?a.is_anomaly:undefined,alert:typeof a.alert==='boolean'?a.alert:undefined,window:o(a.window),needed:o(a.needed),reason:typeof a.reason==='string'?a.reason:undefined}}
 const ev=evaluate({voltage,current,temperature},lim,prediction)
 if(anomaly?.alert&&ev.status==='normal')ev.status='warning' // advisory: anomaly alone is never a confirmed fault
 return{ts,voltage,current,temperature,...ev,prediction,anomaly,sim:false}}
export function eventsFor(p:Reading|undefined,r:Reading):Ev[]{
 const out:Ev[]=[],ps=p?.status??'normal'
 const mk=(kind:Ev['kind'],sev:Ev['sev'],text:string):Ev=>({id:nid(),ts:r.ts,kind,sev,text,sim:r.sim})
 if(r.breaches.length&&(r.breaches.join()!==(p?.breaches.join()??'')||ps==='normal'))
  out.push(mk('limit',r.status==='fault'?'critical':'warning',r.breaches.map(m=>`${META[m].label} ${r[m].toFixed(META[m].dp)} ${META[m].unit} outside limits`).join('; ')))
 const pl=r.prediction?.label
 if(pl&&pl!==p?.prediction?.label&&pl.toLowerCase()!=='normal')out.push(mk('prediction','warning',`ML prediction (unverified): ${pl}`))
 if(r.anomaly?.alert&&!p?.anomaly?.alert)out.push(mk('anomaly','warning',`LSTM anomaly detector: readings unlike normal behaviour (score ${r.anomaly.score?.toFixed(3)} vs threshold ${r.anomaly.threshold?.toFixed(3)}; advisory)`))
 if(!r.anomaly?.alert&&p?.anomaly?.alert)out.push(mk('anomaly','info','LSTM anomaly detector: readings back to normal'))
 if(!r.breaches.length&&p?.breaches.length)out.push(mk('limit','info','Readings back within limits'))
 return out}
/** Deterministic simulator (seeded). Anomalies are injected, never real. */
export function makeSim(seed:number){
 let a=seed>>>0,k=0,tk=0,force:{kind:number;fault:boolean;left:number}|null=null
 const o=[0,0,0],AMP=[[27,45],[3.6,6],[21,35]]
 const rnd=()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}
 const LB=['overvoltage','overcurrent','overtemperature']
 const step=(ts:number,lim:Settings['limits']):Reading=>{
  k++
  if(!force&&rnd()<.004)force={kind:Math.floor(rnd()*3),fault:rnd()<.4,left:15+Math.floor(rnd()*20)}
  for(let i=0;i<3;i++){const tg=force&&force.kind===i?AMP[i][force.fault?1:0]:0;o[i]+=(tg-o[i])*.3}
  if(force&&--force.left<=0)force=null
  const voltage=+(230+2*Math.sin(k/40)+(rnd()-.5)*1.6+o[0]).toFixed(2)
  const current=+Math.max(0,1.8+.25*Math.sin(k/25)+(rnd()-.5)*.1+o[1]).toFixed(3)
  const temperature=+(21+2*Math.sin(k/90)+(rnd()-.5)*.4+o[2]).toFixed(2)
  const e=evaluate({voltage,current,temperature},lim)
  const prediction:Prediction={label:e.status==='fault'?LB[Math.max(0,M.indexOf(e.breaches[0]))]:'normal'}
  return{ts,voltage,current,temperature,...e,prediction,sim:true}}
 return{step,trigger:(fault:boolean)=>{force={kind:tk++%3,fault,left:25}},
  backfill:(end:number,n:number,stepMs:number,lim:Settings['limits'])=>Array.from({length:n},(_,i)=>step(end-(n-1-i)*stepMs,lim))}}
/** Node-RED transport: HTTP polling or WebSocket with exponential backoff. Returns a stop function. */
export function startLive(s:Settings,onData:(raw:unknown)=>void,onConn:(c:Conn,err?:string)=>void):()=>void{
 let stop=false,timer=0
 if(s.transport==='http'){
  const poll=async()=>{const ac=new AbortController(),to=setTimeout(()=>ac.abort(),Math.min(s.pollMs*2,8000))
   try{const r=await fetch(s.baseUrl.replace(/\/$/,'')+s.latestPath,{signal:ac.signal,cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status)
    const d=await r.json();if(!stop){onConn('connected');onData(d)}}
   catch(e:any){if(!stop)onConn('offline',e?.name==='AbortError'?'Request timed out':String(e?.message||e))}
   finally{clearTimeout(to);if(!stop)timer=window.setTimeout(poll,s.pollMs)}}
  onConn('connecting');poll()
  return()=>{stop=true;clearTimeout(timer)}}
 let ws:WebSocket|null=null,retry=0
 const open=()=>{if(stop||ws)return
  onConn('connecting');const w=new WebSocket(s.wsUrl);ws=w
  w.onopen=()=>{retry=0;onConn('connected')}
  w.onmessage=e=>{try{const d=JSON.parse(String(e.data));(Array.isArray(d)?d:[d]).forEach(onData)}catch{onData(null)}}
  w.onerror=()=>w.close()
  w.onclose=()=>{ws=null;if(stop)return;onConn('offline','Connection closed; retrying');timer=window.setTimeout(open,Math.min(30000,1000*2**retry++))}}
 open()
 return()=>{stop=true;clearTimeout(timer);if(ws){ws.onclose=null;ws.close();ws=null}}}
export async function predict(s:Settings,r:Reading):Promise<Prediction|undefined>{
 if(!s.predictUrl)return
 try{const res=await fetch(s.predictUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voltage:r.voltage,current:r.current,temperature:r.temperature})})
  const d=await res.json();const l=d?.[s.labelKey];if(typeof l!=='string')return
  const c=d[s.confKey]==null?NaN:Number(d[s.confKey]);return{label:l,confidence:Number.isFinite(c)&&c>=0&&c<=1?c:undefined}}catch{return}}
export async function testConn(s:Settings):Promise<{ok:boolean;msg:string}>{
 if(s.transport==='http'){try{const ac=new AbortController(),t=setTimeout(()=>ac.abort(),5000)
  const r=await fetch(s.baseUrl.replace(/\/$/,'')+s.latestPath,{signal:ac.signal});clearTimeout(t)
  if(!r.ok)return{ok:false,msg:'HTTP '+r.status}
  return normalize(await r.json(),Date.now(),s.limits)?{ok:true,msg:'Endpoint reachable and payload valid'}:{ok:false,msg:'Reachable, but payload lacks numeric voltage, current and temperature'}}
  catch(e:any){return{ok:false,msg:e?.name==='AbortError'?'Timed out after 5 s':'Unreachable (check URL, Node-RED running, CORS): '+(e?.message||e)}}}
 return new Promise(res=>{let w:WebSocket|undefined;const t=setTimeout(()=>{w?.close();res({ok:false,msg:'Timed out after 5 s'})},5000)
  try{w=new WebSocket(s.wsUrl)}catch(e:any){clearTimeout(t);return res({ok:false,msg:String(e?.message||e)})}
  w.onopen=()=>{clearTimeout(t);w?.close();res({ok:true,msg:'WebSocket opened (payload format is checked once data flows)'})}
  w.onerror=()=>{clearTimeout(t);res({ok:false,msg:'Could not open WebSocket'})}})}
