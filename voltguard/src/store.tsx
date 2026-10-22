import {createContext,useContext,useEffect,useRef,useState,ReactNode} from 'react'
import {Conn,Ev,Reading,Settings,Status,eventsFor,evaluate,loadSettings,makeSim,nid,normalize,predict,saveSettings,startLive} from './core'
function useProvider(){
 const [settings,setS]=useState<Settings>(loadSettings)
 const [readings,setReadings]=useState<Reading[]>([])
 const [events,setEvents]=useState<Ev[]>([])
 const [conn,setConn]=useState<Conn>('connecting')
 const [err,setErr]=useState('')
 const [rejected,setRej]=useState(0)
 const [now,setNow]=useState(Date.now())
 const [snap,setSnap]=useState<Reading[]|null>(null)
 const sim=useRef(makeSim(42)),sRef=useRef(settings),last=useRef<Reading>()
 sRef.current=settings
 useEffect(()=>{const t=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(t)},[])
 const addEv=(kind:Ev['kind'],sev:Ev['sev'],text:string)=>setEvents(p=>[{id:nid(),ts:Date.now(),kind,sev,text,sim:false},...p].slice(0,300))
 const push=(r:Reading)=>{if(last.current&&r.ts<=last.current.ts)return // ignore repeats of the same reading
  const ev=eventsFor(last.current,r);last.current=r
  setReadings(p=>{const n=[...p,r];return n.length>sRef.current.historyLen?n.slice(-sRef.current.historyLen):n})
  if(ev.length)setEvents(p=>[...ev.reverse(),...p].slice(0,300))}
 const s0=settings
 useEffect(()=>{
  const s=sRef.current;last.current=undefined;setSnap(null);setReadings([]);setEvents([]);setErr('');setRej(0)
  if(s.mode==='sim'){
   const g=makeSim(42);sim.current=g
   const seed=g.backfill(Date.now(),2880,30000,s.limits);let p:Reading|undefined,ev:Ev[]=[]
   for(const r of seed){ev=[...eventsFor(p,r).reverse(),...ev];p=r}
   last.current=p;setReadings(seed);setEvents(ev.slice(0,300));setConn('connected')
   const t=setInterval(()=>push(g.step(Date.now(),sRef.current.limits)),s.simMs)
   return()=>clearInterval(t)}
  setConn('connecting');let prev:Conn='connecting'
  return startLive(s,async raw=>{const c=sRef.current,r=normalize(raw,Date.now(),c.limits,c.labelKey,c.confKey)
    if(!r){setRej(n=>n+1);return}
    if(!r.prediction&&c.predictUrl){const p=await predict(c,r);if(p){r.prediction=p;Object.assign(r,evaluate(r,c.limits,p))}}
    push(r)},
   (c,e)=>{setConn(c);setErr(e||'')
    if(c==='offline'&&prev!=='offline')addEv('disconnect','warning','Connection to Node-RED lost'+(e?` (${e})`:''))
    if(c==='connected'&&prev==='offline')addEv('restore','info','Connection to Node-RED restored');prev=c})
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[s0.mode,s0.transport,s0.baseUrl,s0.latestPath,s0.wsUrl,s0.pollMs,s0.simMs])
 const latest=readings.at(-1),age=latest?now-latest.ts:Infinity
 const stale=age>Math.max(5000,(settings.mode==='sim'?settings.simMs:settings.pollMs)*3)
 const status:Status=settings.mode==='live'&&(conn!=='connected'||!latest||stale)?'offline':latest?.status??'offline'
 return{settings,readings,events,conn,err,rejected,now,latest,age,stale,status,view:snap??readings,paused:snap!==null,
  togglePause:()=>setSnap(snap?null:readings),
  save:(s:Settings)=>{saveSettings(s);setS(s)},
  switchMode:()=>{const to=settings.mode==='sim'?'live':'sim'
   const msg=to==='live'?'Switch to live data from Node-RED?\n\nSimulated data will be cleared. If Node-RED is unreachable the dashboard will show Offline — it will not fall back to simulation.':'Switch to simulation?\n\nLive readings in this browser will be cleared and all data will be simulated.'
   if(window.confirm(msg)){const n={...settings,mode:to} as Settings;saveSettings(n);setS(n)}},
  trigger:(fault:boolean)=>sim.current.trigger(fault)}}
type Ctx=ReturnType<typeof useProvider>
const Cx=createContext<Ctx|null>(null)
export const Provider=({children}:{children:ReactNode})=><Cx.Provider value={useProvider()}>{children}</Cx.Provider>
export const useApp=()=>useContext(Cx) as Ctx
