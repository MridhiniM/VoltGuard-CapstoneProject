import {ReactNode,useMemo,useState} from 'react'
import {Area,AreaChart,CartesianGrid,Line,LineChart,ReferenceLine,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts'
import {Ev,Lim,M,META,Metric,Reading,Status} from './core'
export const SC:Record<Status,string>={normal:'#34d399',warning:'#facc15',fault:'#f87171',offline:'#8296b3'}
export const SL:Record<Status,string>={normal:'Normal',warning:'Warning',fault:'Fault detected',offline:'Offline'}
export const fmtT=(t:number)=>new Date(t).toLocaleTimeString([],{hour12:false})
export const fmtD=(t:number)=>new Date(t).toLocaleString([],{hour12:false})
export const Panel=({title,right,children,className=''}:{title?:string;right?:ReactNode;children:ReactNode;className?:string})=>
 <section className={`panel ${className}`}>{title&&<header className="mb-3 flex items-center justify-between gap-2"><h3 className="text-[15px] font-semibold">{title}</h3>{right}</header>}{children}</section>
export const Tag=({children,color='#38bdf8'}:{children:ReactNode;color?:string})=><span className="tag" style={{color,borderColor:color}}>{children}</span>
export const SimTag=()=><Tag>SIMULATION</Tag>
export const Empty=({text,busy}:{text:string;busy?:boolean})=>
 <div className="grid place-items-center gap-2 py-8 text-center text-sm text-mute">{busy&&<div className="h-5 w-5 animate-spin rounded-full border-2 border-ln border-t-acc"/>}{text}</div>
/** Segmented range meter: lit segments show the value, the lighter band is the allowed range. */
export function Meter({v,lim,color}:{v:number|undefined;lim:Lim;color:string}){
 const sp=lim.max-lim.min,lo=lim.min-sp*.3,hi=lim.max+sp*.3,N=36
 const pos=v==null?-1:Math.max(0,Math.min(1,(v-lo)/(hi-lo)))
 return <div className="flex h-3 gap-[2px]" aria-hidden>{Array.from({length:N},(_,i)=>{const f=(i+.5)/N,x=lo+f*(hi-lo),inb=x>=lim.min&&x<=lim.max,lit=f<=pos
  return <i key={i} className="flex-1 rounded-[1px]" style={{background:lit?(inb?color:'#f87171'):inb?'#1d3050':'#2a1c26'}}/>})}</div>}
export function Spark({d,m}:{d:Reading[];m:Metric}){
 return <div className="h-10 w-full"><ResponsiveContainer><AreaChart data={d.slice(-40)}><YAxis hide domain={['auto','auto']}/>
  <Area dataKey={m} stroke={META[m].color} fill={META[m].color} fillOpacity={.12} strokeWidth={1.5} dot={false} isAnimationActive={false}/></AreaChart></ResponsiveContainer></div>}
const SEV={info:'#22d3ee',warning:SC.warning,critical:SC.fault}
const KIND={limit:'Limit',prediction:'Prediction',anomaly:'Anomaly',disconnect:'Connection',restore:'Connection'}
export function EventRow({e}:{e:Ev}){return <li className="grid grid-cols-[auto_1fr] gap-x-3 border-t border-ln py-2 first:border-0 text-sm">
 <i className="mt-1.5 h-2 w-2 rounded-full" style={{background:SEV[e.sev]}}/>
 <div><div className="flex flex-wrap items-center gap-2"><span className="font-medium">{KIND[e.kind]}</span><span className="lbl">{e.sev} · {fmtD(e.ts)}</span>{e.sim&&<SimTag/>}</div><div className="text-[#cbd5e1]">{e.text}</div></div></li>}
export function EventList({list,max=8}:{list:Ev[];max?:number}){
 return list.length?<ul className="max-h-72 overflow-y-auto pr-1">{list.slice(0,max).map(e=><EventRow key={e.id} e={e}/>)}</ul>:<Empty text="No events yet. Limit breaches, predictions and connection changes appear here."/>}
export const RG:Record<string,number>={'1m':6e4,'5m':3e5,'15m':9e5,'1h':36e5,'24h':864e5}
export function Range({v,set}:{v:string;set:(s:string)=>void}){
 return <div className="flex rounded-full border border-ln p-0.5" role="group" aria-label="Time range">{Object.keys(RG).map(k=>
  <button key={k} onClick={()=>set(k)} aria-pressed={v===k} className={`rounded-full px-3 py-1 text-xs ${v===k?'bg-acc font-semibold text-ink':'text-mute hover:text-white'}`}>{k}</button>)}</div>}
/** Average into time bins; empty bins stay null so missing data shows as a gap. */
function bin(d:Reading[],from:number,to:number){
 const pts=d.filter(r=>r.ts>=from&&r.ts<=to);if(!pts.length)return[]
 const dts=pts.slice(1,60).map((r,i)=>r.ts-pts[i].ts).sort((a,b)=>a-b),med=dts[dts.length>>1]||1000
 const w=Math.max((to-from)/240,med*2),n=Math.ceil((to-from)/w),acc=Array.from({length:n},()=>[0,0,0,0])
 for(const r of pts){const a=acc[Math.min(n-1,Math.floor((r.ts-from)/w))];a[0]+=r.voltage;a[1]+=r.current;a[2]+=r.temperature;a[3]++}
 return acc.map((a,i)=>({ts:from+i*w+w/2,voltage:a[3]?a[0]/a[3]:null,current:a[3]?a[1]/a[3]:null,temperature:a[3]?a[2]/a[3]:null}))}
export function TrendChart({data,from,to,marks=[],h=300}:{data:Reading[];from:number;to:number;marks?:number[];h?:number}){
 const [off,setOff]=useState<Metric[]>([])
 const rows=useMemo(()=>bin(data,from,to),[data,from,to])
 const tick=(t:number)=>new Date(t).toLocaleTimeString([],{hour12:false,hour:'2-digit',minute:'2-digit'})
 return <div><div className="mb-2 flex flex-wrap gap-2">{M.map(m=>
  <button key={m} className="chip" aria-pressed={!off.includes(m)} style={{opacity:off.includes(m)?.4:1}} onClick={()=>setOff(o=>o.includes(m)?o.filter(x=>x!==m):[...o,m])}>
   <i className="inline-block h-[3px] w-4" style={{background:META[m].color}}/>{META[m].label} ({META[m].unit})</button>)}</div>
  {rows.length?<div style={{height:h}}><ResponsiveContainer><LineChart data={rows} margin={{top:6,right:0,left:0,bottom:0}}>
   <CartesianGrid stroke="#17263d" vertical={false}/>
   <XAxis dataKey="ts" type="number" scale="time" domain={[from,to]} tickFormatter={tick} tick={{fontSize:11,fill:'#8296b3'}} tickLine={false} axisLine={{stroke:'#1c2c46'}} minTickGap={50}/>
   {M.map(m=><YAxis key={m} yAxisId={m} orientation={m==='voltage'?'left':'right'} hide={off.includes(m)} width={44} domain={['auto','auto']} tickFormatter={(v:number)=>v.toFixed(m==='voltage'?0:1)} tick={{fontSize:11,fill:META[m].color}} tickLine={false} axisLine={false}/>)}
   <Tooltip contentStyle={{background:'#08111f',border:'1px solid #1c2c46',borderRadius:10,fontSize:12}} labelFormatter={t=>fmtD(Number(t))}
    formatter={(v:any,n:any)=>[v==null?'—':Number(v).toFixed(META[n as Metric].dp)+' '+META[n as Metric].unit,META[n as Metric].label]}/>
   {marks.slice(0,80).map((t,i)=><ReferenceLine key={i} x={t} yAxisId="voltage" stroke="#f87171" strokeDasharray="3 3" strokeOpacity={.7}/>)}
   {M.map(m=><Line key={m} yAxisId={m} dataKey={m} stroke={META[m].color} strokeWidth={1.8} dot={false} hide={off.includes(m)} connectNulls={false} isAnimationActive={false}/>)}
  </LineChart></ResponsiveContainer></div>:<Empty text="No readings in this window yet."/>}</div>}
