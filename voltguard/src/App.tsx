import {useState} from 'react'
import {Activity,CloudSun,History,LayoutDashboard,Settings as Cog,ShieldAlert,Zap,Home as HomeIcon,Radio} from 'lucide-react'
import {Provider,useApp} from './store'
import {Faults,HistoryPage,Live,Overview,SettingsPage} from './pages'
import {Homes,LiveHome} from './homes'
import {ForecastPage} from './forecast'
import {SC,SimTag,Tag,fmtD} from './ui'
const NAV=[['Overview',LayoutDashboard,Overview],['Live monitoring',Activity,Live],['Fault analysis',ShieldAlert,Faults],['Forecast (ML)',CloudSun,ForecastPage],['Homes (10 nodes)',HomeIcon,Homes],['Live home',Radio,LiveHome],['Historical data',History,HistoryPage],['System settings',Cog,SettingsPage]] as const
function Top(){const A=useApp();const sim=A.settings.mode==='sim'
 const c=sim?'#38bdf8':A.conn==='connected'?SC.normal:A.conn==='connecting'?SC.warning:SC.fault
 return <header className="sticky top-0 z-20 flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-ln bg-ink/90 px-4 py-2.5 backdrop-blur">
  <div className="flex items-center gap-2 text-lg font-bold"><Zap size={20} color="#2dd4bf"/>VoltGuard</div>
  <span className="flex items-center gap-2 text-sm"><i className="h-2 w-2 rounded-full" style={{background:c}}/>{sim?'Local simulator':A.conn==='connected'?'Gateway / Node-RED connected':A.conn==='connecting'?'Connecting to Node-RED…':'Gateway / Node-RED unreachable'}</span>
  <span className="lbl hidden sm:inline">{fmtD(A.now)}</span><span className="flex-1"/>
  {sim?<SimTag/>:<Tag color="#34d399">LIVE</Tag>}
  <button className="btn" onClick={A.switchMode}>{sim?'Switch to live':'Switch to simulation'}</button></header>}
function Shell(){const [i,setI]=useState(()=>{const q=new URLSearchParams(location.search).get('page')?.toLowerCase();const k=q?NAV.findIndex(n=>n[0].toLowerCase().startsWith(q)):0;return k<0?0:k}) // ?page=forecast opens a page directly
 const Page=NAV[i][2]
 return <div className="min-h-screen md:pl-[92px]">
  <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-30 flex justify-around border-t border-ln bg-deep md:inset-y-0 md:right-auto md:w-[92px] md:flex-col md:justify-start md:gap-1 md:border-r md:border-t-0 md:pt-20">
   {NAV.map(([n,Icon],k)=><button key={n} onClick={()=>setI(k)} aria-current={i===k} className={`flex flex-1 flex-col items-center gap-1 px-1 py-2.5 text-[11px] leading-tight md:flex-none md:py-3 ${i===k?'text-acc md:border-l-[3px] md:border-acc':'text-mute hover:text-white md:border-l-[3px] md:border-transparent'}`}><Icon size={20}/><span className="text-center">{n}</span></button>)}</nav>
  <Top/><main className="mx-auto max-w-[1500px] p-4 pb-24 md:pb-6"><h1 className="mb-4 text-2xl font-semibold">{NAV[i][0]}</h1><Page/></main></div>}
export default function App(){return <Provider><Shell/></Provider>}
