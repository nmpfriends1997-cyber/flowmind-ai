import React, { useEffect, useRef, useState, useCallback } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import {
  fetchHeatmap, fetchRecentEvents,
  fetchLiveSnapshot, fetchApiConfigStatus
} from '../lib/api'
import { Panel, PanelHeader, RiskBadge, Spinner } from '../components/UI'
import { RefreshCw, Wifi, WifiOff, Database, Radio, AlertTriangle, Navigation } from 'lucide-react'

const CAUSE_COLORS: Record<string,string> = {
  vehicle_breakdown:'#3B82F6', accident:'#EF4444', public_event:'#8B5CF6',
  construction:'#F97316', water_logging:'#06B6D4', pot_holes:'#F59E0B',
  tree_fall:'#10B981', congestion:'#EC4899', others:'#94A3B8',
  procession:'#A78BFA', vip_movement:'#FB7185', protest:'#FCD34D',
}

const CONGESTION_COLOR = (pct: number) =>
  pct >= 70 ? '#EF4444' : pct >= 45 ? '#F97316' : pct >= 20 ? '#F59E0B' : '#10B981'

const SEV_COLOR: Record<string,string> = {
  Critical:'#EF4444', Major:'#F97316', Moderate:'#F59E0B', Minor:'#10B981', Unknown:'#94A3B8'
}

type DataMode = 'historical' | 'live'
type MapView  = 'heatmap' | 'markers' | 'traffic' | 'incidents' | 'events'

export default function CityMap() {
  const mapRef         = useRef<HTMLDivElement>(null)
  const mapInstance    = useRef<any>(null)
  const layerGroup     = useRef<any>(null)
  const heatLayer      = useRef<any>(null)

  const [mode, setMode]           = useState<DataMode>('live')
  const [view, setView]           = useState<MapView>('incidents')
  const [loading, setLoading]     = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [apiStatus, setApiStatus] = useState<any>(null)

  // historical
  const [histPoints, setHistPoints]   = useState<any[]>([])
  const [histEvents, setHistEvents]   = useState<any[]>([])

  // live
  const [liveIncidents, setLiveIncidents] = useState<any[]>([])
  const [liveTraffic,   setLiveTraffic]   = useState<any[]>([])
  const [liveEvents,    setLiveEvents]    = useState<any[]>([])
  const [liveSummary,   setLiveSummary]   = useState<any>(null)

  const [selectedItem, setSelectedItem] = useState<any>(null)
  const [filter, setFilter]             = useState('all')

  // ── Init map ──────────────────────────────────────────────────────────────
  // L is imported directly as an ES module (see top of file), so there's no
  // "is window.L ready yet" race — it's available the moment this module loads.
  useEffect(() => {
    if (mapInstance.current || !mapRef.current) return
    const map = L.map(mapRef.current, { center: [12.9716, 77.5946], zoom: 12, zoomControl: true })
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '©OpenStreetMap ©CartoDB', subdomains: 'abcd', maxZoom: 19,
    }).addTo(map)
    layerGroup.current = L.layerGroup().addTo(map)
    mapInstance.current = map
    setTimeout(() => map.invalidateSize(), 0)

    return () => {
      map.remove()
      mapInstance.current = null
    }
  }, [])

  // Re-measure the map whenever the loading skeleton is removed, in case the
  // container's size changed while it was hidden.
  useEffect(() => {
    if (!loading && mapInstance.current) {
      setTimeout(() => mapInstance.current.invalidateSize(), 0)
    }
  }, [loading])

  // ── Load data ──────────────────────────────────────────────────────────────
  const loadHistorical = useCallback(async () => {
    const [pts, evs] = await Promise.all([fetchHeatmap(600), fetchRecentEvents(60)])
    setHistPoints(pts); setHistEvents(evs)
  }, [])

  const loadLive = useCallback(async () => {
    setRefreshing(true)
    try {
      const snap = await fetchLiveSnapshot()
      setLiveIncidents(snap.incidents || [])
      setLiveTraffic(snap.corridors || [])
      setLiveEvents(snap.events || [])
      setLiveSummary(snap.summary || null)
      setLastUpdate(new Date().toLocaleTimeString('en-IN', { hour12: false }))
    } catch(e) { console.error(e) }
    setRefreshing(false)
  }, [])

  useEffect(() => {
    setLoading(true)
    Promise.all([
      fetchApiConfigStatus().then(setApiStatus).catch(() => {}),
      loadHistorical(),
      loadLive(),
    ]).finally(() => setLoading(false))
  }, [])

  // Auto-refresh live every 60s
  useEffect(() => {
    if (mode !== 'live') return
    const id = setInterval(loadLive, 60000)
    return () => clearInterval(id)
  }, [mode, loadLive])

  // ── Render map layers ──────────────────────────────────────────────────────
  useEffect(() => {
    const map = mapInstance.current
    if (!map || loading) return

    // Clear
    layerGroup.current.clearLayers()
    if (heatLayer.current) { map.removeLayer(heatLayer.current); heatLayer.current = null }

    if (mode === 'historical') {
      const filtered = filter === 'all' ? histPoints : histPoints.filter(p => p.cause === filter)
      if (view === 'heatmap' && (L as any).heatLayer) {
        heatLayer.current = (L as any).heatLayer(
          filtered.map(p => [p.lat, p.lng, p.weight]),
          { radius:22, blur:15, maxZoom:17, gradient:{'0.2':'#3B82F6','0.5':'#F59E0B','0.8':'#F97316','1.0':'#EF4444'} }
        ).addTo(map)
      } else {
        const toShow = filter === 'all' ? histEvents : histEvents.filter(e => e.event_cause === filter)
        toShow.slice(0, 200).forEach(e => {
          const color = CAUSE_COLORS[e.event_cause] || '#94A3B8'
          const icon = L.divIcon({
            html: `<div style="width:10px;height:10px;border-radius:50%;background:${color};border:2px solid rgba(255,255,255,0.3);box-shadow:0 0 6px ${color}80"></div>`,
            className: '', iconSize: [10,10],
          })
          L.marker([e.latitude, e.longitude], { icon })
            .bindPopup(`<b>${e.event_cause?.replace(/_/g,' ')}</b><br/>${e.police_station||e.zone||''}<br/><span style="color:${color}">${e.priority} · ${e.status}</span>`)
            .addTo(layerGroup.current)
        })
      }

    } else {
      // ── LIVE modes ────────────────────────────────────────────────────────
      if (view === 'incidents') {
        liveIncidents.forEach(inc => {
          const color = SEV_COLOR[inc.severity] || '#94A3B8'
          const size  = inc.magnitude >= 3 ? 14 : inc.magnitude >= 2 ? 11 : 8
          const icon = L.divIcon({
            html: `<div style="width:${size}px;height:${size}px;border-radius:50%;background:${color};border:2px solid rgba(255,255,255,0.5);box-shadow:0 0 8px ${color}90;animation:pulse-glow 2s infinite"></div>`,
            className: '', iconSize: [size,size],
          })
          L.marker([inc.latitude, inc.longitude], { icon })
            .on('click', () => setSelectedItem({ type:'incident', data:inc }))
            .bindTooltip(`<b>${inc.cause}</b><br/>${inc.road}<br/>${inc.delay_sec >= 60 ? `Delay: +${Math.round(inc.delay_sec/60)} min` : inc.description || 'Active incident'}`, { permanent:false })
            .addTo(layerGroup.current)
        })

      } else if (view === 'traffic') {
        liveTraffic.forEach(t => {
          const color = CONGESTION_COLOR(t.congestion_pct)
          const r = 14 + Math.round(t.congestion_pct / 10)
          const icon = L.divIcon({
            html: `<div style="width:${r*2}px;height:${r*2}px;border-radius:50%;background:${color}22;border:2px solid ${color};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:${color}">${t.congestion_pct}%</div>`,
            className: '', iconSize: [r*2, r*2], iconAnchor: [r, r],
          })
          L.marker([t.latitude, t.longitude], { icon })
            .on('click', () => setSelectedItem({ type:'traffic', data:t }))
            .bindTooltip(`<b>${t.corridor}</b><br/>Congestion: ${t.congestion_pct}%<br/>Delay: +${t.delay_min} min`, { permanent:false })
            .addTo(layerGroup.current)
        })

      } else if (view === 'events') {
        liveEvents.forEach(ev => {
          const color = ev.risk_level === 'High' ? '#EF4444' : ev.risk_level === 'Moderate' ? '#F59E0B' : '#10B981'
          const size  = ev.crowd_estimate > 10000 ? 16 : ev.crowd_estimate > 3000 ? 12 : 9
          const icon = L.divIcon({
            html: `<div style="width:${size}px;height:${size+3}px;clip-path:polygon(50% 0%,100% 40%,80% 100%,20% 100%,0% 40%);background:${color};border:2px solid white;box-shadow:0 0 8px ${color}80"></div>`,
            className: '', iconSize: [size, size+3], iconAnchor: [size/2, size+3],
          })
          L.marker([ev.latitude, ev.longitude], { icon })
            .on('click', () => setSelectedItem({ type:'event', data:ev }))
            .bindTooltip(`<b>${ev.name||ev.event_type}</b><br/>Crowd: ~${(ev.crowd_estimate||0).toLocaleString()}<br/>Risk: ${ev.risk_level}`, { permanent:false })
            .addTo(layerGroup.current)
        })

      } else if (view === 'heatmap' && (L as any).heatLayer) {
        // Live heatmap — blend incidents + congestion
        const pts = [
          ...liveIncidents.map(i => [i.latitude, i.longitude, i.magnitude / 4]),
          ...liveTraffic.map(t => [t.latitude, t.longitude, t.congestion_pct / 100]),
        ]
        heatLayer.current = (L as any).heatLayer(pts, {
          radius:28, blur:18, maxZoom:17,
          gradient:{'0.2':'#3B82F6','0.5':'#F59E0B','0.8':'#F97316','1.0':'#EF4444'}
        }).addTo(map)
      }
    }
  }, [mode, view, filter, histPoints, histEvents, liveIncidents, liveTraffic, liveEvents, loading])

  // ── Sidebar items ──────────────────────────────────────────────────────────
  const sidebarItems = mode === 'historical'
    ? histEvents.slice(0, 12)
    : view === 'incidents' ? liveIncidents.slice(0, 12)
    : view === 'traffic'   ? liveTraffic.slice(0, 12)
    : liveEvents.slice(0, 12)

  const historicalCauses = ['all', ...Array.from(new Set(histPoints.map(p => p.cause))).sort()]

  const LIVE_VIEWS: { key: MapView; label: string; icon: any }[] = [
    { key:'incidents', label:'Incidents', icon: AlertTriangle },
    { key:'traffic',   label:'Corridor Traffic', icon: Navigation },
    { key:'events',    label:'Live Events', icon: Radio },
    { key:'heatmap',   label:'Combined Heatmap', icon: Database },
  ]

  return (
    <div className="p-6 space-y-4 animate-slide-in">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-lg font-semibold" style={{ color:'var(--text-primary)' }}>City Map</h1>
          <p className="text-sm" style={{ color:'var(--text-secondary)' }}>Bengaluru traffic intelligence — real-time + historical</p>
        </div>

        {/* Mode toggle */}
        <div className="flex items-center gap-2 p-1 rounded-xl border" style={{ background:'var(--bg-panel)', borderColor:'var(--border)' }}>
          {(['historical','live'] as DataMode[]).map(m => (
            <button key={m} onClick={() => { setMode(m); setView(m === 'historical' ? 'heatmap' : 'incidents') }}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold cursor-pointer transition-all"
              style={{
                background: mode === m ? (m === 'live' ? '#10B981' : 'var(--blue)') : 'transparent',
                color: mode === m ? '#fff' : 'var(--text-muted)',
              }}>
              {m === 'live' ? <Wifi size={13}/> : <Database size={13}/>}
              {m === 'live' ? 'Live Data' : 'Historical Data'}
            </button>
          ))}
        </div>
      </div>

      {/* API key status banners — one per data source so the user knows exactly what is live vs ML */}
      {apiStatus && mode === 'live' && apiStatus.simulation_mode && (
        <div className="flex items-start gap-3 px-4 py-3 rounded-lg border text-xs"
          style={{ background:'rgba(245,158,11,0.08)', borderColor:'rgba(245,158,11,0.3)', color:'var(--text-secondary)' }}>
          <WifiOff size={14} style={{ color:'var(--amber)', flexShrink:0, marginTop:1 }} />
          <div>
            <span style={{ color:'var(--amber)', fontWeight:600 }}>Simulation Mode — </span>
            No API key detected. Incidents and corridor traffic are ML-predicted from historical Bengaluru data (not random).
            Add <code style={{ color:'var(--blue)' }}>TOMTOM_API_KEY</code> to <code>backend/.env</code> for real live data.
            Events are always real (OpenStreetMap).
          </div>
        </div>
      )}
      {apiStatus && mode === 'live' && !apiStatus.simulation_mode && apiStatus.flow_quota_exhausted && (
        <div className="flex items-start gap-3 px-4 py-3 rounded-lg border text-xs"
          style={{ background:'rgba(239,68,68,0.08)', borderColor:'rgba(239,68,68,0.3)', color:'var(--text-secondary)' }}>
          <WifiOff size={14} style={{ color:'var(--red)', flexShrink:0, marginTop:1 }} />
          <div>
            <span style={{ color:'var(--red)', fontWeight:600 }}>Corridor Quota Exhausted — </span>
            TomTom Routing daily limit reached. Corridor traffic is showing ML-predicted data until midnight UTC.
            Incidents and Events are still live.
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Left panel */}
        <div className="space-y-3">

          {/* Live summary cards */}
          {mode === 'live' && liveSummary && (
            <Panel>
              <PanelHeader title="Live Summary" badge="NOW" badgeColor="live" />
              <div className="p-3 grid grid-cols-2 gap-2">
                {[
                  { label:'Incidents',    val: liveSummary.total_incidents,         color:'#EF4444' },
                  { label:'Critical Roads', val: liveSummary.critical_corridors,     color:'#F97316' },
                  { label:'Live Events',  val: liveSummary.live_events,             color:'#8B5CF6' },
                  { label:'Avg Traffic', val: `${liveSummary.avg_congestion_pct}%`, color:'#F59E0B' },
                ].map(s => (
                  <div key={s.label} className="p-2 rounded-lg text-center" style={{ background:'var(--bg-elevated)' }}>
                    <div className="text-lg font-bold font-mono" style={{ color:s.color }}>{s.val}</div>
                    <div className="text-[10px]" style={{ color:'var(--text-muted)' }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </Panel>
          )}

          {/* View mode selector */}
          <Panel>
            <PanelHeader title={mode === 'live' ? 'Live View Mode' : 'Historical View'} />
            <div className="p-3 space-y-1.5">
              {mode === 'live'
                ? LIVE_VIEWS.map(v => (
                  <button key={v.key} onClick={() => setView(v.key)}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium cursor-pointer transition-all border"
                    style={{
                      background: view === v.key ? 'rgba(59,130,246,0.12)' : 'transparent',
                      borderColor: view === v.key ? 'rgba(59,130,246,0.4)' : 'transparent',
                      color: view === v.key ? 'var(--blue)' : 'var(--text-secondary)',
                    }}>
                    <v.icon size={14} />
                    {v.label}
                    {v.key === 'incidents' && <span className="ml-auto font-mono text-[10px]" style={{ color:'var(--text-muted)' }}>{liveIncidents.length}</span>}
                    {v.key === 'traffic'   && <span className="ml-auto font-mono text-[10px]" style={{ color:'var(--text-muted)' }}>{liveTraffic.length}</span>}
                    {v.key === 'events'    && <span className="ml-auto font-mono text-[10px]" style={{ color:'var(--text-muted)' }}>{liveEvents.length}</span>}
                  </button>
                ))
                : (
                  <>
                    {(['heatmap','markers'] as const).map(v => (
                      <button key={v} onClick={() => setView(v)}
                        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium cursor-pointer transition-all border"
                        style={{
                          background: view === v ? 'rgba(59,130,246,0.12)' : 'transparent',
                          borderColor: view === v ? 'rgba(59,130,246,0.4)' : 'transparent',
                          color: view === v ? 'var(--blue)' : 'var(--text-secondary)',
                        }}>
                        <Database size={14} />
                        {v === 'heatmap' ? 'Density Heatmap' : 'Event Markers'}
                        <span className="ml-auto font-mono text-[10px]" style={{ color:'var(--text-muted)' }}>
                          {v === 'heatmap' ? histPoints.length : histEvents.length}
                        </span>
                      </button>
                    ))}
                    <div className="pt-1">
                      <label className="text-[10px] font-semibold uppercase tracking-wider block mb-1.5" style={{ color:'var(--text-muted)' }}>Filter by Cause</label>
                      <select value={filter} onChange={e => setFilter(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-xs outline-none"
                        style={{ background:'var(--bg-elevated)', border:'1px solid var(--border)', color:'var(--text-primary)' }}>
                        {historicalCauses.map(c => <option key={c} value={c}>{c === 'all' ? 'All Causes' : c.replace(/_/g,' ')}</option>)}
                      </select>
                    </div>
                  </>
                )
              }
            </div>
          </Panel>

          {/* Legend */}
          <Panel>
            <PanelHeader title="Legend" />
            <div className="p-3 space-y-1.5">
              {mode === 'live' && view === 'incidents' && (
                Object.entries(SEV_COLOR).map(([k,v]) => (
                  <div key={k} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background:v }} />
                    <span style={{ color:'var(--text-secondary)' }}>{k} Incident</span>
                  </div>
                ))
              )}
              {mode === 'live' && view === 'traffic' && (
                [['#EF4444','Critical (>70%)'],['#F97316','High (45–70%)'],['#F59E0B','Moderate (20–45%)'],['#10B981','Free Flow (<20%)']].map(([c,l]) => (
                  <div key={l} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background:c }} />
                    <span style={{ color:'var(--text-secondary)' }}>{l}</span>
                  </div>
                ))
              )}
              {mode === 'live' && view === 'events' && (
                [['#EF4444','High Risk (>15k crowd)'],['#F59E0B','Moderate (3–15k)'],['#10B981','Low (<3k)']].map(([c,l]) => (
                  <div key={l} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 flex-shrink-0" style={{ background:c, clipPath:'polygon(50% 0%,100% 40%,80% 100%,20% 100%,0% 40%)' }} />
                    <span style={{ color:'var(--text-secondary)' }}>{l}</span>
                  </div>
                ))
              )}
              {(mode === 'historical' || view === 'heatmap') && (
                [['#EF4444','Critical density'],['#F97316','High density'],['#F59E0B','Moderate'],['#3B82F6','Low density']].map(([c,l]) => (
                  <div key={l} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background:c }} />
                    <span style={{ color:'var(--text-secondary)' }}>{l}</span>
                  </div>
                ))
              )}
            </div>
          </Panel>

          {/* Feed */}
          <Panel>
            <PanelHeader title={mode === 'live' ? '🔴 Live Feed' : 'Recent Events'} badge={mode === 'live' ? 'LIVE' : 'HIST'} badgeColor={mode === 'live' ? 'live' : 'blue'}>
              {mode === 'live' && (
                <button onClick={loadLive} disabled={refreshing}
                  className="p-1 rounded cursor-pointer transition-opacity"
                  style={{ color:'var(--text-muted)', opacity: refreshing ? 0.4 : 1 }}>
                  <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
                </button>
              )}
            </PanelHeader>
            {mode === 'live' && lastUpdate && (
              <div className="px-3 py-1 text-[10px] font-mono border-b" style={{ color:'var(--text-muted)', borderColor:'var(--border)' }}>
                Last update: {lastUpdate}
              </div>
            )}
            <div className="p-3 space-y-1.5 max-h-64 overflow-y-auto">
              {sidebarItems.map((item: any, i: number) => {
                if (mode === 'live' && view === 'incidents') {
                  const color = SEV_COLOR[item.severity] || '#94A3B8'
                  return (
                    <div key={i} onClick={() => setSelectedItem({ type:'incident', data:item })}
                      className="p-2 rounded-lg border cursor-pointer transition-all hover:border-blue-500/30 text-xs"
                      style={{ background:'var(--bg-elevated)', borderColor: selectedItem?.data?.id === item.id ? 'rgba(59,130,246,0.4)' : 'var(--border)' }}>
                      <div className="flex items-center gap-2 mb-0.5">
                        <div className="w-2 h-2 rounded-full flex-shrink-0 animate-pulse-glow" style={{ background:color }} />
                        <span className="font-semibold truncate" style={{ color:'var(--text-primary)' }}>{item.cause}</span>
                        <span className="ml-auto text-[10px] font-mono" style={{ color }}>{item.severity}</span>
                      </div>
                      <div className="truncate" style={{ color:'var(--text-secondary)' }}>
                        {item.from && item.to ? `${item.from} → ${item.to}` : item.road}
                      </div>
                      <div style={{ color: item.delay_sec >= 60 ? 'var(--amber)' : 'var(--text-muted)' }}>
                        {item.delay_sec >= 60 ? `+${Math.round(item.delay_sec/60)} min delay` : item.description || 'Active incident'}
                      </div>
                    </div>
                  )
                }
                if (mode === 'live' && view === 'traffic') {
                  const color = CONGESTION_COLOR(item.congestion_pct)
                  return (
                    <div key={i} onClick={() => setSelectedItem({ type:'traffic', data:item })}
                      className="p-2 rounded-lg border cursor-pointer transition-all text-xs"
                      style={{ background:'var(--bg-elevated)', borderColor:'var(--border)' }}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[11px] truncate pr-2" style={{ color:'var(--text-primary)' }}>{item.corridor}</span>
                        <span className="font-mono font-bold text-[11px] flex-shrink-0" style={{ color }}>{item.congestion_pct}%</span>
                      </div>
                      <div className="w-full h-1 rounded-full" style={{ background:'rgba(255,255,255,0.06)' }}>
                        <div className="h-full rounded-full" style={{ width:`${item.congestion_pct}%`, background:color }} />
                      </div>
                    </div>
                  )
                }
                if (mode === 'live' && view === 'events') {
                  const color = item.risk_level === 'High' ? '#EF4444' : item.risk_level === 'Moderate' ? '#F59E0B' : '#10B981'
                  return (
                    <div key={i} onClick={() => setSelectedItem({ type:'event', data:item })}
                      className="p-2 rounded-lg border cursor-pointer text-xs"
                      style={{ background:'var(--bg-elevated)', borderColor:'var(--border)' }}>
                      <div className="font-semibold truncate" style={{ color:'var(--text-primary)' }}>{item.name || item.event_type}</div>
                      <div className="flex items-center justify-between mt-0.5">
                        <span style={{ color:'var(--text-secondary)' }}>{item.event_type}</span>
                        <span style={{ color }}>{item.risk_level}</span>
                      </div>
                      <div style={{ color:'var(--text-muted)' }}>~{(item.crowd_estimate||0).toLocaleString()} people</div>
                    </div>
                  )
                }
                // historical
                const ev = item as any
                const color = CAUSE_COLORS[ev.event_cause] || '#94A3B8'
                return (
                  <div key={i} className="p-2 rounded-lg border text-xs" style={{ background:'var(--bg-elevated)', borderColor:'var(--border)' }}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background:color }} />
                      <span className="font-semibold capitalize truncate" style={{ color:'var(--text-primary)' }}>{ev.event_cause?.replace(/_/g,' ')}</span>
                    </div>
                    <div className="truncate mt-0.5" style={{ color:'var(--text-secondary)' }}>{ev.police_station || ev.zone}</div>
                  </div>
                )
              })}
            </div>
          </Panel>
        </div>

        {/* Map + detail panel */}
        <div className="lg:col-span-3 space-y-3">
          <div className="relative w-full rounded-xl overflow-hidden border" style={{ height:540, background:'var(--bg-card)', borderColor:'var(--border)' }}>
            <div ref={mapRef} className="absolute inset-0" />
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center" style={{ background:'var(--bg-card)' }}>
                <Spinner />
              </div>
            )}
          </div>

          {/* Selected item detail */}
          {selectedItem && (
            <Panel>
              <PanelHeader
                title={selectedItem.type === 'incident' ? '🚨 Incident Detail' : selectedItem.type === 'traffic' ? '🛣 Corridor Detail' : '📍 Event Detail'}
                badge={selectedItem.type === 'incident' ? selectedItem.data.severity : selectedItem.type === 'traffic' ? `${selectedItem.data.congestion_pct}% congested` : selectedItem.data.risk_level}
                badgeColor={selectedItem.data.severity === 'Critical' || selectedItem.data.risk_level === 'High' || selectedItem.data.congestion_pct > 70 ? 'red' : 'amber'}>
                <button onClick={() => setSelectedItem(null)} className="text-xs cursor-pointer px-2 py-0.5 rounded" style={{ color:'var(--text-muted)', background:'var(--bg-elevated)' }}>✕ Close</button>
              </PanelHeader>
              <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                {selectedItem.type === 'incident' && (<>
                  {[['Cause',selectedItem.data.cause],['Severity',selectedItem.data.severity],['Road',selectedItem.data.road],['Source',selectedItem.data.source],
                    ['Delay',selectedItem.data.delay_sec >= 60 ? `+${Math.round(selectedItem.data.delay_sec/60)} min` : selectedItem.data.delay_sec > 0 ? '< 1 min' : 'Minimal'],
                    ['Length',selectedItem.data.length_m > 0 ? `${selectedItem.data.length_m}m` : 'Unknown'],
                    ['From',selectedItem.data.from||'—'],['Updated', new Date(selectedItem.data.timestamp).toLocaleTimeString()]
                  ].map(([l,v]) => (
                    <div key={l} className="p-2 rounded-lg" style={{ background:'var(--bg-elevated)' }}>
                      <div style={{ color:'var(--text-muted)' }}>{l}</div>
                      <div className="font-semibold mt-0.5 truncate" style={{ color:'var(--text-primary)' }}>{v}</div>
                    </div>
                  ))}
                </>)}
                {selectedItem.type === 'traffic' && (<>
                  {[['Corridor',selectedItem.data.corridor],['Congestion',`${selectedItem.data.congestion_pct}%`],
                    ['Level',selectedItem.data.congestion_level],['Normal Travel',`${selectedItem.data.duration_normal_min} min`],
                    ['With Traffic',`${selectedItem.data.duration_traffic_min} min`],['Extra Delay',`+${selectedItem.data.delay_min} min`],
                    ['Distance',`${selectedItem.data.distance_km} km`],['Source',selectedItem.data.source],
                  ].map(([l,v]) => (
                    <div key={l} className="p-2 rounded-lg" style={{ background:'var(--bg-elevated)' }}>
                      <div style={{ color:'var(--text-muted)' }}>{l}</div>
                      <div className="font-semibold mt-0.5 truncate" style={{ color:'var(--text-primary)' }}>{v}</div>
                    </div>
                  ))}
                </>)}
                {selectedItem.type === 'event' && (<>
                  {[['Name',selectedItem.data.name||selectedItem.data.event_type],['Type',selectedItem.data.event_type],
                    ['Crowd Est.',`~${(selectedItem.data.crowd_estimate||0).toLocaleString()}`],['Risk Level',selectedItem.data.risk_level],
                    ['Source',selectedItem.data.source],['Updated', new Date(selectedItem.data.timestamp).toLocaleTimeString()],
                  ].map(([l,v]) => (
                    <div key={l} className="p-2 rounded-lg" style={{ background:'var(--bg-elevated)' }}>
                      <div style={{ color:'var(--text-muted)' }}>{l}</div>
                      <div className="font-semibold mt-0.5 truncate" style={{ color:'var(--text-primary)' }}>{v}</div>
                    </div>
                  ))}
                </>)}
              </div>
            </Panel>
          )}

          {/* Corridor traffic table (live traffic view) */}
          {mode === 'live' && view === 'traffic' && liveTraffic.length > 0 && (
            <Panel>
              <PanelHeader title="All Corridor Status" badge={`${liveTraffic.length} corridors`} badgeColor="blue" />
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b" style={{ borderColor:'var(--border)' }}>
                      {['Corridor','Congestion','Level','Normal','w/ Traffic','Delay','Distance'].map(h => (
                        <th key={h} className="text-left px-4 py-2 text-[10px] uppercase tracking-wider" style={{ color:'var(--text-muted)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...liveTraffic].sort((a,b) => b.congestion_pct - a.congestion_pct).map((t,i) => {
                      const color = CONGESTION_COLOR(t.congestion_pct)
                      return (
                        <tr key={i} onClick={() => setSelectedItem({ type:'traffic', data:t })}
                          className="border-b cursor-pointer hover:bg-white/[0.02] transition-colors"
                          style={{ borderColor:'rgba(59,130,246,0.05)' }}>
                          <td className="px-4 py-2 text-xs font-semibold" style={{ color:'var(--text-primary)' }}>{t.corridor}</td>
                          <td className="px-4 py-2">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background:'rgba(255,255,255,0.06)' }}>
                                <div className="h-full rounded-full" style={{ width:`${t.congestion_pct}%`, background:color }} />
                              </div>
                              <span className="text-xs font-mono font-bold" style={{ color }}>{t.congestion_pct}%</span>
                            </div>
                          </td>
                          <td className="px-4 py-2 text-xs font-semibold" style={{ color }}>{t.congestion_level}</td>
                          <td className="px-4 py-2 text-xs font-mono" style={{ color:'var(--text-secondary)' }}>{t.duration_normal_min} min</td>
                          <td className="px-4 py-2 text-xs font-mono" style={{ color:'var(--text-secondary)' }}>{t.duration_traffic_min} min</td>
                          <td className="px-4 py-2 text-xs font-mono" style={{ color:'var(--amber)' }}>+{t.delay_min} min</td>
                          <td className="px-4 py-2 text-xs font-mono" style={{ color:'var(--text-muted)' }}>{t.distance_km} km</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  )
}