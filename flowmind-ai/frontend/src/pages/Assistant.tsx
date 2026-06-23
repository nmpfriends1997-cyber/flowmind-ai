import React, { useState, useRef, useEffect } from 'react'
import { sendChatMessage } from '../lib/api'
import { Panel, PanelHeader } from '../components/UI'
import { Send, Bot, User, Zap } from 'lucide-react'

interface Message { role: 'user' | 'assistant'; content: string }

const QUICK = [
  'What are the top risk zones in Bengaluru?',
  'Which event type causes the most road closures?',
  'What resources should I deploy for a VIP movement event?',
  'When is the peak incident time in Bengaluru?',
  'Which corridor has the highest event frequency?',
  'How does FlowMind AI predict congestion?',
]

export default function Assistant() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I\'m FlowMind AI, your intelligent traffic command assistant. I\'m trained on 8,173 real Bengaluru traffic events and can help with predictions, resource recommendations, diversion strategies, and data insights. What would you like to know?' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async (text?: string) => {
    const msg = text || input.trim()
    if (!msg || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: msg }])
    setLoading(true)
    try {
      const { reply } = await sendChatMessage(msg, messages)
      setMessages(m => [...m, { role: 'assistant', content: reply }])
    } catch {
      setMessages(m => [...m, { role: 'assistant', content: '⚠️ Connection error. Make sure the backend is running and OPENROUTER_API_KEY is set in backend/.env' }])
    }
    setLoading(false)
  }

  return (
    <div className="p-6 h-[calc(100vh-56px)] flex flex-col gap-4 animate-slide-in">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>AI Assistant</h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Powered by OpenRouter · Trained on Bengaluru event intelligence</p>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        {/* Quick prompts */}
        <div className="w-56 flex-shrink-0 space-y-2">
          <Panel>
            <PanelHeader title="Quick Questions" />
            <div className="p-3 space-y-1.5">
              {QUICK.map((q, i) => (
                <button key={i} onClick={() => send(q)}
                  className="w-full text-left text-[11px] px-3 py-2 rounded-lg border transition-all cursor-pointer"
                  style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
                  onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = 'rgba(59,130,246,0.4)'; (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)' }}
                  onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'; (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)' }}>
                  {q}
                </button>
              ))}
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Dataset Facts" badge="Grounded" badgeColor="green" />
            <div className="p-3 space-y-2">
              {[
                ['8,173', 'total events'],
                ['1,007', 'active now'],
                ['4,896', 'vehicle breakdowns'],
                ['5,030', 'high priority'],
                ['676', 'road closures'],
                ['8–10pm', 'peak hour'],
              ].map(([v, l]) => (
                <div key={l} className="flex justify-between text-xs">
                  <span style={{ color: 'var(--text-muted)' }}>{l}</span>
                  <span className="font-mono font-bold" style={{ color: 'var(--blue)' }}>{v}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {/* Chat area */}
        <Panel className="flex-1 flex flex-col min-h-0">
          <PanelHeader title="FlowMind AI Chat">
            <span className="inline-flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--green)' }}>
              <span className="w-1.5 h-1.5 rounded-full animate-pulse-glow" style={{ background: 'var(--green)' }} />
              Online
            </span>
          </PanelHeader>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((m, i) => (
              <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {m.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)' }}>
                    <Bot size={16} style={{ color: 'var(--blue)' }} />
                  </div>
                )}
                <div className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${m.role === 'user' ? 'rounded-tr-md' : 'rounded-tl-md'}`}
                  style={{
                    background: m.role === 'user' ? 'var(--blue)' : 'var(--bg-elevated)',
                    color: 'var(--text-primary)',
                    border: m.role === 'assistant' ? '1px solid var(--border)' : 'none',
                  }}>
                  {m.content}
                </div>
                {m.role === 'user' && (
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'rgba(255,255,255,0.07)' }}>
                    <User size={16} style={{ color: 'var(--text-secondary)' }} />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)' }}>
                  <Bot size={16} style={{ color: 'var(--blue)' }} />
                </div>
                <div className="px-4 py-3 rounded-2xl rounded-tl-md" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}>
                  <div className="flex gap-1 items-center h-4">
                    {[0,1,2].map(i => (
                      <span key={i} className="w-1.5 h-1.5 rounded-full animate-pulse-glow" style={{ background: 'var(--blue)', animationDelay: `${i*0.2}s` }} />
                    ))}
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <div className="flex gap-2">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder="Ask about Bengaluru traffic patterns, risk zones, resources…"
                className="flex-1 px-4 py-2.5 rounded-xl text-sm outline-none"
                style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
              <button onClick={() => send()} disabled={loading || !input.trim()}
                className="w-10 h-10 rounded-xl flex items-center justify-center cursor-pointer transition-all disabled:opacity-40"
                style={{ background: 'var(--blue)' }}>
                <Send size={15} color="#fff" />
              </button>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}