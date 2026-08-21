import { useState, useEffect } from 'react';
import { Header } from '../components/common/Header';
import {
  Search, BookOpen, Brain, Shield, Activity, ArrowRight, Share2,
  Layers, Cpu, Compass, RefreshCw
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface GraphRAGResponse {
  answer: string;
  observed_evidence: string[];
  analytics: string[];
  model_predictions: string[];
  knowledge_guidance: string[];
  recommendations: string[];
  graph_entities: any[];
  relationships_used: any[];
  knowledge_sources: string[];
  confidence: number;
  insufficient_evidence: boolean;
  query_latency_ms: number;
}

interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_distribution: Record<string, number>;
  edge_distribution: Record<string, number>;
  last_synced: string | null;
}

interface SubgraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, string>;
}

interface SubgraphLink {
  source: string;
  target: string;
  relation: string;
}

interface SubgraphData {
  nodes: SubgraphNode[];
  links: SubgraphLink[];
  total_nodes: number;
  total_links: number;
}

interface SafetyStandardChunk {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  source: string;
  category: string;
  section_id: string;
  heading: string;
  text: string;
}

const RESEARCH_QUERIES = [
  'Why is Worker W001 high risk?',
  'Why is the project predicted to be delayed?',
  'Which PPE item is most frequently missing?',
  'Which zone has the highest safety risk and which PPE is missing?',
  'What safety events occurred during structural work?',
  'What is the current construction stage and progress status?',
  'What are the OSHA head protection and hard hat requirements?',
  'What are the 6-foot fall protection rules in construction?',
];

const NODE_COLORS: Record<string, string> = {
  'NodeType.PROJECT': '#a855f7',
  'NodeType.CONSTRUCTION_STAGE': '#06b6d4',
  'NodeType.PPE_ITEM': '#10b981',
  'NodeType.WORKER': '#3b82f6',
  'NodeType.ZONE': '#f59e0b',
  'NodeType.VIOLATION': '#ef4444',
  'Project': '#a855f7',
  'Worker': '#3b82f6',
  'Violation': '#ef4444',
  'PPEItem': '#10b981',
  'Zone': '#f59e0b',
  'ConstructionStage': '#06b6d4',
};

export default function SafetyKnowledge() {
  const [query, setQuery] = useState('');
  const [ragResult, setRagResult] = useState<GraphRAGResponse | null>(null);
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [subgraph, setSubgraph] = useState<SubgraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<SubgraphNode | null>(null);
  const [standards, setStandards] = useState<SafetyStandardChunk[]>([]);
  const [standardSearch, setStandardSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [activeTab, setActiveTab] = useState<'assistant' | 'graph' | 'standards'>('assistant');

  useEffect(() => {
    fetchGraphStats();
    fetchSubgraph();
    fetchStandards();
  }, []);

  const fetchGraphStats = async () => {
    try {
      const res = await fetch(`${API_URL}/api/graph/stats`);
      if (res.ok) {
        const data = await res.json();
        setGraphStats(data);
      }
    } catch { /* ignore */ }
  };

  const fetchSubgraph = async () => {
    setGraphLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/graph/subgraph?max_nodes=80`);
      if (res.ok) {
        const data = await res.json();
        setSubgraph(data);
      }
    } catch { /* ignore */ } finally {
      setGraphLoading(false);
    }
  };

  const fetchStandards = async () => {
    try {
      const res = await fetch(`${API_URL}/api/safety/standards`);
      if (res.ok) {
        const data = await res.json();
        setStandards(data);
      }
    } catch { /* ignore */ }
  };

  const executeGraphRAG = async (q: string = query) => {
    if (!q.trim()) return;
    setLoading(true);
    setSearched(true);
    try {
      const res = await fetch(`${API_URL}/api/graphrag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      if (res.ok) {
        const data = await res.json();
        setRagResult(data);
        fetchGraphStats();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const categories = ['All', ...Array.from(new Set(standards.map(s => s.category)))];

  const filteredStandards = standards.filter(s => {
    const matchesCat = selectedCategory === 'All' || s.category === selectedCategory;
    const matchesSearch = standardSearch === '' ||
      s.heading.toLowerCase().includes(standardSearch.toLowerCase()) ||
      s.text.toLowerCase().includes(standardSearch.toLowerCase()) ||
      s.source.toLowerCase().includes(standardSearch.toLowerCase()) ||
      s.doc_title.toLowerCase().includes(standardSearch.toLowerCase());
    return matchesCat && matchesSearch;
  });

  return (
    <>
      <Header title="Safety Knowledge & Construction Intelligence" subtitle="OSHA Standards, Multi-hop Knowledge Graph & Grounded GraphRAG Assistant" />
      <div className="app-content">

        {/* Top Summary Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 16 }}>
          <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ padding: 10, borderRadius: 8, background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6' }}>
              <Brain size={24} />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Graph Entities</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{graphStats?.total_nodes || 0}</div>
            </div>
          </div>

          <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ padding: 10, borderRadius: 8, background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>
              <Share2 size={24} />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Multi-Hop Relations</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{graphStats?.total_edges || 0}</div>
            </div>
          </div>

          <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ padding: 10, borderRadius: 8, background: 'rgba(6, 182, 212, 0.15)', color: '#06b6d4' }}>
              <BookOpen size={24} />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Safety Standards Indexed</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{standards.length}</div>
            </div>
          </div>

          <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ padding: 10, borderRadius: 8, background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}>
              <Cpu size={24} />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Grounding Mode</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#f59e0b' }}>Strict Evidence Only</div>
            </div>
          </div>
        </div>

        {/* View Mode Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          <button
            className={`btn ${activeTab === 'assistant' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('assistant')}
          >
            <Brain size={16} /> Explainable AI Assistant
          </button>
          <button
            className={`btn ${activeTab === 'graph' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => { setActiveTab('graph'); fetchSubgraph(); }}
          >
            <Layers size={16} /> Dynamic Knowledge Graph
          </button>
          <button
            className={`btn ${activeTab === 'standards' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => { setActiveTab('standards'); fetchStandards(); }}
          >
            <BookOpen size={16} /> OSHA & Safety Standards ({standards.length})
          </button>
        </div>

        {/* ── TAB 1: EXPLAINABLE AI ASSISTANT ────────────────────── */}
        {activeTab === 'assistant' && (
          <>
            {/* Search / Prompt Bar */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '14px 18px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-secondary)',
              borderRadius: 'var(--radius-lg)',
              marginBottom: 16,
            }}>
              <Search size={18} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <input
                type="text"
                placeholder="Ask about worker risk, delay causes, missing PPE, stage progress, OSHA regulations..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && executeGraphRAG()}
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  fontSize: 14,
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-sans)',
                }}
              />
              <button className="btn btn-primary" onClick={() => executeGraphRAG()} disabled={loading}>
                {loading ? 'Synthesizing...' : 'Query GraphRAG'}
              </button>
            </div>

            {/* Quick Prompts */}
            <div className="filter-bar" style={{ marginBottom: 16 }}>
              {RESEARCH_QUERIES.map(q => (
                <button key={q} className="filter-chip" onClick={() => { setQuery(q); executeGraphRAG(q); }}>
                  {q}
                </button>
              ))}
            </div>

            {/* Structured Evidence Results */}
            {searched && (
              <div className="card">
                <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Compass size={18} color="#3b82f6" />
                    Evidence-Grounded Query Response
                  </span>
                  {ragResult && (
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      Inference Latency: {ragResult.query_latency_ms} ms | Confidence: {(ragResult.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>

                {loading ? (
                  <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    <Activity size={32} className="animate-spin" style={{ margin: '0 auto 12px' }} />
                    Traversing Multi-Hop Knowledge Graph & Retrieving Evidence...
                  </div>
                ) : ragResult ? (
                  <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>

                    {/* Executive Answer */}
                    <div style={{
                      padding: 16,
                      borderRadius: 8,
                      background: ragResult.insufficient_evidence ? 'rgba(239, 68, 68, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                      border: `1px solid ${ragResult.insufficient_evidence ? 'rgba(239, 68, 68, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
                    }}>
                      <div style={{ fontSize: 13, fontWeight: 700, color: ragResult.insufficient_evidence ? '#ef4444' : '#3b82f6', marginBottom: 4 }}>
                        {ragResult.insufficient_evidence ? '⚠️ INSUFFICIENT EVIDENCE' : 'SYNTHESIZED EXPLANATION'}
                      </div>
                      <div style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.6 }}>
                        {ragResult.answer}
                      </div>
                    </div>

                    {/* Grounded Sections Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>

                      {/* 1. Observed Evidence */}
                      <div className="card" style={{ background: 'var(--bg-card-subtle)', padding: 14 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                          <Shield size={16} /> 1. OBSERVED EVIDENCE (MongoDB Telemetry)
                        </div>
                        {ragResult.observed_evidence.length > 0 ? (
                          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: 'var(--text-secondary)' }}>
                            {ragResult.observed_evidence.map((ev, i) => (
                              <li key={i} style={{ marginBottom: 4 }}>{ev}</li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No direct camera observations logged for this entity.</div>
                        )}
                      </div>

                      {/* 2. Model Predictions */}
                      <div className="card" style={{ background: 'var(--bg-card-subtle)', padding: 14 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                          <Cpu size={16} /> 2. MODEL PREDICTIONS (Delay & Progress ML)
                        </div>
                        {ragResult.model_predictions.length > 0 ? (
                          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: 'var(--text-secondary)' }}>
                            {ragResult.model_predictions.map((p, i) => (
                              <li key={i} style={{ marginBottom: 4 }}>{p}</li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No active model predictions queried.</div>
                        )}
                      </div>

                      {/* 3. Analytics & Calculations */}
                      <div className="card" style={{ background: 'var(--bg-card-subtle)', padding: 14 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#6366f1', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                          <Activity size={16} /> 3. ANALYTICS & CAUSAL FACTORS
                        </div>
                        {ragResult.analytics.length > 0 ? (
                          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: 'var(--text-secondary)' }}>
                            {ragResult.analytics.map((a, i) => (
                              <li key={i} style={{ marginBottom: 4 }}>{a}</li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>N/A</div>
                        )}
                      </div>

                      {/* 4. Actionable Recommendations */}
                      <div className="card" style={{ background: 'var(--bg-card-subtle)', padding: 14 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#06b6d4', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                          <ArrowRight size={16} /> 4. ACTIONABLE RECOMMENDATIONS
                        </div>
                        {ragResult.recommendations.length > 0 ? (
                          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: 'var(--text-secondary)' }}>
                            {ragResult.recommendations.map((r, i) => (
                              <li key={i} style={{ marginBottom: 4 }}>{r}</li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No corrective actions required at this time.</div>
                        )}
                      </div>

                    </div>

                    {/* Knowledge Base Guidance Sources */}
                    {ragResult.knowledge_guidance.length > 0 && (
                      <div style={{ marginTop: 8, padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid var(--border-secondary)' }}>
                        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                          <BookOpen size={14} /> STANDARD OPERATING PROCEDURES & CITATIONS
                        </div>
                        {ragResult.knowledge_guidance.map((g, i) => (
                          <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                            • {g}
                          </div>
                        ))}
                      </div>
                    )}

                  </div>
                ) : null}
              </div>
            )}
          </>
        )}

        {/* ── TAB 2: DYNAMIC KNOWLEDGE GRAPH VISUALIZATION ───────── */}
        {activeTab === 'graph' && (
          <div style={{ display: 'grid', gridTemplateColumns: selectedNode ? '1fr 340px' : '1fr', gap: 16 }}>
            <div className="card">
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Layers size={18} color="#3b82f6" />
                  Site Knowledge Graph Entity Map ({subgraph?.total_nodes || 0} Nodes, {subgraph?.total_links || 0} Edges)
                </span>
                <button className="btn btn-ghost" onClick={fetchSubgraph} disabled={graphLoading}>
                  <RefreshCw size={13} className={graphLoading ? 'animate-spin' : ''} /> Refresh Graph
                </button>
              </div>

              {/* Node Legend */}
              <div style={{ padding: '10px 16px', display: 'flex', gap: 12, flexWrap: 'wrap', borderBottom: '1px solid var(--border-primary)', fontSize: 11 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#a855f7' }}></span> Project</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#06b6d4' }}></span> Construction Stage</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#3b82f6' }}></span> Worker</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#10b981' }}></span> PPE Item</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#f59e0b' }}></span> Danger Zone</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444' }}></span> Violation</div>
              </div>

              {/* Interactive Node Explorer */}
              <div style={{ padding: 16 }}>
                {subgraph && subgraph.nodes.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10, maxHeight: 520, overflowY: 'auto' }}>
                    {subgraph.nodes.map(node => {
                      const color = NODE_COLORS[node.type] || '#6366f1';
                      const isSelected = selectedNode?.id === node.id;
                      return (
                        <div
                          key={node.id}
                          onClick={() => setSelectedNode(node)}
                          style={{
                            padding: '10px 12px',
                            background: isSelected ? 'rgba(59,130,246,0.15)' : 'var(--bg-surface)',
                            border: `1px solid ${isSelected ? color : 'var(--border-secondary)'}`,
                            borderRadius: 8,
                            cursor: 'pointer',
                            transition: 'all 0.15s',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                            <span style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                              {node.type.replace('NodeType.', '')}
                            </span>
                          </div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', wordBreak: 'break-word' }}>
                            {node.label}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No knowledge graph entities synced.
                  </div>
                )}
              </div>
            </div>

            {/* Node Inspector Drawer */}
            {selectedNode && (
              <div className="card slide-in">
                <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="card-title" style={{ fontSize: 14 }}>Entity Inspector</span>
                  <button className="btn btn-ghost" onClick={() => setSelectedNode(null)} style={{ padding: '2px 6px', fontSize: 12 }}>✕</button>
                </div>
                <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: NODE_COLORS[selectedNode.type] || 'var(--text-primary)' }}>
                    {selectedNode.label}
                  </div>
                  <div style={{ fontSize: 11, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                    Type: {selectedNode.type.replace('NodeType.', '')}
                  </div>
                  <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    ID: {selectedNode.id}
                  </div>

                  <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border-primary)' }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                      Node Properties
                    </div>
                    {Object.entries(selectedNode.properties || {}).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '4px 0', borderBottom: '1px solid var(--border-secondary)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                        <span style={{ fontWeight: 500, color: 'var(--text-primary)', maxWidth: 160, textAlign: 'right', wordBreak: 'break-word' }}>{v}</span>
                      </div>
                    ))}
                  </div>

                  {/* Connected Relationships */}
                  {subgraph && (
                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border-primary)' }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                        Connected Graph Edges
                      </div>
                      {subgraph.links
                        .filter(l => l.source === selectedNode.id || l.target === selectedNode.id)
                        .slice(0, 8)
                        .map((l, i) => (
                          <div key={i} style={{ fontSize: 11, padding: '4px 0', color: 'var(--text-secondary)' }}>
                            <span style={{ color: '#06b6d4', fontWeight: 600 }}>{l.relation.replace('RelationType.', '')}</span> → {l.source === selectedNode.id ? l.target : l.source}
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── TAB 3: OSHA & SAFETY STANDARDS LIBRARY ─────────────── */}
        {activeTab === 'standards' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Search & Category Filter */}
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{
                flex: 1, minWidth: 260,
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 14px', background: 'var(--bg-card)',
                border: '1px solid var(--border-secondary)', borderRadius: 8,
              }}>
                <Search size={16} style={{ color: 'var(--text-muted)' }} />
                <input
                  type="text"
                  placeholder="Search OSHA 1926 regulations, fall protection, respiratory silica, PPE rules..."
                  value={standardSearch}
                  onChange={e => setStandardSearch(e.target.value)}
                  style={{
                    flex: 1, background: 'transparent', border: 'none', outline: 'none',
                    fontSize: 13, color: 'var(--text-primary)',
                  }}
                />
              </div>

              <div className="filter-bar" style={{ margin: 0 }}>
                {categories.map(c => (
                  <button
                    key={c}
                    className={`filter-chip ${selectedCategory === c ? 'active' : ''}`}
                    onClick={() => setSelectedCategory(c)}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>

            {/* Standards List */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 16 }}>
              {filteredStandards.map((std) => (
                <div key={std.chunk_id} className="card" style={{ padding: 18, border: '1px solid var(--border-secondary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <span className="badge cyan" style={{ fontSize: 10 }}>{std.category}</span>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {std.source}
                    </span>
                  </div>
                  <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 6px' }}>
                    {std.heading}
                  </h3>
                  <div style={{ fontSize: 11, color: 'var(--accent-blue)', marginBottom: 10, fontWeight: 500 }}>
                    {std.doc_title}
                  </div>
                  <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
                    {std.text}
                  </p>
                  <div style={{ marginTop: 14, paddingTop: 10, borderTop: '1px solid var(--border-primary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      ID: {std.chunk_id}
                    </span>
                    <button
                      className="btn btn-ghost"
                      style={{ padding: '3px 8px', fontSize: 11 }}
                      onClick={() => {
                        setQuery(`What are the requirements for ${std.heading}?`);
                        setActiveTab('assistant');
                        executeGraphRAG(`What are the requirements for ${std.heading}?`);
                      }}
                    >
                      <Brain size={12} /> Query in GraphRAG
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </>
  );
}
