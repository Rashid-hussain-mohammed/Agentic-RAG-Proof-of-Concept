import React, { useState } from 'react';
import { Upload, Send, FileText, Cpu, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

interface ChatMessage {
  sender: 'user' | 'agent';
  text: string;
  loops?: number;
  timeTaken?: string;
}

const API_BASE_URL = 'http://localhost:8000/api/v1';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus('Uploading and indexing document into ChromaDB...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      setChunkCount(data.chunks);
      setUploadStatus(`Indexed successfully! Created ${data.chunks} vector chunks.`);
    } catch (err) {
      setUploadStatus('Error uploading or processing document.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage: ChatMessage = { sender: 'user', text: question };
    setChatHistory((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsQuerying(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.text }),
      });

      const processTime = response.headers.get('X-Process-Time');
      const data = await response.json();

      const agentMessage: ChatMessage = {
        sender: 'agent',
        text: data.answer || 'No response generated.',
        loops: data.retrieval_loops,
        timeTaken: processTime ? `${parseFloat(processTime).toFixed(2)}s` : undefined,
      };

      setChatHistory((prev) => [...prev, agentMessage]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        { sender: 'agent', text: 'Error connecting to the Agentic RAG backend.' },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Cpu size={28} color="#3b82f6" />
          <h1 style={{ margin: 0, fontSize: '1.4rem' }}>Agentic RAG Control Center</h1>
        </div>
        <span style={styles.badge}>LangGraph + ChromaDB + Ollama</span>
      </header>

      {/* Main Layout */}
      <div style={styles.bodyGrid}>
        {/* Left: Document Ingestion Panel */}
        <section style={styles.card}>
          <h2 style={styles.cardTitle}>
            <FileText size={20} /> Ingest Knowledge Base
          </h2>
          <p style={styles.cardSubtitle}>
            Upload a PDF to parse text, run Recursive Text Splitting, and store embeddings.
          </p>

          <form onSubmit={handleFileUpload} style={styles.form}>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              style={styles.fileInput}
            />

            <button
              type="submit"
              disabled={!selectedFile || isUploading}
              style={{
                ...styles.button,
                opacity: !selectedFile || isUploading ? 0.6 : 1,
              }}
            >
              {isUploading ? <Loader2 size={18} style={styles.spin} /> : <Upload size={18} />}
              {isUploading ? 'Indexing Chunks...' : 'Process & Index PDF'}
            </button>
          </form>

          {uploadStatus && (
            <div
              style={{
                ...styles.statusBox,
                backgroundColor: uploadStatus.includes('Error') ? '#fee2e2' : '#eff6ff',
                color: uploadStatus.includes('Error') ? '#b91c1c' : '#1d4ed8',
              }}
            >
              {uploadStatus.includes('Error') ? (
                <AlertCircle size={18} />
              ) : (
                <CheckCircle2 size={18} />
              )}
              <span>{uploadStatus}</span>
            </div>
          )}

          {chunkCount !== null && (
            <div style={styles.metricsBox}>
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#475569' }}>
                Active Vector Collection:
              </p>
              <h3 style={{ margin: '4px 0 0 0', color: '#0f172a' }}>{chunkCount} Chunks Indexed</h3>
            </div>
          )}
        </section>

        {/* Right: Agent Chat Panel */}
        <section style={{ ...styles.card, display: 'flex', flexDirection: 'column' }}>
          <h2 style={styles.cardTitle}>
            <Cpu size={20} /> Query Reasoning Engine
          </h2>

          {/* Chat Transcript */}
          <div style={styles.chatArea}>
            {chatHistory.length === 0 ? (
              <div style={styles.emptyState}>
                Ask a question about your indexed document. The LangGraph agent will evaluate relevance,
                rewrite if needed, and synthesize an answer.
              </div>
            ) : (
              chatHistory.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    ...styles.chatBubble,
                    alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                    backgroundColor: msg.sender === 'user' ? '#2563eb' : '#f1f5f9',
                    color: msg.sender === 'user' ? '#ffffff' : '#0f172a',
                  }}
                >
                  <p style={{ margin: 0, lineHeight: 1.5 }}>{msg.text}</p>
                  {msg.sender === 'agent' && (msg.loops || msg.timeTaken) && (
                    <div style={styles.agentMeta}>
                      {msg.loops && <span>Loops: {msg.loops}</span>}
                      {msg.timeTaken && <span>Latency: {msg.timeTaken}</span>}
                    </div>
                  )}
                </div>
              ))
            )}
            {isQuerying && (
              <div style={{ ...styles.chatBubble, alignSelf: 'flex-start', backgroundColor: '#f1f5f9' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#64748b' }}>
                  <Loader2 size={16} style={styles.spin} /> Agent evaluating context & generating...
                </span>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <form onSubmit={handleSendQuery} style={styles.queryForm}>
            <input
              type="text"
              placeholder="Ask anything from your uploaded PDF..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isQuerying}
              style={styles.queryInput}
            />
            <button
              type="submit"
              disabled={isQuerying || !question.trim()}
              style={styles.sendButton}
            >
              <Send size={18} />
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    fontFamily: 'Inter, system-ui, sans-serif',
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '24px',
    color: '#0f172a',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '20px',
    borderBottom: '1px solid #e2e8f0',
    marginBottom: '24px',
  },
  badge: {
    fontSize: '0.8rem',
    background: '#eff6ff',
    color: '#1d4ed8',
    padding: '6px 12px',
    borderRadius: '16px',
    fontWeight: 600,
  },
  bodyGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 2fr',
    gap: '24px',
    alignItems: 'stretch',
  },
  card: {
    background: '#ffffff',
    border: '1px solid #e2e8f0',
    borderRadius: '12px',
    padding: '20px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    margin: '0 0 8px 0',
    fontSize: '1.1rem',
  },
  cardSubtitle: {
    margin: '0 0 16px 0',
    fontSize: '0.875rem',
    color: '#64748b',
    lineHeight: 1.4,
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  fileInput: {
    padding: '10px',
    border: '1px dashed #cbd5e1',
    borderRadius: '8px',
    cursor: 'pointer',
  },
  button: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 16px',
    background: '#0f172a',
    color: '#ffffff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: 500,
  },
  statusBox: {
    marginTop: '16px',
    padding: '12px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.85rem',
  },
  metricsBox: {
    marginTop: '16px',
    padding: '12px',
    background: '#f8fafc',
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
  },
  chatArea: {
    flex: 1,
    minHeight: '360px',
    maxHeight: '480px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '12px',
    border: '1px solid #f1f5f9',
    borderRadius: '8px',
    marginBottom: '16px',
    background: '#fafafa',
  },
  emptyState: {
    margin: 'auto',
    textAlign: 'center',
    maxWidth: '320px',
    fontSize: '0.875rem',
    color: '#94a3b8',
    lineHeight: 1.5,
  },
  chatBubble: {
    maxWidth: '80%',
    padding: '10px 14px',
    borderRadius: '12px',
    fontSize: '0.9rem',
  },
  agentMeta: {
    display: 'flex',
    gap: '12px',
    marginTop: '6px',
    fontSize: '0.72rem',
    color: '#64748b',
    borderTop: '1px solid #e2e8f0',
    paddingTop: '4px',
  },
  queryForm: {
    display: 'flex',
    gap: '8px',
  },
  queryInput: {
    flex: 1,
    padding: '12px 14px',
    borderRadius: '8px',
    border: '1px solid #cbd5e1',
    fontSize: '0.9rem',
    outline: 'none',
  },
  sendButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0 16px',
    background: '#2563eb',
    color: '#ffffff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
  },
  spin: {
    animation: 'spin 1s linear infinite',
  },
};