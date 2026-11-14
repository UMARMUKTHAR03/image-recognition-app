import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Studio } from './components/Studio';
import { ResultsView } from './components/ResultsView';
import { ExplainView } from './components/ExplainView';
import { BatchView } from './components/BatchView';
import { Dashboard } from './components/Dashboard';
import { HistoryView } from './components/HistoryView';
import { TaxonomyView } from './components/TaxonomyView';
import { AnalyticsView } from './components/AnalyticsView';
import { SettingsView } from './components/SettingsView';
import { ModelInfo, PredictionResult } from './types';
import { checkHealth, getModelInfo } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [latestResult, setLatestResult] = useState<PredictionResult | null>(null);
  const [latestPreviewUrl, setLatestPreviewUrl] = useState<string | null>(null);
  const [history, setHistory] = useState<PredictionResult[]>([]);

  // Load history from localStorage on startup
  useEffect(() => {
    try {
      const saved = localStorage.getItem('image_recognition_history');
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  // Poll backend health & model info
  useEffect(() => {
    const initBackend = async () => {
      try {
        await checkHealth();
        setIsBackendOnline(true);
        const info = await getModelInfo();
        setModelInfo(info);
      } catch {
        setIsBackendOnline(false);
      }
    };
    initBackend();
  }, []);

  const handlePredictionComplete = (result: PredictionResult, previewUrl: string) => {
    const itemWithMeta: PredictionResult = {
      ...result,
      timestamp: new Date().toISOString(),
      previewUrl,
    };
    setLatestResult(itemWithMeta);
    setLatestPreviewUrl(previewUrl);

    // Save to history (omit large base64 strings if any)
    const historyItem: PredictionResult = {
      ...result,
      timestamp: itemWithMeta.timestamp,
    };
    const updated = [historyItem, ...history.slice(0, 99)];
    setHistory(updated);
    try {
      localStorage.setItem('image_recognition_history', JSON.stringify(updated));
    } catch {
      // Storage quota safety
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('image_recognition_history');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-slate-950 font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendOnline={isBackendOnline}
      />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6">
        {activeTab === 'studio' && (
          <div className="space-y-8">
            <Studio onPredictionComplete={handlePredictionComplete} />
            {latestResult && latestPreviewUrl && (
              <div className="border-t border-slate-800 pt-8">
                <ResultsView
                  result={latestResult}
                  previewUrl={latestPreviewUrl}
                  onExplainClick={() => setActiveTab('explain')}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === 'explain' && (
          <ExplainView
            initialPreviewUrl={latestPreviewUrl}
          />
        )}

        {activeTab === 'batch' && <BatchView />}
        {activeTab === 'dashboard' && (
          <Dashboard
            modelInfo={modelInfo}
            history={history}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}
        {activeTab === 'history' && (
          <HistoryView
            history={history}
            onClearHistory={handleClearHistory}
            onSelectResult={(item) => {
              setLatestResult(item);
              setActiveTab('studio');
            }}
          />
        )}
        {activeTab === 'taxonomy' && <TaxonomyView />}
        {activeTab === 'analytics' && <AnalyticsView modelInfo={modelInfo} />}
        {activeTab === 'settings' && (
          <SettingsView
            onBackendStatusChange={(status) => setIsBackendOnline(status)}
          />
        )}
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-900/60 py-6 text-center text-xs text-slate-500">
        <p>
          Image Recognition App – Deep Learning Computer Vision Platform • Engineered by{' '}
          <span className="font-semibold text-slate-300">Umar Mukthar</span>
        </p>
        <p className="mt-1 text-[11px] text-slate-600">
          PyTorch MobileNetV3-Small • ImageNet-1K • Grad-CAM Explainable AI • React + TypeScript
        </p>
      </footer>
    </div>
  );
}

export default App;
