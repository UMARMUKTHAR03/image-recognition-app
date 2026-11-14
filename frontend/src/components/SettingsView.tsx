import React, { useState } from 'react';
import { Settings as SettingsIcon, Server, RefreshCw, CheckCircle2, XCircle } from 'lucide-react';
import { getApiBaseUrl, setApiBaseUrl, checkHealth } from '../services/api';
import { HealthResponse } from '../types';

interface SettingsViewProps {
  onBackendStatusChange: (online: boolean) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({ onBackendStatusChange }) => {
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl());
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<'success' | 'error' | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const handleSave = () => {
    setApiBaseUrl(apiUrl);
    testConnection();
  };

  const testConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    setStatusMessage(null);

    try {
      const data = await checkHealth();
      setHealth(data);
      setTestResult('success');
      setStatusMessage('Connected successfully to production vision server.');
      onBackendStatusChange(true);
    } catch (err: any) {
      setTestResult('error');
      setStatusMessage(err.message || 'Failed to reach API endpoint.');
      onBackendStatusChange(false);
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <SettingsIcon className="h-5 w-5 text-cyan-400" />
            <span>Platform Configuration & Gateway Settings</span>
          </h2>
          <p className="text-xs text-slate-400">
            Configure backend API host, Render deployment endpoint, and inspect platform status
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl space-y-6">
        <div>
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Backend API Endpoint URL
          </label>
          <div className="mt-2 flex space-x-3">
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://image-recognition-api-bgzb.onrender.com"
              className="flex-1 rounded-xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-xs font-mono text-slate-200 placeholder-slate-600 focus:border-cyan-500 focus:outline-none"
            />
            <button
              onClick={handleSave}
              disabled={isTesting}
              className="flex items-center space-x-2 rounded-xl bg-cyan-500 px-5 py-2.5 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition"
            >
              {isTesting && <RefreshCw className="h-3.5 w-3.5 animate-spin" />}
              <span>Save & Test</span>
            </button>
          </div>
          <p className="mt-1.5 text-[11px] text-slate-500">
            Default Render production URL: <code className="text-cyan-400">https://image-recognition-api-bgzb.onrender.com</code> or local development <code className="text-cyan-400">http://localhost:8000</code>
          </p>
        </div>

        {statusMessage && (
          <div
            className={`flex items-center space-x-2 rounded-xl border p-3.5 text-xs ${
              testResult === 'success'
                ? 'border-emerald-500/30 bg-emerald-950/30 text-emerald-300'
                : 'border-rose-500/30 bg-rose-950/30 text-rose-300'
            }`}
          >
            {testResult === 'success' ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="h-4 w-4 text-rose-400 shrink-0" />
            )}
            <span>{statusMessage}</span>
          </div>
        )}

        {health && (
          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-2 text-xs">
            <h4 className="font-semibold text-white">Live Backend Telemetry</h4>
            <div className="grid grid-cols-2 gap-2 text-slate-400">
              <div>Title: <span className="text-slate-200">{health.app_title}</span></div>
              <div>Version: <span className="text-slate-200">{health.version}</span></div>
              <div>Model: <span className="text-cyan-400 font-mono">{health.model}</span></div>
              <div>Classes: <span className="text-slate-200">{health.classes_count}</span></div>
            </div>
          </div>
        )}

        <div className="border-t border-slate-800 pt-5 text-xs text-slate-400 space-y-2">
          <h4 className="font-semibold text-slate-200">System Limits & Security Policy</h4>
          <ul className="list-disc list-inside space-y-1 text-slate-400">
            <li><b>Max Upload Payload:</b> 10 MB per image file</li>
            <li><b>Allowed Image Formats:</b> JPG, JPEG, PNG, WEBP</li>
            <li><b>Concurrent Batch Limit:</b> 8 images per batch inference request</li>
            <li><b>Zero Latency Simulation:</b> Inference latency reflects real CPU execution duration</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
