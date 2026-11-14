import React, { useState } from 'react';
import {
  History as HistoryIcon,
  Search,
  Trash2,
  Clock,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { PredictionResult } from '../types';

interface HistoryViewProps {
  history: PredictionResult[];
  onClearHistory: () => void;
  onSelectResult: (result: PredictionResult) => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onClearHistory,
  onSelectResult,
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredHistory = history.filter(
    (item) =>
      item.predicted_class.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.class_index.toString().includes(searchTerm)
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <HistoryIcon className="h-5 w-5 text-cyan-400" />
            <span>Local Prediction History</span>
          </h2>
          <p className="text-xs text-slate-400">
            Metadata from evaluations persisted securely in browser localStorage (privacy-first: image bytes are not uploaded or permanently retained)
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="flex items-center space-x-1.5 rounded-lg border border-rose-900/50 bg-rose-950/30 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-900/50 transition"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search prediction history by class name, file, or class index..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full rounded-xl border border-slate-800 bg-slate-950 py-2.5 pl-10 pr-4 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        {filteredHistory.length > 0 ? (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-4 py-3">Filename</th>
                  <th className="px-4 py-3">Predicted Class</th>
                  <th className="px-4 py-3">Class Index</th>
                  <th className="px-4 py-3">Top-1 Confidence</th>
                  <th className="px-4 py-3">Latency</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                {filteredHistory.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 text-slate-400">
                      {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Recent'}
                    </td>
                    <td className="px-4 py-3 font-medium text-white max-w-[140px] truncate">
                      {item.filename}
                    </td>
                    <td className="px-4 py-3 capitalize font-semibold text-cyan-300">
                      {item.predicted_class.replace(/_/g, ' ')}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-400">#{item.class_index}</td>
                    <td className="px-4 py-3 font-mono font-medium text-slate-200">
                      {item.confidence_percent.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-400">{item.latency_ms} ms</td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => onSelectResult(item)}
                        className="rounded-lg bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-cyan-400 hover:bg-slate-700 transition"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-slate-800 p-10 text-center text-xs text-slate-500">
            {history.length === 0
              ? 'No prediction records found. Run an inference in the Vision Studio to start tracking history.'
              : 'No results match your search query.'}
          </div>
        )}
      </div>
    </div>
  );
};
