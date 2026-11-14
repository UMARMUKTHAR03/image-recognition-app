import React, { useRef, useState } from 'react';
import {
  Layers,
  Upload,
  AlertTriangle,
  Download,
  CheckCircle2,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import { BatchResponse } from '../types';
import { predictBatch } from '../services/api';

export const BatchView: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [response, setResponse] = useState<BatchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files).slice(0, 8);
      setFiles(selected);
      setResponse(null);
      setError(null);
    }
  };

  const handleRunBatch = async () => {
    if (files.length === 0) return;
    setIsLoading(true);
    setError(null);

    try {
      const res = await predictBatch(files);
      setResponse(res);
    } catch (err: any) {
      setError(err.message || 'Batch classification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const exportCSV = () => {
    if (!response) return;
    const rows = [
      ['Filename', 'Status', 'Predicted Class', 'Class Index', 'Confidence (%)', 'Latency (ms)'],
      ...response.results.map((r) => [
        r.filename,
        r.status,
        r.predicted_class || 'N/A',
        r.class_index !== undefined ? r.class_index.toString() : 'N/A',
        r.confidence_percent !== undefined ? r.confidence_percent.toString() : 'N/A',
        r.latency_ms !== undefined ? r.latency_ms.toString() : 'N/A',
      ]),
    ];

    const csvContent =
      'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `batch_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Layers className="h-5 w-5 text-cyan-400" />
            <span>High-Throughput Batch Recognition Studio</span>
          </h2>
          <p className="text-xs text-slate-400">
            Submit up to 8 images simultaneously for concurrent MobileNetV3 evaluation and export CSV results
          </p>
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFiles}
          multiple
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center space-x-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
        >
          <Upload className="h-3.5 w-3.5 text-cyan-400" />
          <span>Select Images (Max 8)</span>
        </button>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
        {files.length > 0 ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-medium text-slate-300">
                {files.length} image(s) queued for batch inference
              </span>
              <div className="flex space-x-2">
                <button
                  onClick={() => {
                    setFiles([]);
                    setResponse(null);
                  }}
                  className="rounded px-2.5 py-1 text-xs text-slate-400 hover:text-rose-400 transition"
                >
                  Clear Queue
                </button>
                <button
                  onClick={handleRunBatch}
                  disabled={isLoading}
                  className="flex items-center space-x-1.5 rounded-lg bg-cyan-500 px-4 py-1.5 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition"
                >
                  {isLoading && <RefreshCw className="h-3.5 w-3.5 animate-spin" />}
                  <span>{isLoading ? 'Processing Batch...' : 'Run Batch Inference'}</span>
                </button>
              </div>
            </div>

            {/* Selected Queue List */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {files.map((f, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-slate-800 bg-slate-950 p-3 text-xs text-slate-300"
                >
                  <p className="truncate font-semibold text-slate-200">{f.name}</p>
                  <p className="mt-1 text-[11px] text-slate-500">
                    {(f.size / 1024).toFixed(1)} KB • {f.type}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center justify-center p-10 text-center cursor-pointer border-2 border-dashed border-slate-800 rounded-xl hover:border-cyan-500/50 transition"
          >
            <Layers className="h-10 w-10 text-slate-600 mb-2" />
            <p className="text-sm font-semibold text-slate-200">
              Click to select up to 8 images for batch prediction
            </p>
            <p className="text-xs text-slate-500 mt-1">Supports JPG, JPEG, PNG, WEBP</p>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-center space-x-2 rounded-lg border border-rose-800/60 bg-rose-950/30 p-3 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results Table */}
        {response && (
          <div className="mt-6 space-y-4 border-t border-slate-800 pt-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 text-xs">
                <span className="text-slate-400">
                  Total: <b className="text-white">{response.total_files}</b>
                </span>
                <span className="text-emerald-400">
                  Successful: <b>{response.successful_predictions}</b>
                </span>
              </div>

              <button
                onClick={exportCSV}
                className="flex items-center space-x-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
              >
                <Download className="h-3.5 w-3.5 text-cyan-400" />
                <span>Export CSV Report</span>
              </button>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider text-[10px]">
                  <tr>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Filename</th>
                    <th className="px-4 py-3">Predicted Class</th>
                    <th className="px-4 py-3">Class Index</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                  {response.results.map((res, i) => (
                    <tr key={i} className="hover:bg-slate-800/30">
                      <td className="px-4 py-3">
                        {res.status === 'success' ? (
                          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                        ) : (
                          <XCircle className="h-4 w-4 text-rose-400" />
                        )}
                      </td>
                      <td className="px-4 py-3 font-medium text-white max-w-[150px] truncate">
                        {res.filename}
                      </td>
                      <td className="px-4 py-3 capitalize font-semibold text-cyan-300">
                        {res.predicted_class ? res.predicted_class.replace(/_/g, ' ') : (
                          <span className="text-rose-400">{res.error || 'Failed'}</span>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-400">
                        {res.class_index !== undefined ? `#${res.class_index}` : '—'}
                      </td>
                      <td className="px-4 py-3 font-mono font-medium text-slate-200">
                        {res.confidence_percent !== undefined
                          ? `${res.confidence_percent.toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-400">
                        {res.latency_ms !== undefined ? `${res.latency_ms} ms` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
