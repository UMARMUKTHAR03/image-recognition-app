import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  BarChart2,
  ExternalLink,
  Eye,
} from 'lucide-react';
import { PredictionResult } from '../types';

interface ResultsViewProps {
  result: PredictionResult;
  previewUrl: string;
  onExplainClick: () => void;
}

export const ResultsView: React.FC<ResultsViewProps> = ({
  result,
  previewUrl,
  onExplainClick,
}) => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left: Original Target with Main Prediction Banner */}
        <div className="space-y-4 lg:col-span-5">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            <h3 className="text-sm font-semibold text-white">Analyzed Subject</h3>
            <p className="text-xs text-slate-400">
              {result.filename} ({result.image_dimensions.width}×{result.image_dimensions.height} px)
            </p>

            <div className="relative mt-4 flex min-h-[260px] items-center justify-center overflow-hidden rounded-xl bg-black border border-slate-800">
              <img
                src={previewUrl}
                alt={result.predicted_class}
                className="max-h-[300px] w-auto object-contain"
              />
            </div>

            <div className="mt-4 flex items-center justify-between border-t border-slate-800 pt-4">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <Clock className="h-4 w-4 text-cyan-400" />
                <span>Inference Latency:</span>
                <span className="font-mono font-semibold text-cyan-300">
                  {result.latency_ms} ms
                </span>
              </div>

              <button
                type="button"
                onClick={onExplainClick}
                className="flex items-center space-x-1.5 rounded-lg border border-cyan-500/30 bg-cyan-950/40 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-900/50 transition"
              >
                <Eye className="h-3.5 w-3.5" />
                <span>Compute Grad-CAM</span>
              </button>
            </div>
          </div>
        </div>

        {/* Right: Top-K Softmax Distribution */}
        <div className="space-y-4 lg:col-span-7">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            {/* Primary Prediction Highlight */}
            <div className="flex items-start justify-between">
              <div>
                <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Top Class Identified
                </span>
                <h2 className="text-2xl font-extrabold capitalize text-white">
                  {result.predicted_class.replace(/_/g, ' ')}
                </h2>
                <div className="mt-1 flex items-center space-x-2 text-xs text-slate-400">
                  <span>Class Index: #{result.class_index}</span>
                  <span>•</span>
                  <span>Model: {result.model}</span>
                </div>
              </div>

              <div className="flex flex-col items-end">
                <span className="text-xs text-slate-400">Confidence</span>
                <span className="text-3xl font-black text-cyan-400">
                  {result.confidence_percent.toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Warning if Low Confidence */}
            {result.is_low_confidence && (
              <div className="mt-4 flex items-center space-x-2 rounded-xl border border-amber-500/30 bg-amber-950/30 p-3 text-xs text-amber-300">
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
                <span>{result.warning || 'Low confidence prediction. Subject may be ambiguous or unrepresented in ImageNet-1K.'}</span>
              </div>
            )}

            {/* Top Predictions Ranking */}
            <div className="mt-6 space-y-3">
              <h4 className="flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <BarChart2 className="h-4 w-4 text-cyan-400" />
                <span>Top-{result.top_predictions.length} Probability Distribution</span>
              </h4>

              <div className="space-y-3">
                {result.top_predictions.map((pred, i) => (
                  <div
                    key={pred.class_index}
                    className="rounded-xl border border-slate-800/80 bg-slate-950/70 p-3 transition hover:border-slate-700"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 font-mono text-[10px] text-slate-300">
                          {i + 1}
                        </span>
                        <span className="font-semibold capitalize text-slate-200">
                          {pred.label.replace(/_/g, ' ')}
                        </span>
                        <span className="text-slate-500">#{pred.class_index}</span>
                      </div>
                      <span className="font-mono font-medium text-cyan-300">
                        {pred.confidence_percent.toFixed(2)}%
                      </span>
                    </div>

                    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          i === 0
                            ? 'bg-gradient-to-r from-cyan-500 to-blue-500'
                            : 'bg-slate-600'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(1, pred.confidence_percent))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
