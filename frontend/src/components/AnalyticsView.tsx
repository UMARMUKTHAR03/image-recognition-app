import React from 'react';
import { BarChart3, AlertCircle, CheckCircle2, TrendingUp, Layers } from 'lucide-react';
import { ModelInfo } from '../types';

interface AnalyticsViewProps {
  modelInfo: ModelInfo | null;
}

export const AnalyticsView: React.FC<AnalyticsViewProps> = ({ modelInfo }) => {
  const hasCustom = modelInfo?.has_custom_artifacts;
  const customEval = modelInfo?.custom_evaluation;
  const customHist = modelInfo?.custom_training_history;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <span>Machine Learning Training & Validation Analytics</span>
          </h2>
          <p className="text-xs text-slate-400">
            Performance metrics, loss curves, and validation reports produced by backend transfer learning pipeline
          </p>
        </div>
      </div>

      {hasCustom && customEval ? (
        <div className="space-y-6">
          {/* Custom Evaluation Summary Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Accuracy</span>
              <p className="mt-2 text-3xl font-extrabold text-cyan-400">
                {(customEval.metrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Precision</span>
              <p className="mt-2 text-3xl font-extrabold text-emerald-400">
                {(customEval.metrics.precision * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Recall</span>
              <p className="mt-2 text-3xl font-extrabold text-blue-400">
                {(customEval.metrics.recall * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <span className="text-xs text-slate-400 uppercase tracking-wider">F1-Score</span>
              <p className="mt-2 text-3xl font-extrabold text-purple-400">
                {(customEval.metrics.f1_score * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Confusion Matrix */}
          {customEval.confusion_matrix && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
              <h3 className="text-sm font-semibold text-white">Validation Confusion Matrix</h3>
              <p className="mt-1 text-xs text-slate-400">
                True class vs predicted class across validation splits
              </p>
              <div className="mt-4 overflow-x-auto">
                <table className="border-collapse text-center text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-slate-500">True \ Pred</th>
                      {customEval.classes.map((c: string, idx: number) => (
                        <th key={idx} className="p-2 text-cyan-400 font-semibold capitalize">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {customEval.confusion_matrix.map((row: number[], rIdx: number) => (
                      <tr key={rIdx}>
                        <td className="p-2 text-left font-semibold text-slate-300 capitalize">
                          {customEval.classes[rIdx]}
                        </td>
                        {row.map((val: number, cIdx: number) => (
                          <td
                            key={cIdx}
                            className={`p-3 font-mono ${
                              rIdx === cIdx ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400'
                            }`}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-8 shadow-xl space-y-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-5 w-5 text-cyan-400 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-white">
                Pretrained ImageNet-1K Default Inference Mode
              </h3>
              <p className="mt-1 text-xs text-slate-300 leading-relaxed">
                The production server is currently executing inference using official torchvision <b>MobileNet_V3_Small_Weights.DEFAULT</b> trained on ImageNet-1K (1,000 object categories).
              </p>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                To train and evaluate a custom classifier on your own domain-specific dataset:
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 font-mono text-xs text-slate-300 space-y-2">
            <p className="text-slate-500"># 1. Structure dataset in standard ImageFolder format:</p>
            <p className="text-cyan-400">dataset/train/&lt;class_a&gt;/, dataset/val/&lt;class_a&gt;/</p>
            <p className="text-slate-500 mt-2"># 2. Execute transfer learning pipeline:</p>
            <p className="text-cyan-400">python -m backend.training.train</p>
            <p className="text-slate-500 mt-2"># 3. Artifacts and evaluation curves will be generated into:</p>
            <p className="text-slate-400">backend/artifacts/evaluation_metrics.json</p>
          </div>
        </div>
      )}
    </div>
  );
};
