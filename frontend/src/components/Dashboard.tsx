import React from 'react';
import {
  Activity,
  Layers,
  Clock,
  Sparkles,
  BarChart2,
  CheckCircle2,
  Cpu,
} from 'lucide-react';
import { ModelInfo, PredictionResult } from '../types';

interface DashboardProps {
  modelInfo: ModelInfo | null;
  history: PredictionResult[];
  onNavigate: (tab: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  modelInfo,
  history,
  onNavigate,
}) => {
  const totalPredictions = history.length;
  const avgLatency =
    totalPredictions > 0
      ? (
          history.reduce((acc, curr) => acc + (curr.latency_ms || 0), 0) /
          totalPredictions
        ).toFixed(1)
      : '0.0';

  const avgConfidence =
    totalPredictions > 0
      ? (
          history.reduce((acc, curr) => acc + (curr.confidence_percent || 0), 0) /
          totalPredictions
        ).toFixed(1)
      : '0.0';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Activity className="h-5 w-5 text-cyan-400" />
            <span>Computer Vision Operations & System Telemetry</span>
          </h2>
          <p className="text-xs text-slate-400">
            Real-time inference metrics measured from client sessions and PyTorch backend engine
          </p>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Total Inferences
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-400">
              <BarChart2 className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white">{totalPredictions}</span>
            <p className="mt-1 text-[11px] text-slate-500">Live evaluated sessions</p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Avg Measured Latency
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
              <Clock className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white">{avgLatency} ms</span>
            <p className="mt-1 text-[11px] text-slate-500">CPU forward pass latency</p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Supported Classes
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white">
              {modelInfo?.num_classes || 1000}
            </span>
            <p className="mt-1 text-[11px] text-slate-500">ImageNet-1K Taxonomy</p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Avg Top-1 Confidence
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-500/10 text-purple-400">
              <Sparkles className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white">{avgConfidence}%</span>
            <p className="mt-1 text-[11px] text-slate-500">Softmax probability mean</p>
          </div>
        </div>
      </div>

      {/* Model Specifications & Architecture */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl lg:col-span-8">
          <h3 className="text-sm font-semibold text-white">Production Computer Vision Stack</h3>
          <p className="mt-1 text-xs text-slate-400">
            Real deep neural network topology deployed on FastAPI backend
          </p>

          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <span className="text-xs text-slate-400">Model Architecture</span>
              <p className="mt-1 font-mono text-sm font-semibold text-cyan-400">
                {modelInfo?.architecture || 'MobileNetV3-Small'}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">Hardswish activations & Squeeze-and-Excitation</p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <span className="text-xs text-slate-400">Torchvision Weights</span>
              <p className="mt-1 font-mono text-sm font-semibold text-emerald-400">
                {modelInfo?.weights || 'MobileNet_V3_Small_Weights.DEFAULT'}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">ImageNet-1K supervised pretraining</p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <span className="text-xs text-slate-400">Input Preprocessing</span>
              <p className="mt-1 font-mono text-sm font-semibold text-slate-200">
                256 resize → 224 center crop
              </p>
              <p className="mt-1 text-[11px] text-slate-500">Standard ImageNet mean & std normalization</p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <span className="text-xs text-slate-400">Explainability Saliency</span>
              <p className="mt-1 font-mono text-sm font-semibold text-slate-200">
                Grad-CAM features.12
              </p>
              <p className="mt-1 text-[11px] text-slate-500">576-channel convolutional activation map</p>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <button
              onClick={() => onNavigate('studio')}
              className="rounded-xl bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition"
            >
              Launch Recognition Studio →
            </button>
          </div>
        </div>

        {/* Quick Recent Activity */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl lg:col-span-4">
          <h3 className="text-sm font-semibold text-white">Recent Predictions</h3>
          <p className="mt-1 text-xs text-slate-400">Last 5 evaluations in session</p>

          <div className="mt-4 space-y-2.5">
            {history.length > 0 ? (
              history.slice(0, 5).map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-950 p-2.5 text-xs"
                >
                  <div className="truncate max-w-[150px]">
                    <p className="font-semibold capitalize text-slate-200 truncate">
                      {item.predicted_class.replace(/_/g, ' ')}
                    </p>
                    <p className="text-[10px] text-slate-500">{item.filename}</p>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-medium text-cyan-300">
                      {item.confidence_percent.toFixed(1)}%
                    </span>
                    <p className="text-[10px] text-slate-500">{item.latency_ms} ms</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-lg border border-dashed border-slate-800 p-6 text-center text-xs text-slate-500">
                No predictions evaluated yet.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
