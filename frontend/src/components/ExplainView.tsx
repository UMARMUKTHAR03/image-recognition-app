import React, { useRef, useState } from 'react';
import {
  Eye,
  Upload,
  AlertTriangle,
  Clock,
  Sparkles,
  Layers,
  RefreshCw,
} from 'lucide-react';
import { GradCamResult } from '../types';
import { explainImage } from '../services/api';

interface ExplainViewProps {
  initialFile?: File | null;
  initialPreviewUrl?: string | null;
}

export const ExplainView: React.FC<ExplainViewProps> = ({
  initialFile,
  initialPreviewUrl,
}) => {
  const [file, setFile] = useState<File | null>(initialFile || null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(initialPreviewUrl || null);
  const [result, setResult] = useState<GradCamResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<'overlay' | 'heatmap' | 'original'>('overlay');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleRunGradCam = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const data = await explainImage(file);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Grad-CAM calculation failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Eye className="h-5 w-5 text-cyan-400" />
            <span>Explainable AI — Grad-CAM Visual Heatmap</span>
          </h2>
          <p className="text-xs text-slate-400">
            Gradient-weighted Class Activation Mapping computes feature saliency on MobileNetV3-Small layer features.12
          </p>
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center space-x-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
        >
          <Upload className="h-3.5 w-3.5 text-cyan-400" />
          <span>Upload Different Image</span>
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left: Viewport */}
        <div className="space-y-4 lg:col-span-8">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            {/* View Mode Switcher */}
            {result && (
              <div className="mb-4 flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex space-x-1 rounded-lg bg-slate-950 p-1 border border-slate-800">
                  <button
                    onClick={() => setDisplayMode('overlay')}
                    className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                      displayMode === 'overlay'
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Heatmap Overlay
                  </button>
                  <button
                    onClick={() => setDisplayMode('heatmap')}
                    className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                      displayMode === 'heatmap'
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Raw Jet Heatmap
                  </button>
                  <button
                    onClick={() => setDisplayMode('original')}
                    className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                      displayMode === 'original'
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Original Image
                  </button>
                </div>

                <div className="flex items-center space-x-2 text-xs text-slate-400">
                  <Clock className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Computation: {result.latency_ms} ms</span>
                </div>
              </div>
            )}

            {/* Visualizer Canvas */}
            <div className="relative flex min-h-[380px] items-center justify-center overflow-hidden rounded-xl bg-black border border-slate-800">
              {result ? (
                <img
                  src={
                    displayMode === 'overlay'
                      ? result.overlay_base64
                      : displayMode === 'heatmap'
                      ? result.heatmap_base64
                      : previewUrl || ''
                  }
                  alt="Grad CAM visualization"
                  className="max-h-[440px] w-auto object-contain transition-all duration-300"
                />
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Target"
                  className="max-h-[440px] w-auto object-contain"
                />
              ) : (
                <div className="flex flex-col items-center p-8 text-center text-slate-500">
                  <Eye className="h-12 w-12 text-slate-700 mb-2" />
                  <p className="text-sm">Please select or analyze an image first to generate Grad-CAM</p>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-4 flex items-center space-x-2 rounded-lg border border-rose-800/60 bg-rose-950/30 p-3 text-xs text-rose-300">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Run Button */}
            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={handleRunGradCam}
                disabled={!file || isLoading}
                className={`flex items-center space-x-2 rounded-xl px-5 py-2.5 text-xs font-semibold shadow-lg transition ${
                  file && !isLoading
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-cyan-500/20 hover:from-cyan-400 hover:to-blue-500'
                    : 'cursor-not-allowed bg-slate-800 text-slate-500'
                }`}
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin text-white" />
                    <span>Backpropagating Class Activation...</span>
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 text-white" />
                    <span>Generate Grad-CAM Heatmap</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Technical Explanation */}
        <div className="space-y-4 lg:col-span-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl space-y-4">
            <h3 className="text-sm font-semibold text-white">How Grad-CAM Works</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Grad-CAM uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse 2D localization map highlighting important regions in the image.
            </p>

            <div className="space-y-2 border-t border-slate-800 pt-3">
              <div className="text-xs text-slate-400">Target Feature Layer</div>
              <div className="rounded bg-slate-950 p-2 font-mono text-xs text-cyan-400 border border-slate-800">
                model.features[-1] (Conv2d 576ch)
              </div>
            </div>

            {result && (
              <div className="space-y-2 border-t border-slate-800 pt-3 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Activated Class:</span>
                  <span className="font-semibold text-white capitalize">{result.target_class_label}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Class ID:</span>
                  <span className="font-mono text-cyan-400">#{result.target_class_index}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Activation Resolution:</span>
                  <span className="font-mono text-slate-200">{result.cam_resolution}</span>
                </div>
              </div>
            )}

            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
              <p className="font-semibold text-slate-300 mb-1">Color Interpretation:</p>
              <ul className="list-disc list-inside space-y-1">
                <li><b className="text-rose-400">Red / Warm:</b> High activation saliency</li>
                <li><b className="text-emerald-400">Green:</b> Moderate feature relevance</li>
                <li><b className="text-blue-400">Blue / Cool:</b> Low or neutral contribution</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
