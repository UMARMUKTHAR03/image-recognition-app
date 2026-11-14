import React from 'react';
import {
  Activity,
  Layers,
  Sparkles,
  ListOrdered,
  History,
  BookOpen,
  BarChart3,
  Settings,
  Eye,
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendOnline,
}) => {
  const tabs = [
    { id: 'studio', label: 'Vision Studio', icon: Sparkles },
    { id: 'batch', label: 'Batch Recognition', icon: Layers },
    { id: 'explain', label: 'Grad-CAM XAI', icon: Eye },
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'history', label: 'History', icon: History },
    { id: 'taxonomy', label: 'ImageNet Classes', icon: BookOpen },
    { id: 'analytics', label: 'ML Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20">
            <Eye className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-white">
                Image Recognition App
              </h1>
              <span className="rounded-full bg-cyan-500/10 px-2 py-0.5 text-xs font-semibold text-cyan-400 border border-cyan-500/20">
                PyTorch Vision
              </span>
            </div>
            <p className="text-xs text-slate-400">
              See. Recognize. Understand. • Engineered by Umar Mukthar
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 rounded-full border border-slate-800 bg-slate-950 px-3 py-1">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className="text-xs font-medium text-slate-300">
              {isBackendOnline ? 'Backend Online' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation tabs */}
      <div className="mx-auto flex max-w-7xl space-x-1 overflow-x-auto px-4 sm:px-6 scrollbar-none">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 border-b-2 px-3 py-2.5 text-xs font-medium transition-colors whitespace-nowrap ${
                isActive
                  ? 'border-cyan-400 text-cyan-400'
                  : 'border-transparent text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
