import React, { useEffect, useState } from 'react';
import { BookOpen, Search, Layers, RefreshCw } from 'lucide-react';
import { ImageNetClass } from '../types';
import { getClasses } from '../services/api';

export const TaxonomyView: React.FC = () => {
  const [classes, setClasses] = useState<ImageNetClass[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadClasses();
  }, []);

  const loadClasses = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getClasses();
      setClasses(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch ImageNet taxonomy');
    } finally {
      setIsLoading(false);
    }
  };

  const filtered = classes.filter(
    (c) =>
      c.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.index.toString().includes(searchTerm)
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <BookOpen className="h-5 w-5 text-cyan-400" />
            <span>ImageNet-1K Taxonomy (1,000 Classes)</span>
          </h2>
          <p className="text-xs text-slate-400">
            Browse the genuine 1,000 object categories mapped to MobileNetV3-Small output logits
          </p>
        </div>

        <button
          onClick={loadClasses}
          disabled={isLoading}
          className="flex items-center space-x-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
        >
          <RefreshCw className={`h-3.5 w-3.5 text-cyan-400 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Reload</span>
        </button>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl space-y-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search all 1,000 ImageNet categories by label or class ID (e.g. golden retriever, tench, 207)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full rounded-xl border border-slate-800 bg-slate-950 py-2.5 pl-10 pr-4 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <span>Showing {filtered.length} of {classes.length} total categories</span>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center p-12 text-xs text-slate-400">
            <RefreshCw className="h-5 w-5 animate-spin text-cyan-400 mr-2" />
            <span>Loading ImageNet dictionary from backend...</span>
          </div>
        ) : error ? (
          <div className="rounded-xl border border-rose-900 bg-rose-950/40 p-4 text-xs text-rose-300">
            {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 max-h-[580px] overflow-y-auto pr-1">
            {filtered.map((item) => (
              <div
                key={item.index}
                className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-950 p-2.5 text-xs transition hover:border-slate-700"
              >
                <span className="capitalize text-slate-200 font-medium truncate mr-2">
                  {item.label.replace(/_/g, ' ')}
                </span>
                <span className="font-mono text-[10px] text-cyan-400/80 bg-cyan-950/50 px-1.5 py-0.5 rounded border border-cyan-800/30">
                  #{item.index}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
