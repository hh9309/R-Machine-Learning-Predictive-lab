import React, { useState } from 'react';
import { DatasetInfo, FeatureStats } from '../../types/ml';
import { INITIAL_DATASETS } from '../../data/datasets';
import { calculateFeatureStats, calculateCorrelationMatrix } from '../../utils/mlEngine';
import { Database, Filter, Table, TrendingUp, CheckSquare, Square, Upload, Sparkles } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  onSelectDataset: (ds: DatasetInfo) => void;
  activeFeatures: string[];
  onToggleFeature: (feat: string) => void;
  onSelectAllFeatures: () => void;
  onCustomDatasetUpload?: (newDataset: DatasetInfo) => void;
}

export const Module1DatasetFeatures: React.FC<Props> = ({
  dataset,
  onSelectDataset,
  activeFeatures,
  onToggleFeature,
  onSelectAllFeatures,
}) => {
  const [activeTab, setActiveTab] = useState<'stats' | 'correlation' | 'preview'>('stats');
  const [previewPage, setPreviewPage] = useState(0);

  // Compute stats and correlation
  const featureStats: FeatureStats[] = calculateFeatureStats(dataset.data, dataset.features);
  const corrData = calculateCorrelationMatrix(dataset.data, activeFeatures);

  const pageSize = 7;
  const totalPages = Math.ceil(dataset.data.length / pageSize);
  const currentRows = dataset.data.slice(previewPage * pageSize, (previewPage + 1) * pageSize);

  return (
    <div className="space-y-6">
      {/* Header Info Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 01 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>数据接入与特征空间工程</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              数据集与特征矩阵选择 (Dataset & Feature Space)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              提供经典 R 统计基准数据（Fisher 鸢尾花、波士顿房价、威斯康星肿瘤等）。支持响应变量定义、多维特征子集过滤与皮尔逊相关性矩阵探索。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 数据源:</span>
            <code className="text-sky-700 font-semibold">{dataset.rName}</code>
          </div>
        </div>
      </div>

      {/* Dataset Selection Shelf */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            1. 选择 R 标杆数据集 (6 个内置基准)
          </span>
          <span className="text-xs text-slate-400 font-mono">
            当前样本量: {dataset.rows} | 特征数: {dataset.features.length}
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {INITIAL_DATASETS.map((ds) => {
            const isSelected = ds.id === dataset.id;
            return (
              <button
                key={ds.id}
                onClick={() => onSelectDataset(ds)}
                className={`text-left p-3 rounded-lg border transition-all text-xs ${
                  isSelected
                    ? 'bg-sky-50/70 border-sky-400 ring-2 ring-sky-400/20 shadow-xs'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/50'
                }`}
              >
                <div className="font-semibold text-slate-900 truncate">{ds.name}</div>
                <div className="font-mono text-[10px] text-sky-700 mt-0.5 truncate">{ds.rName}</div>
                <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500">
                  <span>{ds.problemType === 'classification' ? '分类任务' : '回归任务'}</span>
                  <span className="font-mono tabular-nums">{ds.rows}行</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Feature Selection Grid & Target Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Feature Matrix Filtering */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-sky-600" />
              <h3 className="text-sm font-bold text-slate-900">特征矩阵变量过滤 (Features $X$)</h3>
              <span className="text-xs text-slate-400 font-mono">
                已启用 {activeFeatures.length} / {dataset.features.length}
              </span>
            </div>
            <button
              onClick={onSelectAllFeatures}
              className="text-xs text-sky-600 hover:text-sky-800 font-medium"
            >
              全选特征
            </button>
          </div>

          <p className="text-xs text-slate-500 mb-3">
            勾选进入后续模型训练的特征列。去除共线性强或噪声特征，可显著提升模型泛化性能。
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
            {dataset.features.map((feat) => {
              const isChecked = activeFeatures.includes(feat);
              return (
                <button
                  key={feat}
                  onClick={() => onToggleFeature(feat)}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border text-left text-xs transition-colors ${
                    isChecked
                      ? 'bg-slate-50 border-slate-300 text-slate-900 font-medium'
                      : 'bg-white border-dashed border-slate-200 text-slate-400 hover:border-slate-300'
                  }`}
                >
                  {isChecked ? (
                    <CheckSquare className="w-4 h-4 text-sky-600 shrink-0" />
                  ) : (
                    <Square className="w-4 h-4 text-slate-300 shrink-0" />
                  )}
                  <span className="font-mono truncate">{feat}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right 1 Col: Target Response Variable */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">响应变量 (Response $y$)</h3>
            </div>

            <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-200 font-mono text-xs space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-slate-500">目标列名:</span>
                <span className="font-bold text-slate-900">{dataset.target}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">问题范式:</span>
                <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-100 text-sky-800">
                  {dataset.problemType === 'classification' ? '分类 (Factor)' : '回归 (Continuous)'}
                </span>
              </div>
              {dataset.classes && (
                <div>
                  <span className="text-slate-500 block mb-1">类别水平 (Levels):</span>
                  <div className="flex flex-wrap gap-1">
                    {dataset.classes.map((cls) => (
                      <span key={cls} className="bg-white px-2 py-0.5 rounded border border-slate-200 text-[10px] text-slate-700">
                        {cls}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4 text-xs text-slate-500 leading-relaxed">
              <span className="font-semibold text-slate-700">R 公式表达:</span>
              <pre className="mt-1 p-2 bg-slate-900 text-slate-100 rounded text-[11px] font-mono overflow-x-auto">
                {dataset.target} ~ {activeFeatures.join(' + ')}
              </pre>
            </div>
          </div>
        </div>
      </div>

      {/* Exploration Tabs: Descriptive Stats, Correlation Heatmap, Data Preview */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
        <div className="border-b border-slate-200 px-5 pt-3 flex items-center justify-between">
          <div className="flex items-center gap-4 text-xs font-semibold">
            <button
              onClick={() => setActiveTab('stats')}
              className={`pb-3 border-b-2 transition-colors ${
                activeTab === 'stats'
                  ? 'border-sky-600 text-sky-700'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              特征描述性统计 (Summary Stats)
            </button>
            <button
              onClick={() => setActiveTab('correlation')}
              className={`pb-3 border-b-2 transition-colors ${
                activeTab === 'correlation'
                  ? 'border-sky-600 text-sky-700'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              皮尔逊相关矩阵 (Pearson Correlation)
            </button>
            <button
              onClick={() => setActiveTab('preview')}
              className={`pb-3 border-b-2 transition-colors ${
                activeTab === 'preview'
                  ? 'border-sky-600 text-sky-700'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              原始数据预览 (Head Preview)
            </button>
          </div>

          <span className="text-xs font-mono text-slate-400 hidden sm:inline">
            stats::summary() & cor()
          </span>
        </div>

        <div className="p-5">
          {activeTab === 'stats' && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50 text-slate-500 font-mono text-[11px] border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">特征变量</th>
                    <th className="py-2.5 px-3">均值 (Mean)</th>
                    <th className="py-2.5 px-3">标准差 (SD)</th>
                    <th className="py-2.5 px-3">最小值 (Min)</th>
                    <th className="py-2.5 px-3">中位数 (Median)</th>
                    <th className="py-2.5 px-3">最大值 (Max)</th>
                    <th className="py-2.5 px-3">缺失值 (NA)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                  {featureStats.map((st) => (
                    <tr key={st.name} className="hover:bg-slate-50/50">
                      <td className="py-2.5 px-3 font-semibold text-slate-800">{st.name}</td>
                      <td className="py-2.5 px-3">{st.mean}</td>
                      <td className="py-2.5 px-3 text-slate-600">{st.sd}</td>
                      <td className="py-2.5 px-3 text-slate-500">{st.min}</td>
                      <td className="py-2.5 px-3 text-slate-700">{st.median}</td>
                      <td className="py-2.5 px-3 text-slate-500">{st.max}</td>
                      <td className="py-2.5 px-3">
                        {st.missing === 0 ? (
                          <span className="text-emerald-600">0</span>
                        ) : (
                          <span className="text-rose-600 font-bold">{st.missing}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === 'correlation' && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">
                特征两两之间的皮尔逊相关系数 $r \in [-1, +1]$。深蓝表示强正相关，深红表示强负相关。高相关特征可能导致多重共线性。
              </p>

              <div className="overflow-x-auto">
                <div className="inline-block min-w-full">
                  <table className="text-xs border-collapse font-mono tabular-nums">
                    <thead>
                      <tr>
                        <th className="p-2 border border-slate-200 bg-slate-50 text-slate-500">特征</th>
                        {corrData.features.map((f) => (
                          <th key={f} className="p-2 border border-slate-200 bg-slate-50 text-slate-600 truncate max-w-[100px]">
                            {f}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {corrData.matrix.map((row, i) => (
                        <tr key={corrData.features[i]}>
                          <td className="p-2 border border-slate-200 bg-slate-50 font-semibold text-slate-800 truncate max-w-[120px]">
                            {corrData.features[i]}
                          </td>
                          {row.map((val, j) => {
                            // Compute heatmap color: -1 (red) -> 0 (white) -> +1 (blue)
                            const intensity = Math.abs(val);
                            const bgColor =
                              val >= 0
                                ? `rgba(2, 132, 199, ${intensity * 0.7})`
                                : `rgba(225, 29, 72, ${intensity * 0.7})`;
                            const textColor = intensity > 0.5 ? '#ffffff' : '#1e293b';

                            return (
                              <td
                                key={j}
                                className="p-2.5 border border-slate-200 text-center font-semibold text-xs"
                                style={{ backgroundColor: bgColor, color: textColor }}
                                title={`${corrData.features[i]} 与 ${corrData.features[j]}: ${val}`}
                              >
                                {val.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'preview' && (
            <div className="space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-500 font-mono text-[11px] border-b border-slate-200">
                    <tr>
                      <th className="py-2 px-3">#ID</th>
                      {dataset.features.map((f) => (
                        <th key={f} className="py-2 px-3">{f}</th>
                      ))}
                      <th className="py-2 px-3 text-sky-700 bg-sky-50/50">{dataset.target} (Target)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                    {currentRows.map((row) => (
                      <tr key={row.id} className="hover:bg-slate-50/50">
                        <td className="py-2 px-3 text-slate-400">{row.id}</td>
                        {dataset.features.map((f) => (
                          <td key={f} className="py-2 px-3 text-slate-700">{row[f]}</td>
                        ))}
                        <td className="py-2 px-3 font-semibold text-sky-800 bg-sky-50/30">
                          {String(row[dataset.target])}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
                <span>
                  显示第 {previewPage * pageSize + 1} 至 {Math.min(dataset.data.length, (previewPage + 1) * pageSize)} 行 (共 {dataset.data.length} 行)
                </span>
                <div className="flex items-center gap-1">
                  <button
                    disabled={previewPage === 0}
                    onClick={() => setPreviewPage((p) => p - 1)}
                    className="px-2.5 py-1 border border-slate-200 rounded disabled:opacity-40 hover:bg-slate-50"
                  >
                    上一页
                  </button>
                  <span className="font-mono px-2">
                    {previewPage + 1} / {totalPages}
                  </span>
                  <button
                    disabled={previewPage >= totalPages - 1}
                    onClick={() => setPreviewPage((p) => p + 1)}
                    className="px-2.5 py-1 border border-slate-200 rounded disabled:opacity-40 hover:bg-slate-50"
                  >
                    下一页
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
