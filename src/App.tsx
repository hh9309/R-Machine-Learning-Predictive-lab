import React, { useState, useMemo } from 'react';
import { INITIAL_DATASETS } from './data/datasets';
import { DatasetInfo, SplitResult } from './types/ml';
import { splitDataset } from './utils/mlEngine';
import { generateRScript } from './utils/rCodeGenerator';

// Modules
import { Module1DatasetFeatures } from './components/modules/Module1DatasetFeatures';
import { Module2DataSplit } from './components/modules/Module2DataSplit';
import { Module3TreesForest } from './components/modules/Module3TreesForest';
import { Module4KNNClassifier } from './components/modules/Module4KNNClassifier';
import { Module5KMeansClustering } from './components/modules/Module5KMeansClustering';
import { Module6CrossValidation } from './components/modules/Module6CrossValidation';
import { Module7ConfusionROC } from './components/modules/Module7ConfusionROC';
import { Module8FeatureImportance } from './components/modules/Module8FeatureImportance';
import { Module9PredictionVis } from './components/modules/Module9PredictionVis';
import { Module10KnowledgeSlices } from './components/modules/Module10KnowledgeSlices';

// Modal
import { RScriptModal } from './components/RScriptModal';
import { AIInsightsModal } from './components/ai/AIInsightsModal';
import { ReportExportModal } from './components/ReportExportModal';

// Icons
import {
  FileCode2,
  FileText,
  ChevronLeft,
  ChevronRight,
  Database,
  GitFork,
  Network,
  Target,
  Layers,
  GitCommit,
  Activity,
  BarChart,
  PlayCircle,
  BookOpen,
  Sparkles,
} from 'lucide-react';

export default function App() {
  // Global State: Current Dataset
  const [dataset, setDataset] = useState<DatasetInfo>(INITIAL_DATASETS[0]);

  // Global State: Active Feature Subset
  const [activeFeatures, setActiveFeatures] = useState<string[]>(dataset.features);

  // Global State: Split Config
  const [splitRatio, setSplitRatio] = useState<number>(0.7);
  const [isStratified, setIsStratified] = useState<boolean>(true);
  const [seed, setSeed] = useState<number>(42);

  // Active Module (1 to 10)
  const [activeModule, setActiveModule] = useState<number>(1);

  // R Script Export Modal
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);

  // AI Insights Assistant Modal
  const [isAIAssistantOpen, setIsAIAssistantOpen] = useState<boolean>(false);

  // Report Export & Datasets Download Modal
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);

  // When dataset changes, update active features to all features of that dataset
  const handleSelectDataset = (newDs: DatasetInfo) => {
    setDataset(newDs);
    setActiveFeatures(newDs.features);
  };

  // Toggle single feature
  const handleToggleFeature = (feat: string) => {
    if (activeFeatures.includes(feat)) {
      if (activeFeatures.length <= 1) return; // Keep at least one feature
      setActiveFeatures(activeFeatures.filter((f) => f !== feat));
    } else {
      setActiveFeatures([...activeFeatures, feat]);
    }
  };

  const handleSelectAllFeatures = () => {
    setActiveFeatures(dataset.features);
  };

  const handleUpdateSplit = (ratio: number, stratified: boolean, newSeed: number) => {
    setSplitRatio(ratio);
    setIsStratified(stratified);
    setSeed(newSeed);
  };

  // Perform Train / Test Split
  const splitResult: SplitResult = useMemo(() => {
    return splitDataset(
      dataset.data,
      dataset.target,
      splitRatio,
      isStratified,
      seed,
      dataset.problemType === 'classification'
    );
  }, [dataset.data, dataset.target, splitRatio, isStratified, seed, dataset.problemType]);

  // Generated R Script
  const rScript = useMemo(() => {
    return generateRScript(
      dataset,
      activeFeatures,
      splitRatio,
      isStratified,
      seed,
      activeModule === 3 ? 'rf' : activeModule === 4 ? 'knn' : activeModule === 5 ? 'kmeans' : 'cart',
      { maxDepth: 4, cp: 0.01, ntree: 50, mtry: 2, k: 5, clusters: 3, cvFolds: 5 }
    );
  }, [dataset, activeFeatures, splitRatio, isStratified, seed, activeModule]);

  // 10 Core Modules Definition
  const modules = [
    { id: 1, name: '1. 特征选择', short: '特征工程', icon: Database, desc: '多源数据与特征矩阵过滤' },
    { id: 2, name: '2. 数据划分', short: '数据切分', icon: GitFork, desc: '分层抽样与分布对齐' },
    { id: 3, name: '3. 树与森林', short: '决策树森林', icon: Network, desc: 'CART剪枝与随机森林OOB' },
    { id: 4, name: '4. KNN 分类', short: 'KNN近邻', icon: Target, desc: '度量空间与动态投票探针' },
    { id: 5, name: '5. 聚类分析', short: 'K-means', icon: Layers, desc: '逐步演播与肘部法则' },
    { id: 6, name: '6. 交叉验证', short: 'CV调优', icon: GitCommit, desc: 'K折切片与网格搜索' },
    { id: 7, name: '7. 混淆ROC', short: '混淆ROC', icon: Activity, desc: '动态阈值与AUC面积' },
    { id: 8, name: '8. 特征解释', short: '模型解释', icon: BarChart, desc: '变量重要性与偏依赖PDP' },
    { id: 9, name: '9. 预测可视化', short: '预测诊断', icon: PlayCircle, desc: '残差散点与即时推演台' },
    { id: 10, name: '10. 知识导引', short: '核心知识', icon: BookOpen, desc: '四大切片与三大建模陷阱' },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* 3-Zone Top Bar Contract */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-8">
          {/* Zone 1: Main Title + 报告导出 切片 + AI大模型配置入口 */}
          <div className="flex items-center gap-2.5 shrink-0">
            <span className="text-base font-bold tracking-tight text-slate-900 whitespace-nowrap">
              R-Machine-Learning-lab
            </span>
            <span className="text-xs text-slate-400 font-mono hidden xl:inline">
              R 机器学习与预测分析 WebApp 实验室
            </span>

            {/* AI 洞察助手入口 (已移除"AI洞察助手"文字，保留快捷图标与全部功能) */}
            <button
              onClick={() => setIsAIAssistantOpen(true)}
              className="p-1.5 text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200 transition-colors shadow-2xs cursor-pointer ml-1"
              title="大模型智能诊断与配置 (Gemini 3 Flash / DeepSeek V4 Pro)"
              aria-label="AI 智能分析设置"
            >
              <Sparkles className="w-3.5 h-3.5 text-indigo-600 animate-pulse" />
            </button>

            {/* 报告导出 切片 */}
            <button
              onClick={() => setIsReportModalOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200 transition-colors shadow-2xs cursor-pointer"
              title="打开实验报告导出切片：预览6大核心部分报告与下载六大案例原始数据"
            >
              <FileText className="w-3.5 h-3.5 text-emerald-600" />
              <span>报告导出</span>
            </button>
          </div>

          {/* Zone 2: Navigation Links */}
          <nav className="hidden lg:flex items-center gap-6 text-xs font-medium text-slate-600">
            <button
              onClick={() => setActiveModule(1)}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
                activeModule <= 2 ? 'text-sky-700 font-bold' : ''
              }`}
            >
              数据工程
            </button>
            <button
              onClick={() => setActiveModule(3)}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
                activeModule >= 3 && activeModule <= 5 ? 'text-sky-700 font-bold' : ''
              }`}
            >
              算法建模
            </button>
            <button
              onClick={() => setActiveModule(6)}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
                activeModule >= 6 && activeModule <= 9 ? 'text-sky-700 font-bold' : ''
              }`}
            >
              评估与推演
            </button>
            <button
              onClick={() => setActiveModule(10)}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
                activeModule === 10 ? 'text-sky-700 font-bold' : ''
              }`}
            >
              学术知识切片
            </button>
            <button
              onClick={() => setIsAIAssistantOpen(true)}
              className="hover:text-indigo-600 transition-colors whitespace-nowrap text-indigo-600 font-semibold flex items-center gap-1"
            >
              <Sparkles className="w-3 h-3 text-indigo-600" />
              <span>AI 洞察</span>
            </button>
          </nav>

          {/* Zone 3: 1 Primary Action */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => setIsExportModalOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap shadow-xs"
            >
              <FileCode2 className="w-3.5 h-3.5 text-sky-400" />
              <span>运行与导出 R 代码 (R Console)</span>
            </button>
          </div>
        </div>

        {/* Row 2: 10 Core Modules Navigation Shelf - 全部模块居中 */}
        <div className="w-full bg-white border-t border-slate-200 px-4 sm:px-6 lg:px-8 overflow-x-auto">
          <div className="w-fit mx-auto flex items-center justify-center gap-1.5 py-1.5 min-w-max">
            {modules.map((m) => {
              const isActive = activeModule === m.id;
              const Icon = m.icon;
              return (
                <button
                  key={m.id}
                  onClick={() => setActiveModule(m.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                    isActive
                      ? 'bg-sky-50 text-sky-800 font-bold border border-sky-300/80 ring-1 ring-sky-300/40 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70 border border-transparent'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-sky-600' : 'text-slate-400'}`} />
                  <span>{m.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Row 3: Global Pipeline Telemetry Ribbon - 以中间页面边界为基准向右对齐居中呈现 */}
        <div className="w-full bg-slate-100/80 border-t border-slate-200/80 px-4 sm:px-6 lg:px-8 py-1.5 overflow-x-auto">
          <div className="w-fit mx-auto flex items-center justify-center gap-3 sm:gap-6 text-[11px] font-mono text-slate-600 min-w-max">
            <div className="flex items-center gap-2 bg-white/95 px-3 py-1 rounded-md border border-slate-200/90 shadow-2xs">
              <span className="text-slate-400">当前数据集:</span>
              <span className="font-semibold text-slate-800">{dataset.name}</span>
              <span className="text-slate-300">|</span>
              <span className="text-slate-400">响应变量:</span>
              <span className="text-sky-700 font-semibold">{dataset.target}</span>
              <span className="text-slate-300">|</span>
              <span className="text-slate-400">特征维度:</span>
              <span className="text-slate-700 font-medium">{activeFeatures.length} 维</span>
            </div>

            <div className="flex items-center gap-2 bg-white/95 px-3 py-1 rounded-md border border-slate-200/90 shadow-2xs">
              <span className="text-slate-400">划分比例:</span>
              <span className="text-slate-700 font-medium">
                {splitResult.trainData.length} 训练集 / {splitResult.testData.length} 测试集
              </span>
              <span className="text-slate-300">|</span>
              <span className="text-slate-400">随机种子:</span>
              <span className="text-slate-700 font-medium">set.seed({seed})</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace Stage */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeModule === 1 && (
          <Module1DatasetFeatures
            dataset={dataset}
            onSelectDataset={handleSelectDataset}
            activeFeatures={activeFeatures}
            onToggleFeature={handleToggleFeature}
            onSelectAllFeatures={handleSelectAllFeatures}
          />
        )}

        {activeModule === 2 && (
          <Module2DataSplit
            dataset={dataset}
            splitResult={splitResult}
            onUpdateSplit={handleUpdateSplit}
            onSelectDataset={handleSelectDataset}
          />
        )}

        {activeModule === 3 && (
          <Module3TreesForest
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 4 && (
          <Module4KNNClassifier
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 5 && (
          <Module5KMeansClustering
            dataset={dataset}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 6 && (
          <Module6CrossValidation
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 7 && (
          <Module7ConfusionROC
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 8 && (
          <Module8FeatureImportance
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 9 && (
          <Module9PredictionVis
            dataset={dataset}
            splitResult={splitResult}
            activeFeatures={activeFeatures}
          />
        )}

        {activeModule === 10 && (
          <Module10KnowledgeSlices />
        )}

        {/* Pipeline Navigation Step Footer */}
        <div className="mt-8 pt-4 border-t border-slate-200 flex items-center justify-between">
          <button
            disabled={activeModule <= 1}
            onClick={() => setActiveModule((m) => Math.max(1, m - 1))}
            className="flex items-center gap-1.5 px-4 py-2 border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-40 rounded-lg text-xs font-semibold text-slate-700 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>上一步: {modules[Math.max(0, activeModule - 2)]?.short}</span>
          </button>

          <span className="text-xs font-mono text-slate-400">
            模块 {activeModule} / 10 · {modules[activeModule - 1]?.desc}
          </span>

          <button
            disabled={activeModule >= 10}
            onClick={() => setActiveModule((m) => Math.min(10, m + 1))}
            className="flex items-center gap-1.5 px-4 py-2 bg-sky-600 hover:bg-sky-700 disabled:opacity-40 text-white rounded-lg text-xs font-semibold transition-colors shadow-xs"
          >
            <span>下一步: {modules[Math.min(9, activeModule)]?.short}</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </main>

      {/* R Script Interactive Execution Workbench Modal */}
      <RScriptModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        rScript={rScript}
        dataset={dataset}
        splitResult={splitResult}
        activeFeatures={activeFeatures}
        algorithm={activeModule === 3 ? 'rf' : activeModule === 4 ? 'knn' : activeModule === 5 ? 'kmeans' : 'cart'}
      />

      {/* AI Insights Assistant Modal */}
      <AIInsightsModal
        isOpen={isAIAssistantOpen}
        onClose={() => setIsAIAssistantOpen(false)}
        dataset={dataset}
        splitResult={splitResult}
        activeFeatures={activeFeatures}
        activeModule={activeModule}
      />

      {/* Report Export & Benchmark Cases Data Hub Modal */}
      <ReportExportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        dataset={dataset}
        splitResult={splitResult}
        activeFeatures={activeFeatures}
        activeModule={activeModule}
      />
    </div>
  );
}
