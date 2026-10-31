import React, { useState } from 'react';
import { BookOpen, AlertTriangle, ShieldCheck, Scale, Cpu, Calculator, Check, ArrowRight } from 'lucide-react';

export const Module10KnowledgeSlices: React.FC<Props> = () => {
  const [activeSlice, setActiveSlice] = useState<'slice1' | 'slice2' | 'slice3' | 'slice4'>('slice1');

  // Interactive Slice 2: Bias-Variance Slider
  const [modelComplexity, setModelComplexity] = useState<number>(4);

  // Interactive Slice 3 Trap 2: Scale Distortion Toggle
  const [isStandardized, setIsStandardized] = useState<boolean>(false);

  // Interactive Slice 4: Cost Matrix sliders
  const [fnCost, setFnCost] = useState<number>(5000); // Cost of missing a malignant tumor
  const [fpCost, setFpCost] = useState<number>(200);  // Cost of false alarm

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 10 · 核心知识导引</span>
              <span aria-hidden="true">·</span>
              <span>方法论与建模认知切片</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              机器学习基本要素与多维切片知识导引 (Foundations & Pitfalls)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              深度解析机器学习三大核心要素、偏差与方差数学平衡、三大致命建模陷阱（数据泄露、尺度失真、不平衡悖论）以及业务成本矩阵评估。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>理论指导:</span>
            <code className="text-sky-700 font-semibold">VC-Dimension & No-Free-Lunch</code>
          </div>
        </div>
      </div>

      {/* Slice Navigation Tabs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <button
          onClick={() => setActiveSlice('slice1')}
          className={`p-3.5 rounded-xl border text-left transition-all ${
            activeSlice === 'slice1'
              ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-300/40 shadow-xs'
              : 'bg-white border-slate-200 hover:border-slate-300'
          }`}
        >
          <span className="text-[10px] font-mono text-sky-700 font-bold block mb-1">切片一</span>
          <div className="font-bold text-slate-900 text-xs">机器学习基本要素与泛化解构</div>
          <div className="text-[11px] text-slate-500 mt-1">特征、标签与损失函数</div>
        </button>

        <button
          onClick={() => setActiveSlice('slice2')}
          className={`p-3.5 rounded-xl border text-left transition-all ${
            activeSlice === 'slice2'
              ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-300/40 shadow-xs'
              : 'bg-white border-slate-200 hover:border-slate-300'
          }`}
        >
          <span className="text-[10px] font-mono text-sky-700 font-bold block mb-1">切片二</span>
          <div className="font-bold text-slate-900 text-xs">过拟合与交叉验证平衡</div>
          <div className="text-[11px] text-slate-500 mt-1">偏差-方差分解与正则化</div>
        </button>

        <button
          onClick={() => setActiveSlice('slice3')}
          className={`p-3.5 rounded-xl border text-left transition-all ${
            activeSlice === 'slice3'
              ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-300/40 shadow-xs'
              : 'bg-white border-slate-200 hover:border-slate-300'
          }`}
        >
          <span className="text-[10px] font-mono text-amber-700 font-bold block mb-1">切片三</span>
          <div className="font-bold text-slate-900 text-xs">三大致命建模陷阱</div>
          <div className="text-[11px] text-slate-500 mt-1">数据泄露、尺度失真、准确率陷阱</div>
        </button>

        <button
          onClick={() => setActiveSlice('slice4')}
          className={`p-3.5 rounded-xl border text-left transition-all ${
            activeSlice === 'slice4'
              ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-300/40 shadow-xs'
              : 'bg-white border-slate-200 hover:border-slate-300'
          }`}
        >
          <span className="text-[10px] font-mono text-indigo-700 font-bold block mb-1">切片四</span>
          <div className="font-bold text-slate-900 text-xs">误区警示与成本矩阵</div>
          <div className="text-[11px] text-slate-500 mt-1">代价敏感评估与 R 语法对照</div>
        </button>
      </div>

      {/* Slice 1 Content: Generalization Foundations */}
      {activeSlice === 'slice1' && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="text-base font-bold text-slate-900">
              【切片一：机器学习基本要素的泛化解构】
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              机器学习由**特征（Features）**、**标签（Labels）**与**损失函数（Loss Functions）**三大核心要素交织构成。其本质是通过算法在训练样本上最小化经验误差，同时约束模型复杂度以降低泛化差距，从而具备对未见数据的预测能力。
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-full bg-sky-200 text-sky-800 text-[10px] flex items-center justify-center font-mono">1</span>
                特征空间 (Features X)
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                输入变量的量化表征 X = [x_1, x_2, ..., x_p] ∈ ℝ^p。特征空间的维数决定了假设空间的搜索复杂度；高质量的低相关性特征是高准确率的先决条件。
              </p>
            </div>

            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-full bg-indigo-200 text-indigo-800 text-[10px] flex items-center justify-center font-mono">2</span>
                标签与监督信号 (Labels y)
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                分类任务中为离散因子水平 (y ∈ {'{C_1, ..., C_k}'})，回归任务中为连续实数 (y ∈ ℝ)。无监督学习（如 K-means）则剥离标签，直接探索数据流形。
              </p>
            </div>

            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-full bg-emerald-200 text-emerald-800 text-[10px] flex items-center justify-center font-mono">3</span>
                损失函数 (Loss Functions)
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                量化预测值 ŷ 与真值 y 的惩罚代价。分类常用交叉熵 (Cross-Entropy) 或 0-1 损失；回归常用均方误差 (MSE) 或绝对误差 (MAE)。
              </p>
            </div>
          </div>

          {/* Theoretical Box */}
          <div className="p-4 bg-sky-50/50 border border-sky-200 rounded-lg space-y-2 font-mono text-xs">
            <span className="font-bold text-sky-900 block">统计学习理论泛化界 (Generalization Bound):</span>
            <div className="bg-white p-3 rounded border border-sky-200 text-slate-800 text-center text-sm font-semibold">
              R_gen(h) ≤ R_emp(h) + √[ (d_VC · ln(2n / d_VC) + ln(4/δ)) / n ]
            </div>
            <p className="text-[11px] text-sky-800 font-sans leading-relaxed">
              <strong>经验风险最小化 (ERM) 困境：</strong>如果一味最小化训练集上的经验风险 (R_emp → 0)，模型将退化为死记硬背样本噪点；唯有控制模型 VC 维（复杂度）并扩充样本量 n，方能压缩泛化差距。
            </p>
          </div>
        </div>
      )}

      {/* Slice 2 Content: Overfitting & Bias-Variance Tradeoff */}
      {activeSlice === 'slice2' && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="text-base font-bold text-slate-900">
              【切片二：过拟合与交叉验证平衡】
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              过拟合表现为模型在训练集表现极佳但在测试集崩溃；通过交叉验证与正则化约束，确保模型在偏差与方差之间取得最优平衡。
            </p>
          </div>

          {/* Interactive Bias-Variance Slider */}
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-800">
                模型自由度 / 复杂度调节 ($M = {modelComplexity}$):
              </span>
              <span className="font-mono text-sky-700 font-bold">
                {modelComplexity <= 2
                  ? '高偏差 / 欠拟合状态 (Underfitting)'
                  : modelComplexity >= 6
                  ? '高方差 / 严重过拟合 (Overfitting)'
                  : '最优泛化平衡点 (Optimal Sweet Spot)'}
              </span>
            </div>

            <input
              type="range"
              min="1"
              max="8"
              value={modelComplexity}
              onChange={(e) => setModelComplexity(Number(e.target.value))}
              className="w-full accent-sky-600 cursor-pointer"
            />

            {/* Bias-Variance Visual Curve Breakdown */}
            {(() => {
              const biasSq = Math.max(0.05, Math.pow(0.85, modelComplexity) * 0.9);
              const variance = Math.max(0.05, Math.pow(1.35, modelComplexity) * 0.08);
              const totalError = biasSq + variance + 0.1; // 0.1 is irreducible noise sigma^2

              return (
                <div className="grid grid-cols-3 gap-3 pt-2 font-mono text-xs">
                  <div className="bg-white p-2.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">偏差平方 (Bias²)</span>
                    <span className="text-base font-bold text-sky-700">{biasSq.toFixed(3)}</span>
                    <span className="text-[9px] text-slate-400 block mt-0.5">模型刚性过强产生的系统误差</span>
                  </div>

                  <div className="bg-white p-2.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">方差 (Variance)</span>
                    <span className="text-base font-bold text-amber-700">{variance.toFixed(3)}</span>
                    <span className="text-[9px] text-slate-400 block mt-0.5">对训练数据扰动的过敏波动</span>
                  </div>

                  <div className="bg-white p-2.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">期望测试误差 (Total Error)</span>
                    <span className="text-base font-bold text-slate-900">{totalError.toFixed(3)}</span>
                    <span className="text-[9px] text-slate-400 block mt-0.5">Bias² + Variance + σ²</span>
                  </div>
                </div>
              );
            })()}
          </div>

          <div className="text-xs text-slate-600 leading-relaxed space-y-2">
            <span className="font-semibold text-slate-800">如何通过交叉验证解决过拟合？</span>
            <p>
              当树深度过大或 KNN 的 $K=1$ 时，模型拟合了训练集中的每一个随机噪声点。交叉验证通过将训练集划分为 $K$ 份切片轮流留出验证，能够在不接触测试集的前提下，精准捕捉到验证误差开始反弹的拐点，从而指导剪枝（`cp`）和正则化惩罚。
            </p>
          </div>
        </div>
      )}

      {/* Slice 3 Content: Three Fatal Modeling Traps */}
      {activeSlice === 'slice3' && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="text-base font-bold text-slate-900">
              【切片三：三大致命建模陷阱】
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              实际工程建模中最易发生的隐蔽灾难：数据泄露导致评估虚高；特征尺度未对齐导致距离度量彻底瘫痪；类别不平衡引发准确率盲信。
            </p>
          </div>

          <div className="space-y-4">
            {/* Trap 1 */}
            <div className="p-4 bg-rose-50/50 border border-rose-200 rounded-lg space-y-2">
              <div className="flex items-center gap-2 text-rose-900 font-bold text-xs">
                <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                <span>陷阱 1：数据泄露 (Data Leakage) 导致性能虚高</span>
              </div>
              <p className="text-xs text-rose-800 leading-relaxed">
                <strong>错误做法：</strong>在执行 Train/Test 划分前，对整个数据集直接调用 <code>scale(df)</code> 计算全集均值 μ_all 和方差 σ_all。
                <br />
                <strong>灾难后果：</strong>测试集的数据分布隐式“泄露”进了训练过程，导致本地测试集评估极其完美，但上线遇到真正未见数据时预测瞬间崩溃。
                <br />
                <strong>正确原则：</strong>所有预处理必须在隔离流水线内部进行，参数仅从训练集拟合（Fit on Train, Transform on Test）。
              </p>
            </div>

            {/* Trap 2: Interactive Scale Distortion Demonstration */}
            <div className="p-4 bg-amber-50/50 border border-amber-200 rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                  <Scale className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>陷阱 2：未标准化导致距离算法严重失真 (Scale Distortion)</span>
                </div>
                <button
                  onClick={() => setIsStandardized(!isStandardized)}
                  className={`px-3 py-1 rounded text-xs font-mono font-semibold transition-all ${
                    isStandardized
                      ? 'bg-emerald-600 text-white'
                      : 'bg-amber-200 text-amber-900 hover:bg-amber-300'
                  }`}
                >
                  {isStandardized ? '✔ 已执行 scale() 标准化' : '未标准化 (原始量纲)'}
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                <div className="bg-white p-2.5 rounded border border-amber-200">
                  <span className="text-[10px] text-slate-500 block">特征 A：年收入 (0 ~ 100,000 元)</span>
                  <div className="text-amber-950 font-bold mt-1">
                    {isStandardized ? 'Z-Score: [-2.1, +2.4]' : '数值跨度: Δ = 85,000'}
                  </div>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {isStandardized ? '对欧氏距离贡献权重: ~50%' : '对欧氏距离贡献权重: > 99.9% (完全主导)'}
                  </span>
                </div>

                <div className="bg-white p-2.5 rounded border border-amber-200">
                  <span className="text-[10px] text-slate-400 block">特征 B：违约率 (0.00 ~ 1.00 比例)</span>
                  <div className="text-amber-950 font-bold mt-1">
                    {isStandardized ? 'Z-Score: [-1.8, +2.2]' : '数值跨度: Δ = 0.85'}
                  </div>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {isStandardized ? '对欧氏距离贡献权重: ~50%' : '对欧氏距离贡献权重: < 0.01% (被彻底淹没)'}
                  </span>
                </div>
              </div>

              <p className="text-[11px] text-amber-800 leading-relaxed">
                <strong>核心机理：</strong>KNN 和 K-means 等算法依赖空间欧氏距离 d = √[ Σ(x1 - x2)² ]。若未标准化，大尺度特征会以百万倍方差绝对碾压小尺度特征，导致其他重要特征完全失去投票权！
              </p>
            </div>

            {/* Trap 3 */}
            <div className="p-4 bg-sky-50/50 border border-sky-200 rounded-lg space-y-2">
              <div className="flex items-center gap-2 text-sky-900 font-bold text-xs">
                <ShieldCheck className="w-4 h-4 text-sky-600 shrink-0" />
                <span>陷阱 3：在类别极度不平衡时仅迷信总体准确率 (Accuracy Paradox)</span>
              </div>
              <p className="text-xs text-sky-800 leading-relaxed">
                在罕见疾病诊断或金融反欺诈场景中，正样本（欺诈）占比可能仅为 0.5%。一个毫无智能的模型若全猜“正常”，其总体准确率高达 99.5%，但召回率（Recall）为 0！此时必须强制结合精确率、召回率、F1 值与 ROC-AUC 综合裁决。
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Slice 4 Content: Misconceptions & Business Cost Matrix */}
      {activeSlice === 'slice4' && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="text-base font-bold text-slate-900">
              【切片四：误区警示与业务成本矩阵计算器】
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              警惕“高准确率等于绝对好模型”：在医学诊断、设备故障预测等真实业务场景中，必须结合代价敏感矩阵综合权衡。
            </p>
          </div>

          {/* Interactive Cost Matrix Simulation */}
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <Calculator className="w-4 h-4 text-sky-600" />
                临床诊断业务损失代价矩阵 (Cost-Benefit Matrix)
              </span>
              <span className="text-[10px] font-mono text-slate-400">单位: 元 / 案例</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5 bg-white p-3 rounded border border-slate-200">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-rose-700 font-semibold">伪负例代价 (FN / 恶性肿瘤漏诊)</span>
                  <span className="font-mono font-bold text-rose-700">¥{fnCost}</span>
                </div>
                <input
                  type="range"
                  min="1000"
                  max="20000"
                  step="500"
                  value={fnCost}
                  onChange={(e) => setFnCost(Number(e.target.value))}
                  className="w-full accent-rose-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 block">危及生命安全，代价极高</span>
              </div>

              <div className="space-y-1.5 bg-white p-3 rounded border border-slate-200">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-amber-700 font-semibold">伪正例代价 (FP / 误报警复查)</span>
                  <span className="font-mono font-bold text-amber-700">¥{fpCost}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={fpCost}
                  onChange={(e) => setFpCost(Number(e.target.value))}
                  className="w-full accent-amber-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 block">二次生化化验费用，代价较低</span>
              </div>
            </div>

            {/* Optimal Decision Rule */}
            <div className="p-3 bg-indigo-50/70 border border-indigo-200 rounded text-xs text-indigo-950 font-mono space-y-1">
              <div className="font-bold">最优决策阈值计算准则:</div>
              <div className="text-[11px] text-indigo-900 font-sans">
                理论最优分类截断阈值 P* = C_FP / (C_FP + C_FN) = {fpCost} / ({fpCost} + {fnCost}) ≈ {(fpCost / (fpCost + fnCost)).toFixed(3)}
              </div>
              <p className="text-[10px] text-indigo-800 font-sans mt-1">
                因为漏诊恶性肿瘤代价远高于误报，所以最优决策阈值不应使用默认的 0.50，而应下调至 <strong>{(fpCost / (fpCost + fnCost)).toFixed(3)}</strong>，采取宁可误判不可漏诊的敏感策略！
              </p>
            </div>
          </div>

          {/* R Ecosystem Cheat Sheet */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <span className="text-xs font-semibold text-slate-800 block">
              R 语言常用经典建模包语法速查对照表:
            </span>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono border-collapse">
                <thead className="bg-slate-50 text-slate-500 border-b border-slate-200">
                  <tr>
                    <th className="py-2 px-3 text-left">功能场景</th>
                    <th className="py-2 px-3 text-left">R 包名称</th>
                    <th className="py-2 px-3 text-left">标准调用代码</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-[11px]">
                  <tr>
                    <td className="py-2 px-3 font-semibold text-slate-800 font-sans">数据分层切分</td>
                    <td className="py-2 px-3 text-sky-700">caret</td>
                    <td className="py-2 px-3 text-slate-600">createDataPartition(y, p = 0.8, list = FALSE)</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-slate-800 font-sans">CART 决策树</td>
                    <td className="py-2 px-3 text-sky-700">rpart</td>
                    <td className="py-2 px-3 text-slate-600">rpart(y ~ ., data = train, control = rpart.control(cp=0.01))</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-slate-800 font-sans">随机森林集成</td>
                    <td className="py-2 px-3 text-sky-700">randomForest</td>
                    <td className="py-2 px-3 text-slate-600">randomForest(y ~ ., data = train, ntree = 100, mtry = 3)</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-slate-800 font-sans">ROC 与 AUC 评估</td>
                    <td className="py-2 px-3 text-sky-700">pROC</td>
                    <td className="py-2 px-3 text-slate-600">roc(actual, predicted_prob); auc(roc_obj)</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-slate-800 font-sans">偏依赖可解释图</td>
                    <td className="py-2 px-3 text-sky-700">pdp</td>
                    <td className="py-2 px-3 text-slate-600">partial(model, pred.var = "feat", train = train_data)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
interface Props {}
