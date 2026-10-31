import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  Settings,
  Send,
  Sparkles,
  Bot,
  User,
  Key,
  ShieldAlert,
  Loader2,
  Copy,
  Check,
  RotateCcw,
  Eye,
  EyeOff,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react';
import { LLMConfig, LLMModelType, ChatMessage } from '../../types/ai';
import { callLLM } from '../../utils/aiClient';
import { DatasetInfo, SplitResult } from '../../types/ml';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
  activeModule: number;
}

const STORAGE_KEY_CONFIG = 'r_ml_llm_config_v1';

export const AIInsightsModal: React.FC<Props> = ({
  isOpen,
  onClose,
  dataset,
  splitResult,
  activeFeatures,
  activeModule,
}) => {
  // Load saved config or defaults
  const [config, setConfig] = useState<LLMConfig>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CONFIG);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch {
      // fallback
    }
    return {
      apiKey: '',
      model: 'gemini-3-flash',
      baseUrl: '',
    };
  });

  // Settings Modal Open State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  // Settings form temp state
  const [tempApiKey, setTempApiKey] = useState(config.apiKey);
  const [tempModel, setTempModel] = useState<LLMModelType>(config.model);
  const [tempBaseUrl, setTempBaseUrl] = useState(config.baseUrl || '');

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `你好！我是 **AI 洞察助手**。我已经接入你当前的 R 机器学习实验室环境：\n- **当前数据集**: \`${dataset.name}\` (${dataset.rName})\n- **响应目标**: \`${dataset.target}\` (${dataset.problemType === 'classification' ? '分类任务' : '回归任务'})\n- **选定特征**: \`${activeFeatures.join(', ')}\`\n- **数据切分**: 训练集 ${splitResult.trainData.length} 条 / 测试集 ${splitResult.testData.length} 条 (种子: ${splitResult.seed})\n\n请在右上角 **⚙️ 齿轮图标** 中输入你的 API-Key 并选择模型（支持 **Gemini 3 Flash** 与 **DeepSeek V4 Pro**），随后即可向我咨询任何数据特征、模型调优、混淆矩阵评估或 R 语言代码实现问题！`,
      timestamp: Date.now(),
    },
  ]);

  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  // Context string builder
  const experimentContext = `
- 数据集: ${dataset.name} (${dataset.rName})
- 任务类别: ${dataset.problemType}
- 响应变量 y: ${dataset.target} (水平: ${dataset.classes?.join(', ') || '连续数值'})
- 激活特征集 X: ${activeFeatures.join(', ')} (共 ${activeFeatures.length} 个特征)
- 训练集样本量: ${splitResult.trainData.length}
- 测试集样本量: ${splitResult.testData.length}
- 是否分层抽样: ${splitResult.isStratified ? '是 (Stratified)' : '否 (Random)'}
- 随机种子: ${splitResult.seed}
- 当前活动模块编号: 模块 ${activeModule}
`.trim();

  // Save Settings Handler
  const handleSaveSettings = () => {
    const newConfig: LLMConfig = {
      apiKey: tempApiKey.trim(),
      model: tempModel,
      baseUrl: tempBaseUrl.trim(),
    };
    setConfig(newConfig);
    try {
      localStorage.setItem(STORAGE_KEY_CONFIG, JSON.stringify(newConfig));
    } catch {
      // ignore
    }
    setIsSettingsOpen(false);
    setErrorMessage(null);
  };

  // Open Settings Handler
  const handleOpenSettings = () => {
    setTempApiKey(config.apiKey);
    setTempModel(config.model);
    setTempBaseUrl(config.baseUrl || '');
    setIsSettingsOpen(true);
  };

  // Send Message Handler
  const handleSendMessage = async (textToSend?: string) => {
    const content = textToSend || inputPrompt.trim();
    if (!content || isLoading) return;

    // Check if API Key is configured
    if (!config.apiKey || config.apiKey.trim() === '') {
      setErrorMessage('必须先输入 API-Key 才能调用大模型！请在下方设置中填写后继续。');
      handleOpenSettings();
      return;
    }

    setErrorMessage(null);
    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputPrompt('');
    setIsLoading(true);

    try {
      const history = [...messages, userMsg].map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const reply = await callLLM(config, history, experimentContext);

      const assistantMsg: ChatMessage = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: reply,
        timestamp: Date.now(),
        modelUsed: config.model,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setErrorMessage(err.message || '大模型调用异常，请重试。');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMsgId(id);
    setTimeout(() => setCopiedMsgId(null), 2000);
  };

  const quickPrompts = [
    '分析当前数据集特征与数据划分的合理性',
    '诊断当前 CART 与随机森林超参数的调优方向',
    '解释混淆矩阵与 ROC-AUC 中假阳性/假阴性的业务权衡',
    '检查当前实验流程是否存在数据泄露或尺度失真隐患',
    '请为当前实验生成一份精炼的 R 语言建模报告',
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-slate-900/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-4xl h-[86vh] flex flex-col overflow-hidden relative">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-gradient-to-tr from-sky-600 to-indigo-600 text-white rounded-lg shadow-xs">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900">
                  AI 洞察助手 (AI Insights Assistant)
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-100/70 text-sky-800 font-semibold border border-sky-200">
                  {config.model === 'gemini-3-flash' ? 'Gemini 3 Flash' : 'DeepSeek V4 Pro'}
                </span>
                {config.apiKey ? (
                  <span className="text-[10px] text-emerald-600 font-mono flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> API-Key 就绪
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-600 font-mono flex items-center gap-1">
                    <ShieldAlert className="w-3 h-3" /> 未输入 API-Key
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-500">
                实时关联当前实验状态切片，提供统计学解释、诊断建议与 R 代码导引。
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {/* Gear Settings Icon */}
            <button
              onClick={handleOpenSettings}
              className={`p-2 rounded-lg border transition-all ${
                !config.apiKey
                  ? 'bg-amber-50 border-amber-300 text-amber-700 animate-pulse'
                  : 'border-slate-200 bg-white hover:bg-slate-100 text-slate-600'
              }`}
              title="设置大模型与 API-Key"
            >
              <Settings className="w-4 h-4" />
            </button>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Missing API-Key Warning Banner */}
        {!config.apiKey && (
          <div className="bg-amber-50 border-b border-amber-200 px-5 py-2.5 flex items-center justify-between text-xs text-amber-800">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                本项目部署于 GitHub 浏览器端直接运行。<strong>所有大模型调用必须先输入 API-Key</strong>。
              </span>
            </div>
            <button
              onClick={handleOpenSettings}
              className="px-3 py-1 bg-amber-600 hover:bg-amber-700 text-white rounded font-medium text-[11px] shadow-xs shrink-0"
            >
              立即配置 API-Key
            </button>
          </div>
        )}

        {/* Chat Message Scroll Area */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-slate-50/40">
          {messages.map((msg) => {
            const isUser = msg.role === 'user';
            return (
              <div
                key={msg.id}
                className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-sky-600 to-indigo-600 text-white flex items-center justify-center shrink-0 text-xs shadow-xs mt-1">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-xl p-3.5 text-xs leading-relaxed space-y-1.5 shadow-xs ${
                    isUser
                      ? 'bg-slate-900 text-slate-100'
                      : 'bg-white border border-slate-200 text-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between gap-4 pb-1 border-b border-slate-100/30 text-[10px] opacity-75">
                    <span className="font-mono">
                      {isUser ? '我 (User)' : msg.modelUsed || 'AI 洞察助手'}
                    </span>
                    {!isUser && (
                      <button
                        onClick={() => handleCopyMessage(msg.id, msg.content)}
                        className="hover:opacity-100 transition-opacity flex items-center gap-1 font-mono"
                        title="复制回复内容"
                      >
                        {copiedMsgId === msg.id ? (
                          <Check className="w-3 h-3 text-emerald-600" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                        <span>{copiedMsgId === msg.id ? '已复制' : '复制'}</span>
                      </button>
                    )}
                  </div>

                  <div className="whitespace-pre-wrap font-sans text-xs">
                    {msg.content}
                  </div>
                </div>

                {isUser && (
                  <div className="w-7 h-7 rounded-full bg-slate-700 text-white flex items-center justify-center shrink-0 text-xs mt-1">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })}

          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-sky-600 to-indigo-600 text-white flex items-center justify-center shrink-0 text-xs shadow-xs">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-white border border-slate-200 rounded-xl p-3.5 text-xs text-slate-500 flex items-center gap-2 shadow-xs">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-600" />
                <span>
                  {config.model === 'gemini-3-flash' ? 'Gemini 3 Flash' : 'DeepSeek V4 Pro'}{' '}
                  正在深度分析当前实验切片...
                </span>
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 shrink-0 text-rose-600" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts Bar */}
        <div className="px-5 py-2 bg-white border-t border-slate-100 overflow-x-auto whitespace-nowrap">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono text-slate-400 shrink-0">快捷提问:</span>
            {quickPrompts.map((qp, idx) => (
              <button
                key={idx}
                disabled={isLoading}
                onClick={() => handleSendMessage(qp)}
                className="px-2.5 py-1 bg-slate-50 hover:bg-sky-50 hover:text-sky-800 border border-slate-200 rounded-md text-[11px] text-slate-600 transition-colors shrink-0"
              >
                {qp}
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <div className="p-4 bg-white border-t border-slate-200">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              placeholder={
                config.apiKey
                  ? '输入关于当前数据集、模型参数、ROC曲线或R代码的问题...'
                  : '请先在右上角齿轮设置中输入 API-Key 后方可调用...'
              }
              disabled={isLoading}
              className="flex-1 px-4 py-2.5 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 bg-slate-50/50"
            />
            <button
              type="submit"
              disabled={isLoading || !inputPrompt.trim()}
              className="px-4 py-2.5 bg-sky-600 hover:bg-sky-700 disabled:opacity-40 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs shrink-0"
            >
              <Send className="w-3.5 h-3.5" />
              <span>提问</span>
            </button>
          </form>
        </div>

        {/* ========================================================================= */}
        {/* Settings Modal (小齿轮设置大模型与 API-Key 弹窗) */}
        {/* ========================================================================= */}
        {isSettingsOpen && (
          <div className="absolute inset-0 z-50 bg-slate-900/40 backdrop-blur-2xs flex items-center justify-center p-4">
            <div className="bg-white border border-slate-200 rounded-xl shadow-xl w-full max-w-lg p-5 space-y-4 animate-in fade-in zoom-in-95 duration-150">
              {/* Settings Header */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Settings className="w-4 h-4 text-sky-600" />
                  <h4 className="text-sm font-bold text-slate-900">
                    大模型接入与 API-Key 配置 (GitHub 纯前端调用)
                  </h4>
                </div>
                <button
                  onClick={() => setIsSettingsOpen(false)}
                  className="text-slate-400 hover:text-slate-600 p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Requirement 1: 手工输入 API-Key */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label className="font-semibold text-slate-800 flex items-center gap-1">
                    <Key className="w-3.5 h-3.5 text-sky-600" />
                    <span>1. 手工输入 API-Key (必填)</span>
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="text-[11px] text-sky-600 hover:text-sky-800 flex items-center gap-1"
                  >
                    {showApiKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                    <span>{showApiKey ? '隐藏密钥' : '显示密钥'}</span>
                  </button>
                </div>

                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={tempApiKey}
                    onChange={(e) => setTempApiKey(e.target.value)}
                    placeholder={
                      tempModel === 'gemini-3-flash'
                        ? '请输入 Google Gemini API-Key (AIzaSy...)'
                        : '请输入 DeepSeek API-Key (sk-...)'
                    }
                    className="w-full px-3 py-2 text-xs font-mono border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 bg-slate-50"
                  />
                </div>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  为保障项目部署于 GitHub 后的纯客户端运行，API-Key 仅保存在您当前浏览器的本地存储 (localStorage) 中，绝不经由任何中转服务器。
                </p>
              </div>

              {/* Requirement 2: 选择两个大模型 */}
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <label className="text-xs font-semibold text-slate-800 block">
                  2. 选择目标大模型 (两个核心模型可选)
                </label>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {/* Gemini 3 Flash */}
                  <button
                    type="button"
                    onClick={() => setTempModel('gemini-3-flash')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      tempModel === 'gemini-3-flash'
                        ? 'bg-sky-50/80 border-sky-400 ring-2 ring-sky-300/40 text-sky-950 font-semibold'
                        : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-bold flex items-center justify-between">
                      <span>gemini 3 flash</span>
                      {tempModel === 'gemini-3-flash' && <Check className="w-3.5 h-3.5 text-sky-600" />}
                    </div>
                    <div className="text-[10px] text-slate-500 font-normal mt-1 leading-normal">
                      Google 最新快速推理模型，擅长代码生成与统计矩阵解析。
                    </div>
                  </button>

                  {/* DeepSeek V4 Pro */}
                  <button
                    type="button"
                    onClick={() => setTempModel('deepseek-v4-pro')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      tempModel === 'deepseek-v4-pro'
                        ? 'bg-sky-50/80 border-sky-400 ring-2 ring-sky-300/40 text-sky-950 font-semibold'
                        : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-bold flex items-center justify-between">
                      <span>deepseek-v4-pro</span>
                      {tempModel === 'deepseek-v4-pro' && <Check className="w-3.5 h-3.5 text-sky-600" />}
                    </div>
                    <div className="text-[10px] text-slate-500 font-normal mt-1 leading-normal">
                      DeepSeek 旗舰深度推理模型，数理严密，中文长上下文解析卓越。
                    </div>
                  </button>
                </div>
              </div>

              {/* Requirement 3: 确认大模型与部署说明 */}
              <div className="space-y-1.5 pt-2 border-t border-slate-100">
                <label className="text-xs font-semibold text-slate-800 block">
                  3. 确认与部署机制说明 (GitHub 纯前端模式)
                </label>
                <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200 text-[11px] text-slate-600 space-y-1 leading-relaxed">
                  <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span>已确认：GitHub 纯前端安全调用机制已就绪</span>
                  </div>
                  <p>
                    本项目部署到 GitHub Pages / 静态托管后，完全在用户浏览器沙盒中通过 Fetch 调用官方接口。未填入 API-Key 时系统严禁发起网络请求，绝不消耗未经授权的额度。
                  </p>
                </div>
              </div>

              {/* Optional Custom Base URL */}
              <div className="space-y-1 pt-1">
                <span className="text-[10px] font-mono text-slate-400 block">自定义代理接口地址 (可选，默认走官方 REST API):</span>
                <input
                  type="text"
                  value={tempBaseUrl}
                  onChange={(e) => setTempBaseUrl(e.target.value)}
                  placeholder={
                    tempModel === 'gemini-3-flash'
                      ? 'https://generativelanguage.googleapis.com/v1beta/models/...'
                      : 'https://api.deepseek.com/v1/chat/completions'
                  }
                  className="w-full px-2.5 py-1 text-[11px] font-mono border border-slate-200 rounded focus:outline-none bg-slate-50"
                />
              </div>

              {/* Buttons */}
              <div className="flex items-center justify-between pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => {
                    setTempApiKey('');
                    setTempBaseUrl('');
                  }}
                  className="text-xs text-rose-600 hover:text-rose-800 font-medium"
                >
                  清除当前密钥
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setIsSettingsOpen(false)}
                    className="px-3 py-1.5 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-lg text-xs font-medium"
                  >
                    取消
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveSettings}
                    className="px-4 py-1.5 bg-sky-600 hover:bg-sky-700 text-white rounded-lg text-xs font-semibold shadow-xs"
                  >
                    保存大模型配置
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
