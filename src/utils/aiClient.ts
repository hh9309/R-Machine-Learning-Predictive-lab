import { LLMConfig, ChatMessage } from '../types/ai';

const SYSTEM_INSTRUCTION = `你是一位精通 R 语言统计学与机器学习（caret, rpart, randomForest, pROC, stats, ggplot2）的资深 AI 洞察助手。
你正在为用户提供在线实验诊断与建模指导。
请基于用户当前在 WebApp 实验室中的数据集状态、特征矩阵选择、训练/测试集切分、模型算法参数和评估结果，提供精准、结构清晰、具备统计学依据的回答。
如果提供 R 代码示例，请确保代码规范严谨，并且图表标题、图例与坐标轴全部采用英文。回答语言为简体中文。`;

export async function callLLM(
  config: LLMConfig,
  messages: { role: string; content: string }[],
  experimentContext: string
): Promise<string> {
  if (!config.apiKey || config.apiKey.trim() === '') {
    throw new Error('未检测到 API-Key。所有大模型调用必须先在齿轮设置中输入 API-Key！');
  }

  const cleanKey = config.apiKey.trim();

  // Combine conversation with current experiment context
  const contextMessage = `【当前 R 机器学习实验室状态切片】:\n${experimentContext}\n\n请结合上述实验上下文回答以下用户问题。`;

  if (config.model === 'gemini-3-flash') {
    // Client-side direct call to Google Gemini API
    // Using gemini-2.5-flash / gemini-3.8-flash endpoint
    const endpoint = config.baseUrl && config.baseUrl.trim() !== ''
      ? config.baseUrl.trim()
      : `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${cleanKey}`;

    // Format messages for Gemini contents API
    const contents = [
      {
        role: 'user',
        parts: [{ text: `${SYSTEM_INSTRUCTION}\n\n${contextMessage}` }],
      },
      ...messages.map((m) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      })),
    ];

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 2048,
        },
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      const msg = errJson.error?.message || `HTTP ${response.status} ${response.statusText}`;
      if (response.status === 400 || response.status === 403) {
        throw new Error(`Gemini API 认证失败: ${msg}。请检查输入的 API-Key 是否正确。`);
      } else if (response.status === 429) {
        throw new Error(`Gemini API 配额已耗尽或请求频率超限: ${msg}`);
      }
      throw new Error(`Gemini 调用失败: ${msg}`);
    }

    const data = await response.json();
    const candidateText = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!candidateText) {
      throw new Error('模型未返回有效文本内容。');
    }
    return candidateText;
  } else {
    // deepseek-v4-pro
    // Client-side direct call to DeepSeek API (OpenAI-compatible format)
    const endpoint = config.baseUrl && config.baseUrl.trim() !== ''
      ? config.baseUrl.trim()
      : 'https://api.deepseek.com/v1/chat/completions';

    const formattedMessages = [
      { role: 'system', content: `${SYSTEM_INSTRUCTION}\n\n${contextMessage}` },
      ...messages.map((m) => ({
        role: m.role === 'assistant' ? 'assistant' : 'user',
        content: m.content,
      })),
    ];

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${cleanKey}`,
      },
      body: JSON.stringify({
        model: 'deepseek-chat', // DeepSeek standard model endpoint
        messages: formattedMessages,
        temperature: 0.7,
        max_tokens: 2048,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      const msg = errJson.error?.message || `HTTP ${response.status} ${response.statusText}`;
      if (response.status === 401 || response.status === 403) {
        throw new Error(`DeepSeek API 认证失败: ${msg}。请检查输入的 API-Key 是否有效。`);
      } else if (response.status === 429) {
        throw new Error(`DeepSeek API 请求频率过高或余额不足: ${msg}`);
      }
      throw new Error(`DeepSeek 调用失败: ${msg}`);
    }

    const data = await response.json();
    const reply = data.choices?.[0]?.message?.content;
    if (!reply) {
      throw new Error('DeepSeek 模型未返回有效回复。');
    }
    return reply;
  }
}
