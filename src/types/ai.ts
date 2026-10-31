export type LLMModelType = 'gemini-3-flash' | 'deepseek-v4-pro';

export interface LLMConfig {
  apiKey: string;
  model: LLMModelType;
  baseUrl?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  modelUsed?: string;
}
