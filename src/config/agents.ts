import { Agent } from '../types';

export const AGENTS: Record<string, Agent> = {
  smart: {
    name: 'Smart Chat',
    icon: 'fa-brain',
    color: 'text-teal-600',
    example: 'Find a standing desk under ₹50k with 12-mo 0% APR',
    tooltip: 'AI router - finds the best agent for your query'
  },
  offerpilot: {
    name: 'OfferPilot',
    icon: 'fa-tags',
    color: 'text-blue-600',
    example: 'Show me wireless headphones under $200',
    tooltip: 'Product search with financing options'
  },
  dispute: {
    name: 'Dispute Copilot',
    icon: 'fa-balance-scale',
    color: 'text-red-600',
    example: 'I was charged twice for the same purchase',
    tooltip: 'Credit card dispute assistance'
  },
  collections: {
    name: 'Collections',
    icon: 'fa-credit-card',
    color: 'text-indigo-600',
    example: 'I need help with payment plan options',
    tooltip: 'Hardship and payment assistance'
  },
  devcopilot: {
    name: 'DevCopilot',
    icon: 'fa-code',
    color: 'text-green-600',
    example: 'Generate Python code for payment processing',
    tooltip: 'Code generation and API documentation'
  },
  carecredit: {
    name: 'CareCredit',
    icon: 'fa-heartbeat',
    color: 'text-pink-600',
    example: 'Analyze this dental treatment estimate',
    tooltip: 'Medical/dental expense analysis'
  },
  narrator: {
    name: 'Narrator',
    icon: 'fa-chart-line',
    color: 'text-orange-600',
    example: 'Why did spend drop after 2025-07-31?',
    tooltip: 'Portfolio analytics and business insights'
  },
  imagegen: {
    name: 'ImageGen',
    icon: 'fa-image',
    color: 'text-violet-600',
    example: 'Create a futuristic city with flying cars and neon lights',
    tooltip: 'AI-powered image generation from text descriptions'
  }
};