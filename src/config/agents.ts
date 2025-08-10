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
    example: 'Show me laptops under ₹80k with financing options',
    tooltip: 'Product search with financing pre-qualification'
  },
  trustshield: {
    name: 'TrustShield',
    icon: 'fa-shield-halved',
    color: 'text-red-600',
    example: 'I got an email asking for gift cards as refund',
    tooltip: 'Fraud detection and PII protection system'
  },
  dispute: {
    name: 'Dispute',
    icon: 'fa-gavel',
    color: 'text-amber-600',
    example: 'Charged twice for ₹12,499 at Amazon on Dec 15th',
    tooltip: 'Transaction dispute resolution assistant'
  },
  collections: {
    name: 'Collections',
    icon: 'fa-handshake',
    color: 'text-green-600',
    example: 'I have ₹25k balance at 24% APR, need payment options',
    tooltip: 'Hardship and payment plan assistance'
  },
  contracts: {
    name: 'Contracts',
    icon: 'fa-file-contract',
    color: 'text-purple-600',
    example: 'Review my merchant agreement for key obligations',
    tooltip: 'Contract analysis and obligation tracking'
  },
  devcopilot: {
    name: 'DevCopilot',
    icon: 'fa-code',
    color: 'text-indigo-600',
    example: 'Generate Python code for payments API integration',
    tooltip: 'Developer tools and API integration help'
  },
  carecredit: {
    name: 'CareCredit',
    icon: 'fa-heart-pulse',
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