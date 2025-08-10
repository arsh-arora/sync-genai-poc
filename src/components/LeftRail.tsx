import React from 'react';
import { AGENTS } from '../config/agents';

interface LeftRailProps {
  selectedAgent: string;
  onSelectAgent: (agent: string) => void;
  onUseExample: (agent: string) => void;
}

interface AgentItemProps {
  agentKey: string;
  agent: any;
  isSelected: boolean;
  onSelect: () => void;
  onUseExample: () => void;
}

function AgentItem({ agentKey, agent, isSelected, onSelect, onUseExample }: AgentItemProps) {
  return (
    <div 
      className={`p-4 border-b border-slate-100 cursor-pointer hover:bg-slate-50 transition-colors ${
        isSelected ? 'bg-teal-50 border-r-4 border-teal-500' : ''
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <i className={`fas ${agent.icon} ${agent.color} text-lg`}></i>
          <div className="flex-1">
            <div className="font-semibold text-base text-slate-800 mb-1">
              {agent.name}
            </div>
            <div className="text-sm text-slate-600 leading-relaxed">
              {agent.tooltip}
            </div>
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onUseExample();
          }}
          className="text-sm text-teal-600 hover:text-teal-700 p-2 hover:bg-teal-100 rounded-lg transition-colors"
          title="Try Example"
        >
          <i className="fas fa-play"></i>
        </button>
      </div>
    </div>
  );
}

const LeftRail: React.FC<LeftRailProps> = ({ selectedAgent, onSelectAgent, onUseExample }) => {
  return (
    <div className="w-72 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-6 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Agents</h2>
        <p className="text-sm text-slate-500 mt-1">Choose your AI specialist</p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {Object.entries(AGENTS).map(([key, agent]) => (
          <AgentItem
            key={key}
            agentKey={key}
            agent={agent}
            isSelected={selectedAgent === key}
            onSelect={() => onSelectAgent(key)}
            onUseExample={() => onUseExample(key)}
          />
        ))}
      </div>
    </div>
  );
};

export default LeftRail;