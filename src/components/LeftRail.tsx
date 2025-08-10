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
      className={`p-3 border-b border-slate-100 cursor-pointer hover:bg-slate-50 transition-colors ${
        isSelected ? 'bg-teal-50 border-r-2 border-teal-500' : ''
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <i className={`fas ${agent.icon} ${agent.color}`}></i>
          <div>
            <div className="font-medium text-sm text-slate-800">
              {agent.name}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {agent.tooltip}
            </div>
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onUseExample();
          }}
          className="text-xs text-teal-600 hover:text-teal-700 p-1"
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
    <div className="w-64 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <h2 className="font-medium text-slate-800 mb-3">Agents</h2>
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