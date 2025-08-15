import React, { useState, useEffect } from 'react';
import { AgentTrace, AgentExecution, ToolCall } from '../types';
import { AGENTS } from '../config/agents';

interface AgentTheaterProps {
  trace: AgentTrace;
  isVisible: boolean;
  compact?: boolean;
  onExpandToPanel?: () => void;
}

const AgentTheater: React.FC<AgentTheaterProps> = ({ 
  trace, 
  isVisible, 
  compact = false,
  onExpandToPanel 
}) => {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [animationStage, setAnimationStage] = useState<number>(0);

  // Animation effect to show agents appearing in sequence
  useEffect(() => {
    if (!isVisible) {
      setAnimationStage(0);
      return;
    }

    const totalAgents = trace.agent_executions.length;
    let currentStage = 0;

    const interval = setInterval(() => {
      currentStage++;
      setAnimationStage(currentStage);
      
      if (currentStage >= totalAgents) {
        clearInterval(interval);
      }
    }, 200); // Faster animation for compact view

    return () => clearInterval(interval);
  }, [isVisible, trace.agent_executions.length]);

  if (!isVisible || !trace.agent_executions.length) return null;

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'success': return 'text-emerald-500';
      case 'error': return 'text-red-500';
      case 'timeout': return 'text-amber-500';
      default: return 'text-slate-400';
    }
  };

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'success': return 'fa-check-circle';
      case 'error': return 'fa-exclamation-circle';
      case 'timeout': return 'fa-clock';
      default: return 'fa-circle';
    }
  };

  // Compact inline version
  if (compact) {
    return (
      <div className="agent-theater-compact mb-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
              <i className="fas fa-route text-white text-xs"></i>
            </div>
            <span className="text-sm font-medium text-slate-700">Multi-Agent Execution</span>
            <span className="text-xs text-slate-500">
              ({formatDuration(trace.total_duration_ms)})
            </span>
          </div>
          {onExpandToPanel && (
            <button
              onClick={onExpandToPanel}
              className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
            >
              View Details →
            </button>
          )}
        </div>

        {/* Agent Flow - Compact horizontal layout */}
        <div className="flex items-center space-x-1 overflow-x-auto pb-1">
          {trace.agent_executions.map((execution, index) => {
            const agent = AGENTS[execution.agent_name];
            const isVisible = index < animationStage;

            return (
              <React.Fragment key={execution.agent_name}>
                {/* Agent Badge */}
                <div
                  className={`
                    flex items-center space-x-1 px-2 py-1 rounded-full text-xs
                    transition-all duration-300 bg-white border
                    ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-2'}
                    ${execution.status === 'success' ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200'}
                  `}
                  title={`${execution.display_name}: ${formatDuration(execution.duration_ms)}`}
                >
                  <i className={`fas ${agent?.icon || 'fa-robot'} ${agent?.color || 'text-slate-500'} text-xs`}></i>
                  <span className="font-medium text-slate-700">{execution.display_name}</span>
                  <span className="text-slate-500">{formatDuration(execution.duration_ms)}</span>
                  <i className={`fas ${getStatusIcon(execution.status)} ${getStatusColor(execution.status)} text-xs`}></i>
                  
                  {/* Tool call indicator */}
                  {execution.tool_calls.length > 0 && (
                    <div className="w-4 h-4 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">
                      {execution.tool_calls.length}
                    </div>
                  )}
                </div>

                {/* Arrow between agents */}
                {index < trace.agent_executions.length - 1 && (
                  <i className={`
                    fas fa-arrow-right text-slate-300 text-xs
                    transition-opacity duration-300
                    ${index < animationStage - 1 ? 'opacity-100' : 'opacity-0'}
                  `}></i>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    );
  }

  // Full detailed version (for right panel)
  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
            <i className="fas fa-route text-white text-xs"></i>
          </div>
          <div>
            <h3 className="text-sm font-medium text-slate-800">Execution Trace</h3>
            <p className="text-xs text-slate-500">{trace.agent_executions.length} agents • {formatDuration(trace.total_duration_ms)}</p>
          </div>
        </div>
      </div>

      {/* Compact Agent Flow */}
      <div className="space-y-1">
        {trace.agent_executions.map((execution, index) => {
          const agent = AGENTS[execution.agent_name];
          const isVisible = index < animationStage;
          const isExpanded = expandedAgent === execution.agent_name;

          return (
            <div
              key={execution.agent_name}
              className={`transition-all duration-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}
            >
              {/* Compact Agent Row */}
              <div
                className={`
                  flex items-center space-x-2 p-2 rounded-lg cursor-pointer transition-all duration-200
                  ${isExpanded ? 'bg-blue-50 border border-blue-200' : 'bg-slate-50 hover:bg-slate-100'}
                `}
                onClick={() => setExpandedAgent(isExpanded ? null : execution.agent_name)}
              >
                {/* Agent Avatar */}
                <div className={`
                  w-6 h-6 rounded-md flex items-center justify-center relative flex-shrink-0
                  ${agent?.color.replace('text-', 'bg-').replace('-600', '-100') || 'bg-slate-100'}
                `}>
                  <i className={`fas ${agent?.icon || 'fa-robot'} ${agent?.color || 'text-slate-600'} text-xs`}></i>
                  <div className={`
                    absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full
                    ${execution.status === 'success' ? 'bg-emerald-500' : 'bg-slate-400'}
                  `}></div>
                </div>

                {/* Agent Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-800 truncate">{execution.display_name}</span>
                    <div className="flex items-center space-x-1 text-xs text-slate-500 flex-shrink-0 ml-2">
                      <span>{formatDuration(execution.duration_ms)}</span>
                      {execution.tool_calls.length > 0 && (
                        <div className="w-4 h-4 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">
                          {execution.tool_calls.length}
                        </div>
                      )}
                      <i className={`fas fa-chevron-${isExpanded ? 'up' : 'down'} text-xs`}></i>
                    </div>
                  </div>
                  
                  {execution.input_summary && (
                    <p className="text-xs text-slate-500 truncate mt-0.5">{execution.input_summary}</p>
                  )}
                </div>
              </div>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="mt-1 ml-8 p-2 bg-white rounded border border-slate-200 animate-slideDown">
                  {/* Tool Calls - prioritized since they're most important */}
                  {execution.tool_calls.length > 0 && (
                    <div className="mb-2">
                      <div className="text-xs font-medium text-slate-700 mb-1">Tools:</div>
                      <div className="space-y-1">
                        {execution.tool_calls.map((toolCall, idx) => (
                          <div key={idx} className="flex items-center justify-between p-1.5 bg-slate-50 rounded text-xs">
                            <div className="flex items-center space-x-1.5 flex-1 min-w-0">
                              <i className={`fas ${getStatusIcon(toolCall.status)} ${getStatusColor(toolCall.status)} flex-shrink-0`}></i>
                              <span className="font-medium truncate">{toolCall.tool_name}</span>
                            </div>
                            <span className="text-slate-500 flex-shrink-0">{formatDuration(toolCall.duration_ms)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Input/Output - more concise */}
                  {execution.input_summary && (
                    <div className="mb-2">
                      <div className="text-xs font-medium text-slate-700 mb-0.5">Input:</div>
                      <div className="text-xs text-slate-600 bg-slate-50 p-1.5 rounded break-words">{execution.input_summary}</div>
                    </div>
                  )}

                  {execution.output_summary && (
                    <div className="mb-2">
                      <div className="text-xs font-medium text-slate-700 mb-0.5">Output:</div>
                      <div className="text-xs text-slate-600 bg-slate-50 p-1.5 rounded break-words">{execution.output_summary}</div>
                    </div>
                  )}

                  {/* Confidence bar */}
                  {execution.confidence !== undefined && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium text-slate-700">Confidence:</span>
                      <div className="flex-1 bg-slate-200 rounded-full h-1">
                        <div 
                          className="bg-blue-500 h-1 rounded-full transition-all duration-500"
                          style={{ width: `${execution.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-slate-600">{(execution.confidence * 100).toFixed(0)}%</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Routing Info */}
      {trace.routing_decision && (
        <div className="mt-2 p-2 bg-blue-50 rounded border border-blue-200">
          <div className="flex items-center space-x-1 text-xs">
            <i className="fas fa-route text-blue-600"></i>
            <span className="font-medium text-blue-800">
              Routed to <strong>{trace.routing_decision.primary_agent}</strong>
            </span>
            {trace.routing_decision.confidence && (
              <span className="text-blue-700">
                ({(trace.routing_decision.confidence * 100).toFixed(0)}%)
              </span>
            )}
          </div>
          {trace.routing_decision.fallback_used && (
            <div className="mt-0.5 text-xs text-blue-700">
              Fallback: {trace.routing_decision.fallback_used}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AgentTheater;