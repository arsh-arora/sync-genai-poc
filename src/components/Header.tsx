import React from 'react';

interface HeaderProps {
  allowTavily: boolean;
  setAllowTavily: (value: boolean) => void;
  allowLlmKnowledge: boolean;
  setAllowLlmKnowledge: (value: boolean) => void;
  allowWebSearch: boolean;
  setAllowWebSearch: (value: boolean) => void;
  onClear: () => void;
  onExport: () => void;
}

const Header: React.FC<HeaderProps> = ({
  allowTavily,
  setAllowTavily,
  allowLlmKnowledge,
  setAllowLlmKnowledge,
  allowWebSearch,
  setAllowWebSearch,
  onClear,
  onExport
}) => {
  return (
    <header className="bg-white border-b border-slate-200 px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <h1 className="text-xl font-semibold text-slate-800">
            GenAI Studio <span className="text-sm font-normal text-slate-500">(Hackathon PoC)</span>
          </h1>
          <div className="flex items-center space-x-4">
            <span className="px-2 py-1 text-xs bg-slate-100 text-slate-600 rounded-md">
              Local • No DB
            </span>
            
            {/* Elegant Toggle Switches */}
            <div className="flex items-center space-x-6 ml-4">
              {/* Tavily Toggle */}
              <div className="flex items-center space-x-2">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={allowTavily}
                    onChange={(e) => setAllowTavily(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-teal-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-teal-500"></div>
                </label>
                <span className="text-xs text-slate-600 font-medium">
                  Tavily <span className="text-slate-400">(legacy)</span>
                </span>
              </div>

              {/* LLM Knowledge Toggle */}
              <div className="flex items-center space-x-2">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={allowLlmKnowledge}
                    onChange={(e) => setAllowLlmKnowledge(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500"></div>
                </label>
                <span className="text-xs text-slate-600 font-medium flex items-center space-x-1">
                  <i className="fas fa-brain text-blue-500"></i>
                  <span>LLM Knowledge</span>
                </span>
              </div>

              {/* Web Search Toggle */}
              <div className="flex items-center space-x-2">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={allowWebSearch}
                    onChange={(e) => setAllowWebSearch(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
                </label>
                <span className="text-xs text-slate-600 font-medium flex items-center space-x-1">
                  <i className="fas fa-search text-purple-500"></i>
                  <span>Web Search</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <span className="px-2 py-1 text-xs bg-green-100 text-green-600 rounded-md">
            PII Redaction ON
          </span>
          <button
            onClick={onClear}
            className="px-3 py-1 text-sm text-slate-600 hover:text-slate-800"
          >
            Clear
          </button>
          <button
            onClick={onExport}
            className="px-3 py-1 text-sm text-slate-600 hover:text-slate-800"
          >
            Export
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;