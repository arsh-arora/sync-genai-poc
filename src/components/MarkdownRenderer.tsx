import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import 'highlight.js/styles/github-dark.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  // Process content to handle redacted text
  const processedContent = content.replace(/\[REDACTED[^\]]*\]/g, '████████');

  const components = {
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      
      return !inline ? (
        <div className="my-4">
          {language && (
            <div className="bg-slate-100 px-3 py-1 text-xs font-mono text-slate-600 border-t border-l border-r border-slate-300 rounded-t">
              {language}
            </div>
          )}
          <pre className={`bg-slate-900 text-slate-100 p-4 rounded${language ? '-b' : ''} overflow-x-auto`}>
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        </div>
      ) : (
        <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
          {children}
        </code>
      );
    },
    h1: ({ children }: any) => <h1 className="text-2xl font-bold text-slate-800 mt-8 mb-4">{children}</h1>,
    h2: ({ children }: any) => <h2 className="text-xl font-bold text-slate-800 mt-6 mb-3">{children}</h2>,
    h3: ({ children }: any) => <h3 className="text-lg font-semibold text-slate-800 mt-4 mb-2">{children}</h3>,
    p: ({ children }: any) => <p className="mb-3 leading-relaxed">{children}</p>,
    ul: ({ children }: any) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
    ol: ({ children }: any) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
    li: ({ children }: any) => <li className="ml-2">{children}</li>,
    strong: ({ children }: any) => <strong className="font-semibold text-slate-800">{children}</strong>,
    em: ({ children }: any) => <em className="italic">{children}</em>,
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-slate-300 pl-4 italic text-slate-600 mb-3">
        {children}
      </blockquote>
    ),
    table: ({ children }: any) => (
      <div className="overflow-x-auto mb-4">
        <table className="min-w-full border border-slate-300">{children}</table>
      </div>
    ),
    th: ({ children }: any) => (
      <th className="border border-slate-300 bg-slate-100 px-4 py-2 text-left font-semibold">{children}</th>
    ),
    td: ({ children }: any) => (
      <td className="border border-slate-300 px-4 py-2">{children}</td>
    )
  };

  return (
    <div className={`prose prose-sm max-w-none text-slate-700 markdown-content ${className}`}>
      <ReactMarkdown
        components={components}
        rehypePlugins={[rehypeHighlight]}
        remarkPlugins={[remarkGfm]}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;