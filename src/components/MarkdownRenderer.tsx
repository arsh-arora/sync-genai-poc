import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import 'highlight.js/styles/github.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  // Only process redacted text - preserve all markdown formatting
  const processedContent = content.replace(/\[REDACTED[^\]]*\]/g, '████████');

  return (
    <div className={`prose prose-slate prose-lg max-w-none ${className}`}>
      <ReactMarkdown
        rehypePlugins={[rehypeHighlight]}
        remarkPlugins={[remarkGfm]}
        components={{
          // Enhance headings with better styling
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold text-slate-800 mb-4 pb-2 border-b border-slate-200">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl font-semibold text-slate-800 mb-3 mt-6">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg font-semibold text-slate-700 mb-2 mt-5">
              {children}
            </h3>
          ),
          // Enhanced lists with better spacing
          ul: ({ children }) => (
            <ul className="space-y-2 my-4 ml-6 list-disc marker:text-teal-600">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="space-y-2 my-4 ml-6 list-decimal marker:text-teal-600 marker:font-medium">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-slate-700 leading-relaxed pl-1">
              {children}
            </li>
          ),
          // Enhanced blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-teal-500 bg-teal-50 px-4 py-3 my-4 rounded-r-lg">
              <div className="text-slate-700 italic">
                {children}
              </div>
            </blockquote>
          ),
          // Enhanced strong/bold text
          strong: ({ children }) => (
            <strong className="font-bold text-slate-800">
              {children}
            </strong>
          ),
          // Enhanced code blocks
          code: ({ children, className, ...props }) => {
            const inline = !className;
            if (inline) {
              return (
                <code className="px-2 py-1 bg-slate-100 text-slate-800 rounded text-sm font-mono">
                  {children}
                </code>
              );
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          // Enhanced paragraphs
          p: ({ children }) => (
            <p className="text-slate-700 leading-relaxed my-3">
              {children}
            </p>
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;