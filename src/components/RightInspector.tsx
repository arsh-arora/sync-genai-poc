import React, { useState, useEffect, useRef } from 'react';
import { UploadedPdf, AgentTrace } from '../types';
import AgentTheater from './AgentTheater';

interface RightInspectorProps {
  activePanel: string;
  setActivePanel: (panel: string) => void;
  citations: string[];
  toolTrace: any[];
  agentTrace: AgentTrace | null;
  uploadedPdfs: UploadedPdf[];
  selectedPdfChunk: any;
  setSelectedPdfChunk: (chunk: any) => void;
}

const RightInspector: React.FC<RightInspectorProps> = ({
  activePanel,
  setActivePanel,
  citations,
  toolTrace,
  agentTrace,
  uploadedPdfs,
  selectedPdfChunk,
  setSelectedPdfChunk
}) => {
  return (
    <div className="w-80 bg-white border-l border-slate-200 flex flex-col">
      {/* Tabs */}
      <div className="border-b border-slate-200">
        <div className="flex">
          <button
            onClick={() => setActivePanel('citations')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 ${
              activePanel === 'citations'
                ? 'text-teal-600 border-teal-500'
                : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <i className="fas fa-quote-left mr-2"></i>
            Citations
          </button>
          <button
            onClick={() => setActivePanel('tools')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 relative ${
              activePanel === 'tools'
                ? 'text-teal-600 border-teal-500'
                : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <i className="fas fa-route mr-2"></i>
            Agent Theater
            {agentTrace && agentTrace.agent_executions?.length > 1 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-purple-500 text-white text-xs rounded-full flex items-center justify-center">
                {agentTrace.agent_executions.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActivePanel('pdfs')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 relative ${
              activePanel === 'pdfs'
                ? 'text-teal-600 border-teal-500'
                : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <i className="fas fa-file-pdf mr-2"></i>
            Doc Viewer
            {uploadedPdfs.length > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                {uploadedPdfs.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activePanel === 'citations' && (
          <div className="p-6">
            <h3 className="text-lg font-semibold text-slate-800 mb-6">Sources & Citations</h3>
            {citations.length > 0 ? (
              <div className="space-y-4">
                {citations.map((citation, index) => (
                  <div key={index} className="p-4 bg-slate-50 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors">
                    <div className="flex items-start space-x-3">
                      <span className="flex-shrink-0 w-7 h-7 bg-teal-100 text-teal-700 rounded-full text-sm font-semibold flex items-center justify-center">
                        {index + 1}
                      </span>
                      <div className="text-base text-slate-700 leading-relaxed font-medium">
                        {citation}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <i className="fas fa-quote-left text-4xl text-slate-300 mb-4"></i>
                <p className="text-slate-500 text-base font-medium">No citations yet</p>
                <p className="text-sm text-slate-400 mt-2">Citations will appear here when you chat with agents</p>
              </div>
            )}
          </div>
        )}

        {activePanel === 'tools' && (
          <div className="p-4">
            {agentTrace && agentTrace.agent_executions?.length > 0 ? (
              <AgentTheater 
                trace={agentTrace} 
                isVisible={true} 
                compact={false}
              />
            ) : (
              <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-purple-100 to-blue-100 rounded-full flex items-center justify-center">
                  <i className="fas fa-route text-2xl text-purple-600"></i>
                </div>
                <p className="text-slate-500 text-base font-medium">No agent executions yet</p>
                <p className="text-sm text-slate-400 mt-2">Multi-agent execution traces will appear here when multiple agents collaborate</p>
              </div>
            )}
          </div>
        )}

        {activePanel === 'pdfs' && (
          <DocViewerPanel 
            uploadedPdfs={uploadedPdfs}
            selectedPdfChunk={selectedPdfChunk}
            setSelectedPdfChunk={setSelectedPdfChunk}
          />
        )}
      </div>
    </div>
  );
};

// Doc Viewer Panel Component
function DocViewerPanel({ uploadedPdfs, selectedPdfChunk, setSelectedPdfChunk }: {
  uploadedPdfs: UploadedPdf[];
  selectedPdfChunk: any;
  setSelectedPdfChunk: (chunk: any) => void;
}) {
  const [selectedPdf, setSelectedPdf] = useState<UploadedPdf | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [pageData, setPageData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [imgLoaded, setImgLoaded] = useState<boolean | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (selectedPdf && currentPage !== null) {
      loadPageData();
    }
  }, [selectedPdf, currentPage]);

  // Auto-select PDF when chunk is selected from citations
  useEffect(() => {
    if (selectedPdfChunk && uploadedPdfs.length > 0 && !selectedPdf) {
      // Find which PDF contains this chunk
      const findPdfWithChunk = async () => {
        for (const pdf of uploadedPdfs) {
          try {
            const response = await fetch(`/pdf/${pdf.pdf_id}/info`);
            const pdfInfo = await response.json();
            
            // Check if this PDF has the selected chunk
            const hasChunk = pdfInfo.chunks && pdfInfo.chunks.some((chunk: any) => 
              chunk.chunk_id === selectedPdfChunk
            );
            
            if (hasChunk) {
              setSelectedPdf(pdf);
              
              // Find the page containing this chunk
              const matchingChunk = pdfInfo.chunks.find((chunk: any) => 
                chunk.chunk_id === selectedPdfChunk
              );
              
              if (matchingChunk) {
                setCurrentPage(matchingChunk.page_number || 0);
              }
              break;
            }
          } catch (error) {
            console.error(`Error checking PDF ${pdf.filename}:`, error);
          }
        }
      };
      
      findPdfWithChunk();
    }
  }, [selectedPdfChunk, uploadedPdfs, selectedPdf]);

  const loadPageData = async () => {
    if (!selectedPdf) return;
    
    setLoading(true);
    try {
      const response = await fetch(`/pdf/${selectedPdf.pdf_id}/page/${currentPage}`);
      const data = await response.json();
      setPageData(data);
    } catch (error) {
      console.error('Error loading page data:', error);
      setPageData(null);
    }
    setLoading(false);
  };

  const handlePdfSelect = (pdf: UploadedPdf) => {
    setSelectedPdf(pdf);
    setCurrentPage(0);
    setPageData(null);
  };

  const nextPage = () => {
    if (selectedPdf && currentPage < selectedPdf.total_pages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  const prevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  return (
    <div className="p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-6">Document Viewer</h3>
      
      {uploadedPdfs.length === 0 ? (
        <div className="text-center py-12">
          <i className="fas fa-file-pdf text-4xl text-slate-300 mb-4"></i>
          <p className="text-slate-500 text-base font-medium">No PDFs uploaded</p>
          <p className="text-sm text-slate-400 mt-2">Drag & drop PDF files to analyze them</p>
        </div>
      ) : !selectedPdf ? (
        <div className="space-y-4">
          {uploadedPdfs.map((pdf) => (
            <div 
              key={pdf.pdf_id} 
              className="p-4 bg-slate-50 rounded-xl border border-slate-200 cursor-pointer hover:bg-slate-100 hover:border-slate-300 transition-all"
              onClick={() => handlePdfSelect(pdf)}
            >
              <div className="flex items-center space-x-4">
                <i className="fas fa-file-pdf text-red-500 text-xl"></i>
                <div className="flex-1">
                  <div className="font-semibold text-base text-slate-800 truncate mb-2">
                    {pdf.filename}
                  </div>
                  <div className="text-sm text-slate-600 space-y-1">
                    <div className="flex items-center space-x-4">
                      <span>📄 {pdf.total_pages} pages</span>
                      <span>🧩 {pdf.chunks} chunks</span>
                      <span>⚡ {pdf.processing_time.toFixed(2)}s</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-6">
          {/* PDF Header */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSelectedPdf(null)}
              className="text-teal-600 hover:text-teal-700 text-sm font-medium flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-teal-50 transition-colors"
            >
              <i className="fas fa-arrow-left"></i>
              <span>Back to list</span>
            </button>
            <div className="text-sm text-slate-600 font-medium bg-slate-100 px-3 py-1 rounded-full">
              {selectedPdf.filename}
            </div>
          </div>

          {/* Page Navigation */}
          <div className="flex items-center justify-between py-3 px-4 bg-slate-50 rounded-xl border border-slate-200">
            <button
              onClick={prevPage}
              disabled={currentPage === 0}
              className="p-2 text-slate-600 hover:text-slate-800 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white rounded-lg transition-colors"
            >
              <i className="fas fa-chevron-left text-base"></i>
            </button>
            <span className="text-base font-medium text-slate-700">
              Page {currentPage + 1} of {selectedPdf.total_pages}
            </span>
            <button
              onClick={nextPage}
              disabled={currentPage >= selectedPdf.total_pages - 1}
              className="p-2 text-slate-600 hover:text-slate-800 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white rounded-lg transition-colors"
            >
              <i className="fas fa-chevron-right text-base"></i>
            </button>
          </div>

          {/* PDF Page Viewer */}
          <div className="relative">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin h-6 w-6 border-2 border-teal-500 border-t-transparent rounded-full"></div>
              </div>
            ) : pageData ? (
              <div className="relative border border-slate-200 rounded overflow-hidden">
                <img
                  ref={imgRef}
                  src={`data:image/png;base64,${pageData.image_base64}`}
                  alt={`Page ${currentPage + 1}`}
                  className="w-full h-auto"
                  onLoad={() => setImgLoaded(true)}
                  onError={() => setImgLoaded(false)}
                />
                
                {/* Chunk overlays */}
                {imgLoaded && pageData.chunks && imgRef.current && (() => {
                  const img = imgRef.current;
                  const scaleX = img.offsetWidth / img.naturalWidth;
                  const scaleY = img.offsetHeight / img.naturalHeight;
                  
                  return pageData.chunks.map((chunk: any, index: number) => {
                    if (!chunk.bbox) return null;
                    
                    return (
                      <div
                        key={chunk.chunk_id}
                        className={`absolute border-2 cursor-pointer transition-all ${
                          selectedPdfChunk === chunk.chunk_id 
                            ? 'border-teal-500 bg-teal-500 bg-opacity-20' 
                            : 'border-blue-500 bg-blue-500 bg-opacity-10 hover:bg-opacity-20'
                        }`}
                        style={{
                          left: `${chunk.bbox.x * scaleX}px`,
                          top: `${chunk.bbox.y * scaleY}px`, 
                          width: `${chunk.bbox.width * scaleX}px`,
                          height: `${chunk.bbox.height * scaleY}px`
                        }}
                        onClick={() => setSelectedPdfChunk(
                          selectedPdfChunk === chunk.chunk_id ? null : chunk.chunk_id
                        )}
                        title={chunk.text}
                      >
                        <div className="absolute -top-5 -left-0.5 bg-blue-500 text-white text-xs px-1 rounded">
                          {index + 1}
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8 text-slate-500">
                <i className="fas fa-exclamation-triangle mr-2"></i>
                Failed to load page
              </div>
            )}
          </div>

          {/* Chunk Details */}
          {selectedPdfChunk && pageData && (
            <div className="mt-3 p-3 bg-slate-50 rounded border border-slate-200">
              <h4 className="text-sm font-medium text-slate-800 mb-2">Selected Chunk</h4>
              {(() => {
                const chunk = pageData.chunks.find((c: any) => c.chunk_id === selectedPdfChunk);
                return chunk ? (
                  <div>
                    <div className="text-xs text-slate-600 mb-1">
                      ID: {chunk.chunk_id} • Confidence: {(chunk.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-slate-700 max-h-32 overflow-y-auto">
                      {chunk.text}
                    </div>
                  </div>
                ) : null;
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RightInspector;