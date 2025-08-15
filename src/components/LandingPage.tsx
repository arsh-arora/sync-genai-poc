import React from 'react';

interface LandingPageProps {
  onUserTypeSelect: (userType: 'consumer' | 'partner') => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onUserTypeSelect }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Welcome to Synch GenAI
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            AI-powered financial services platform
          </p>
          <p className="text-lg text-gray-500">
            Choose your role to get started with the right set of tools
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto">
          {/* Consumer Option */}
          <div 
            onClick={() => onUserTypeSelect('consumer')}
            className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer border-2 border-transparent hover:border-blue-500 group"
          >
            <div className="text-center">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-blue-200 transition-colors">
                <i className="fas fa-user text-3xl text-blue-600"></i>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Consumer</h2>
              <p className="text-gray-600 mb-6">
                Access personal financial services, manage your accounts, and get help with transactions
              </p>
              <div className="space-y-2 text-sm text-gray-500">
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Account Management
                </div>
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Transaction Support
                </div>
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Financial Products
                </div>
              </div>
            </div>
          </div>

          {/* Partner Option */}
          <div 
            onClick={() => onUserTypeSelect('partner')}
            className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer border-2 border-transparent hover:border-purple-500 group"
          >
            <div className="text-center">
              <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-purple-200 transition-colors">
                <i className="fas fa-handshake text-3xl text-purple-600"></i>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Partner</h2>
              <p className="text-gray-600 mb-6">
                Access business tools, analytics, and partner-specific services for your organization
              </p>
              <div className="space-y-2 text-sm text-gray-500">
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Business Analytics
                </div>
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Developer Tools
                </div>
                <div className="flex items-center justify-center">
                  <i className="fas fa-check text-green-500 mr-2"></i>
                  Portfolio Insights
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-12">
          <p className="text-sm text-gray-400">
            Powered by AI • Secure • Real-time
          </p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;