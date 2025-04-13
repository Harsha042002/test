// src/components/PaymentModal.tsx
import React, { useState } from 'react';
import { useTheme } from '../context/ThemeContext';

interface PaymentModalProps {
  paymentUrl: string;
  onClose: () => void;
}

const PaymentModal: React.FC<PaymentModalProps> = ({ paymentUrl, onClose }) => {
  useTheme(); // Call useTheme to ensure the context is used if needed
  const [isLoading, setIsLoading] = useState(true);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl h-[80vh] flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Complete Payment
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            ×
          </button>
        </div>
        <div className="flex-1 relative">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white dark:bg-gray-800">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}
          <iframe
            src={paymentUrl}
            className="w-full h-full"
            onLoad={() => setIsLoading(false)}
            title="Payment Gateway"
          />
        </div>
      </div>
    </div>
  );
};

export default PaymentModal;