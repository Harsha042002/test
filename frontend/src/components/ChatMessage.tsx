import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Message } from '../types';
import { BusCard } from './Buscard';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className="p-2"> 
      <div className={`flex gap-4 py-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && !message.busRoutes && (
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center overflow-hidden">
              <img
                src="/src/assets/aiimg.png"
                alt="AI Icon"
                className="w-5 h-5 object-contain"
                style={{ transform: 'scale(1.5)' }}
              />
            </div>
          </div>
        )}
        <div className={`flex-1 space-y-2 ${isUser ? 'text-right' : 'text-left'}`}>
          <div className="flex items-center gap-2 justify-between">
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {isUser 
                ? 'You'
                : message.busRoutes 
                  ? (<><span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI</>)
                  : (<><span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI</>)}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {format(message.timestamp, 'h:mm a')}
            </span>
          </div>
          <div className="prose dark:prose-invert max-w-none">
            {message.busRoutes ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {message.busRoutes.map(route => (
                  <BusCard
                    key={route.id}
                    {...route}
                    onSeatSelect={() => {}}
                  />
                ))}
              </div>
            ) : (
              <ReactMarkdown>{message.content}</ReactMarkdown>
            )}
          </div>
        </div>
        {isUser && (
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-200">You</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// For example, in App.tsx:
<div className="flex-1 text-center font-semibold text-xl">
  <span className="text-[var(--color-logo-light)]">Ṧ</span>.AI
</div>