import { useLoginModal } from '../context/loginModalContext';
import LoginForm from './LoginForm';
import { useTheme } from '../context/ThemeContext';

export default function LoginModal() {
    const { isOpen, onClose } = useLoginModal();
    const { theme } = useTheme();

    if (!isOpen) return null;

    return (
        <>
            {/* Overlay */}
            <div
                className="fixed inset-0 bg-black bg-opacity-50 z-[100]"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="fixed inset-0 z-[101] flex items-center justify-center p-4">
                <div className={`rounded-lg shadow-xl w-full max-w-md relative ${
                    theme === 'dark' 
                        ? 'bg-[#121212] border border-[#1e1e1e]' 
                        : 'bg-white border border-gray-200'
                }`}>
                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className={`absolute top-4 right-4 p-2 rounded-lg transition-colors duration-200 ${
                            theme === 'dark'
                                ? 'hover:bg-[#1e1e1e] text-gray-400 hover:text-gray-300'
                                : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
                        }`}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-6 w-6"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M6 18L18 6M6 6l12 12"
                            />
                        </svg>
                    </button>

                    {/* Content */}
                    <div className="p-6">
                        <h2 className={`text-2xl font-bold mb-6 text-center ${
                            theme === 'dark' ? 'text-[#e5e5e5]' : 'text-[#333333]'
                        }`}>Login</h2>
                        <LoginForm />
                    </div>
                </div>
            </div>
        </>
    );
}
