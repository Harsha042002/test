import { useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { useLoginModal } from '../context/loginModalContext';
import { toast } from 'react-toastify';
import { authService } from '../services/api';

export default function LoginForm() {
    const [mobileNumber, setMobileNumber] = useState('');
    const [otp, setOtp] = useState('');
    const [isOtpSent, setIsOtpSent] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isResending, setIsResending] = useState(false); // State for resend OTP button
    const { theme } = useTheme();
    const { onClose } = useLoginModal();

    const handleSendOTP = async () => {
        if (!mobileNumber || mobileNumber.length !== 10) {
            toast.error('Please enter a valid 10-digit mobile number');
            return;
        }

        setIsLoading(true);
        try {
            const result = await authService.sendOTP(mobileNumber);
            setIsOtpSent(true);
            toast.success(result.message);
        } catch (error: any) {
            toast.error(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleResendOTP = async () => {
        setIsResending(true);
        try {
            await authService.sendOTP(mobileNumber); // Reuse the sendOTP API
            toast.success('OTP resent successfully!');
        } catch (error: any) {
            toast.error(error.message || 'Failed to resend OTP');
        } finally {
            setIsResending(false);
        }
    };

    const handleVerifyOTP = async () => {
        if (!otp || otp.length !== 6) {
            toast.error('Please enter a valid 6-digit OTP');
            return;
        }

        setIsLoading(true);
        try {
            await authService.verifyOTP(mobileNumber, otp);
            toast.success('Login successful!');
            onClose(); // Close modal after login
            window.dispatchEvent(new Event('storage')); // Trigger auth state update
        } catch (error: any) {
            toast.error(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            if (!isOtpSent) {
                handleSendOTP();
            } else {
                handleVerifyOTP();
            }
        }
    };

    return (
        <form className="space-y-6" onKeyDown={handleKeyDown}>
            {!isOtpSent ? (
                <>
                    <div>
                        <label htmlFor="mobile" className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                            Mobile Number
                        </label>
                        <input
                            type="tel"
                            id="mobile"
                            value={mobileNumber}
                            onChange={(e) => setMobileNumber(e.target.value.replace(/\D/g, '').slice(0, 10))}
                            placeholder="Enter 10-digit mobile number"
                            className={`w-full px-4 py-2 rounded-lg border ${theme === 'dark' ? 'bg-[#1e1e1e] text-white' : 'bg-[#f3f4f6] text-gray-900'}`}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={handleSendOTP}
                        disabled={isLoading || mobileNumber.length !== 10}
                        className={`w-full py-2 px-4 rounded-lg ${theme === 'dark' ? 'bg-[#FBE822] text-[#1765F3]' : 'bg-[#1765F3] text-[#FBE822]'}`}
                    >
                        {isLoading ? 'Sending...' : 'Send OTP'}
                    </button>
                </>
            ) : (
                <>
                    <div>
                        <label htmlFor="otp" className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                            Enter OTP
                        </label>
                        <input
                            type="text"
                            id="otp"
                            value={otp}
                            onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                            placeholder="Enter 6-digit OTP"
                            className={`w-full px-4 py-2 rounded-lg border ${theme === 'dark' ? 'bg-[#1e1e1e] text-white' : 'bg-[#f3f4f6] text-gray-900'}`}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={handleVerifyOTP}
                        disabled={isLoading || otp.length !== 6}
                        className={`w-full py-2 px-4 rounded-lg ${theme === 'dark' ? 'bg-[#FBE822] text-[#1765F3]' : 'bg-[#1765F3] text-[#FBE822]'}`}
                    >
                        {isLoading ? 'Verifying...' : 'Verify OTP'}
                    </button>
                    <button
                        type="button"
                        onClick={handleResendOTP}
                        disabled={isResending}
                        className={`w-full mt-2 py-2 px-4 rounded-lg border ${theme === 'dark' ? 'bg-[#1e1e1e] text-white' : 'bg-[#f3f4f6] text-gray-900'}`}
                    >
                        {isResending ? 'Resending...' : 'Resend OTP'}
                    </button>
                </>
            )}
        </form>
    );
}
