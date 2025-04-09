import { useState, useEffect } from 'react';

export const useAuth = () => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        const checkAuth = () => {
            const token = localStorage.getItem('auth_token');
            setIsAuthenticated(!!token); // Update state based on token presence
        };

        // Check immediately
        checkAuth();

        // Listen for storage changes
        window.addEventListener('storage', checkAuth);

        return () => {
            window.removeEventListener('storage', checkAuth);
        };
    }, []);

    return isAuthenticated;
};
