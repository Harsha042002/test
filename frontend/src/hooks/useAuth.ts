import { useState, useEffect } from 'react';

export const useAuth = () => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        const checkAuth = () => {
            const token = localStorage.getItem('access_token');
            const user = localStorage.getItem('user');
            setIsAuthenticated(!!token && !!user); // Update state based on token and user presence
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
