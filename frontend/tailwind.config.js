/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
                './src/**/*.{js,jsx,ts,tsx}', // Ensure all files are included
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                dark: {
                    bg: '#1a1b1e',
                    surface: '#2d2d2d',
                    text: '#e5e5e5',
                    border: 'rgba(255, 255, 255, 0.1)',
                    hover: 'rgba(255, 255, 255, 0.05)',
                },
            },
            typography: {
                DEFAULT: {
                    css: {
                        maxWidth: 'none',
                        code: {
                            backgroundColor: '#f3f4f6',
                            padding: '2px 4px',
                            borderRadius: '4px',
                            fontSize: '0.875em',
                        },
                        'code::before': {
                            content: '""',
                        },
                        'code::after': {
                            content: '""',
                        },
                    },
                },
            },
animation: {
                'dots': 'dots 1.2s infinite',
            },
            keyframes: {
                dots: {
                    '0%, 80%, 100%': { transform: 'scale(0)' },
                    '40%': { transform: 'scale(1)' },
                },
            },
        },
    },
    plugins: [],
};