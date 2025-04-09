/** @type {import('tailwindcss').Config} */
export default {
    content: [
        './index.html',
        './src/**/*.{js,ts,jsx,tsx}',
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
        },
    },
    plugins: [],
};