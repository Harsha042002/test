import 'core-js/stable';
import 'regenerator-runtime/runtime';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './global.css';
import './index.css';
import { LoginModalProvider } from './context/loginModalContext';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LoginModalProvider>
      <App />
    </LoginModalProvider>
  </StrictMode>
);
